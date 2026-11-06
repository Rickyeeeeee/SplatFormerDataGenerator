#!/usr/bin/env python3
"""Generate ShapeNet COLMAP renders and gsplat checkpoints."""
import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import threading
import re
import shutil
import subprocess
import sys

import gen_data_gsplat as common

RAW_ROOT = Path('/project/ricky/ShapeNetCore')
IMAGE_ROOT = Path('/project/ricky/shapenet-sr-data')
GS_ROOT = Path('/project/ricky/shapenet-sr-data-mcmc-4096')
DEFAULT_MAX_SPLATS = 4096
SCRIPT_DIR = Path(__file__).resolve().parent
NUM_VIEWS = common.NUM_VIEWS
DEFAULT_TRAIN_STEPS = 15000


def checkpoint_name(train_steps):
    return f'ckpt_{train_steps - 1}_rank0.pt'


def metrics_name(train_steps):
    return f'val_step{train_steps - 1:04d}.json'


def gs_dataset_root(strategy, max_splats):
    return GS_ROOT.parent / f'shapenet-sr-data-{strategy}-{max_splats}'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shapenet-root', type=Path, default=RAW_ROOT)
    p.add_argument('--image-dataset-root', type=Path, default=IMAGE_ROOT)
    p.add_argument('--gs-dataset-root', type=Path, help='Override the strategy- and cap-specific gsplat root')
    p.add_argument('--categories', nargs='+', help='Eight-digit ShapeNet synsets to process')
    p.add_argument('--limit', type=int, help='Maximum number of scenes to process')
    gpu_group = p.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu-id', type=int, help='Use a single physical GPU ID')
    gpu_group.add_argument('--gpu-ids', type=int, nargs='+', help='Physical GPU IDs to use')
    p.add_argument('--scenes-per-gpu', type=int, default=1)
    p.add_argument('--resolutions', type=int, nargs='+', default=[128])
    p.add_argument('--strategy', choices=['mcmc', 'default'], default='mcmc',
                   help='gsplat training strategy (default: mcmc)')
    p.add_argument('--max-splats', type=int, default=DEFAULT_MAX_SPLATS)
    p.add_argument('--train-steps', type=int, default=DEFAULT_TRAIN_STEPS,
                   help='Training iterations (default: 15000)')
    p.add_argument('--skip-gs', action='store_true')
    p.add_argument('--blender', default=str(SCRIPT_DIR / 'blender-3.2.2-linux-x64/blender'))
    args = p.parse_args(argv)
    if not args.shapenet_root.is_dir(): p.error('ShapeNet root does not exist')
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    if args.scenes_per_gpu < 1: p.error('--scenes-per-gpu must be positive')
    if args.gpu_ids and (min(args.gpu_ids) < 0 or len(set(args.gpu_ids)) != len(args.gpu_ids)): p.error('--gpu-ids must be distinct non-negative IDs')
    if args.gpu_id is not None and args.gpu_id < 0: p.error('--gpu-id must be non-negative')
    if args.max_splats < 4: p.error('--max-splats must be at least 4')
    if args.train_steps < 1: p.error('--train-steps must be positive')
    if not args.resolutions or min(args.resolutions) <= 0 or len(set(args.resolutions)) != len(args.resolutions): p.error('resolutions must be distinct positive integers')
    if args.categories and any(not re.fullmatch(r'\d{8}', x) for x in args.categories): p.error('categories must be eight-digit synsets')
    if args.gs_dataset_root is None:
        args.gs_dataset_root = gs_dataset_root(args.strategy, args.max_splats)
    return args


def discover_inventory(root, categories=None):
    """Include model directories even if the expected OBJ is missing."""
    scenes = []
    selected = set(categories) if categories else None
    with os.scandir(root) as entries:
        for category in entries:
            if not re.fullmatch(r'\d{8}', category.name):
                continue
            if selected is not None and category.name not in selected:
                continue
            if not category.is_dir():
                continue
            with os.scandir(category.path) as models:
                scenes.extend(f'{category.name}/{model.name}' for model in models if model.is_dir())
    return sorted(scenes)


def scene_root(root, resolution, kind, scene):
    return root / 'shapenet' / str(resolution) / kind / scene


def render_complete(path):
    """Check only the number of PNG images when resuming generation."""
    try:
        with os.scandir(path / 'images') as entries:
            return sum(1 for entry in entries if entry.name.endswith('.png')) == NUM_VIEWS
    except (FileNotFoundError, NotADirectoryError):
        return False


def render_command(args, mesh, output, resolution):
    cmd = common.build_render_command(mesh, output, resolution)
    cmd[0] = args.blender
    cmd[3] = str(SCRIPT_DIR / 'render_shapenet.py')
    cmd[1:1] = ['--python-exit-code', '1']
    return cmd


class ChildProcesses:
    """Track process groups so interruption stops Blender, trainers and ffmpeg."""
    def __init__(self):
        self.lock = threading.Lock()
        self.processes = set()
        self.stopped = False

    def run(self, command, **kwargs):
        kwargs.pop('check', None)
        with self.lock:
            if self.stopped:
                raise RuntimeError('Generation interrupted')
            process = subprocess.Popen(command, start_new_session=True, **kwargs)
            self.processes.add(process)
        try:
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, command)
        finally:
            with self.lock:
                self.processes.discard(process)

    def stop(self):
        with self.lock:
            self.stopped = True
            processes = list(self.processes)
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


@contextlib.contextmanager
def lock_scene(args, scene):
    # Lock both roots: two runs can share either images or checkpoints.
    roots = {args.image_dataset_root.resolve()}
    if not args.skip_gs:
        roots.add(args.gs_dataset_root.resolve())
    roots = sorted(roots)
    with contextlib.ExitStack() as stack:
        for root in roots:
            path = root / 'shapenet' / '.scene-locks' / (scene + '.lock')
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open('a'))
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f'Scene is locked by another generator: {scene}')
        yield


def process_scene(args, scene, env, run=subprocess.run):
    native = max(args.resolutions)
    native_dir = scene_root(args.image_dataset_root, native, 'colmap', scene)
    if not render_complete(native_dir):
        mesh = args.shapenet_root / scene / 'models/model_normalized.obj'
        if not mesh.is_file(): raise FileNotFoundError(f'Missing mesh: {mesh}')
        shutil.rmtree(native_dir, ignore_errors=True)
        run(render_command(args, mesh, native_dir, native), env=env, check=True, cwd=SCRIPT_DIR)
        if not render_complete(native_dir): raise RuntimeError(f'Invalid or incomplete render: {native_dir}')
    for resolution in args.resolutions:
        image_dir = scene_root(args.image_dataset_root, resolution, 'colmap', scene)
        if resolution != native and not render_complete(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
            common.downscale_scene(native_dir, image_dir, native, resolution,
                                   runner=lambda command: run(command, env=env, check=True, cwd=SCRIPT_DIR))
            if not render_complete(image_dir): raise RuntimeError(f'Invalid downscaled render: {image_dir}')
        if args.skip_gs: continue
        out = scene_root(args.gs_dataset_root, resolution, 'gsplat', scene)
        ckpt = out / 'ckpts' / checkpoint_name(args.train_steps)
        if ckpt.is_file(): continue
        command = common.build_train_command(image_dir, out, scene, max_splats=args.max_splats, max_steps=args.train_steps)
        command[0] = sys.executable
        command[2] = args.strategy
        run(command, env=env, check=True, cwd=SCRIPT_DIR)
        if not ckpt.is_file():
            raise RuntimeError(f'Training did not produce checkpoint: {out}')


def select_gpus(args):
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    available = {}
    for line in result.stdout.splitlines():
        if line.strip():
            index, utilization = line.split(',')
            available[int(index.strip())] = int(utilization.strip())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        allowed = {part.strip() for part in visible.split(',')}
        available = {gpu: usage for gpu, usage in available.items() if str(gpu) in allowed}
    if args.gpu_ids is None and args.gpu_id is None:
        idle = [gpu for gpu, usage in sorted(available.items()) if usage == 0]
        if not idle: raise RuntimeError('No idle GPU found within CUDA_VISIBLE_DEVICES')
        requested = idle[:1]
    else:
        requested = args.gpu_ids if args.gpu_ids is not None else [args.gpu_id]
    if not set(requested).issubset(available):
        raise ValueError(f'Requested physical GPU IDs {requested} are not among visible GPU IDs {sorted(available)}')
    return requested


def run_scenes(args, scenes, gpus):
    tasks = queue.Queue()
    for scene in scenes: tasks.put(scene)
    children = ChildProcesses()
    failures = []
    failure_lock = threading.Lock()

    def worker(gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        while not children.stopped:
            try: scene = tasks.get_nowait()
            except queue.Empty: break
            try:
                print(f'\n=== {scene} (GPU {gpu}) ===', flush=True)
                with lock_scene(args, scene):
                    process_scene(args, scene, env, run=children.run)
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                print(f'[ERROR] {scene} on GPU {gpu}: {exc}', file=sys.stderr, flush=True)
                with failure_lock:
                    failures.append({'scene_id': scene, 'gpu_id': gpu, 'error': str(exc)})
            finally:
                tasks.task_done()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus) * args.scenes_per_gpu)
    futures = [pool.submit(worker, gpu) for gpu in gpus for _ in range(args.scenes_per_gpu)]
    try:
        for future in futures: future.result()
    except BaseException:
        children.stop()
        raise
    finally:
        pool.shutdown(wait=True)
    return failures


def main(argv=None):
    args = parse_args(argv)
    print('[SCAN] Discovering ShapeNet scenes...', flush=True)
    scenes = discover_inventory(args.shapenet_root, args.categories)
    print(f'[SCAN] Found {len(scenes)} scenes.', flush=True)
    if args.limit: scenes = scenes[:args.limit]
    if not scenes:
        print('[ERROR] No models match the requested categories', file=sys.stderr); return 1
    try:
        gpus = select_gpus(args)
        failures = run_scenes(args, scenes, gpus)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr); return 1
    except KeyboardInterrupt:
        print('[ERROR] Generation interrupted', file=sys.stderr); return 130
    if failures:
        failure_path = args.image_dataset_root / 'shapenet_failures.jsonl'
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open('a', encoding='utf-8') as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            for item in failures: handle.write(json.dumps(item) + '\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
