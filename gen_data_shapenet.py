#!/usr/bin/env python3
"""Generate ShapeNet COLMAP renders and gsplat checkpoints."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
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
    p.add_argument('--manifest', type=Path, help='Default: <image root>/shapenet_splits.json')
    p.add_argument('--rebuild-manifest', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--categories', nargs='+', help='Eight-digit ShapeNet synsets to process')
    p.add_argument('--split', choices=['train', 'val', 'test'])
    p.add_argument('--limit', type=int, help='Maximum number of scenes to process')
    p.add_argument('--gpu-id', type=int)
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
    if args.max_splats < 4: p.error('--max-splats must be at least 4')
    if args.train_steps < 1: p.error('--train-steps must be positive')
    if not args.resolutions or min(args.resolutions) <= 0 or len(set(args.resolutions)) != len(args.resolutions): p.error('resolutions must be distinct positive integers')
    if args.categories and any(not re.fullmatch(r'\d{8}', x) for x in args.categories): p.error('categories must be eight-digit synsets')
    if args.gs_dataset_root is None:
        args.gs_dataset_root = gs_dataset_root(args.strategy, args.max_splats)
    return args


def discover_inventory(root):
    """Include model directories even if the expected OBJ is missing."""
    return sorted(f'{category.name}/{model.name}' for category in root.iterdir()
                  if category.is_dir() and re.fullmatch(r'\d{8}', category.name)
                  for model in category.iterdir() if model.is_dir())


def create_manifest(inventory, seed):
    assignments = {}
    groups = {}
    for scene in sorted(inventory):
        category, _ = scene.split('/')
        groups.setdefault(category, []).append(scene)
    for category, ids in sorted(groups.items()):
        category_seed = int.from_bytes(hashlib.sha256(f'{seed}:{category}'.encode()).digest(), 'big')
        random.Random(category_seed).shuffle(ids)
        count = len(ids)
        holdout = max(1, round(count * .05)) if count >= 3 else 0
        for i, scene in enumerate(ids):
            assignments[scene] = 'val' if i < holdout else 'test' if i < 2 * holdout else 'train'
    return {'version': 1, 'seed': seed, 'inventory': sorted(inventory), 'assignments': dict(sorted(assignments.items()))}


def load_manifest(path, inventory, seed, rebuild=False):
    if path.exists() and not rebuild:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('version') != 1 or sorted(inventory) != data.get('inventory'):
            raise ValueError('ShapeNet inventory changed or manifest version invalid; use --rebuild-manifest explicitly')
        if set(data.get('assignments', {})) != set(inventory) or any(x not in ('train', 'val', 'test') for x in data['assignments'].values()):
            raise ValueError('Invalid split assignments in manifest')
        return data
    data = create_manifest(inventory, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    return data


def scene_root(root, split, resolution, kind, scene):
    return root / f'{split}-set' / 'shapenet' / str(resolution) / kind / scene


def render_complete(path):
    if common.count_pngs(path / 'images') != NUM_VIEWS: return False
    required = [path / 'metadata.json'] + [path / 'sparse' / '0' / name for name in ('bbox.txt', 'cameras.txt', 'images.txt')]
    if not all(p.is_file() and p.stat().st_size for p in required): return False
    try:
        metadata = json.loads(required[0].read_text())
        if 'normalization' not in metadata: return False
        images = (path / 'sparse/0/images.txt').read_text().splitlines()
        records = [line.split() for line in images if line.strip() and not line.lstrip().startswith('#')]
        if len(records) != NUM_VIEWS or any(len(row) != 10 or not (path / 'images' / row[-1]).is_file() or not (path / 'images' / row[-1]).stat().st_size for row in records): return False
        if len({row[-1] for row in records}) != NUM_VIEWS or any(not all(math.isfinite(float(x)) for x in row[1:8]) for row in records): return False
        bbox = [[float(x) for x in line.split()] for line in (path / 'sparse/0/bbox.txt').read_text().splitlines()]
        if len(bbox) != 2 or any(len(row) != 3 or not all(math.isfinite(x) for x in row) for row in bbox) or any(b <= a for a, b in zip(*bbox)): return False
        camera = (path / 'sparse/0/cameras.txt').read_text().split()
        if len(camera) != 7 or camera[1] != 'SIMPLE_PINHOLE' or not all(math.isfinite(float(x)) and float(x) > 0 for x in camera[2:]): return False
    except (OSError, ValueError, TypeError):
        return False
    return True


def render_command(args, mesh, output, resolution):
    cmd = common.build_render_command(mesh, output, resolution)
    cmd[0] = args.blender
    cmd[3] = str(SCRIPT_DIR / 'render_shapenet.py')
    cmd[1:1] = ['--python-exit-code', '1']
    return cmd


def process_scene(args, scene, split, env):
    mesh = args.shapenet_root / scene / 'models/model_normalized.obj'
    if not mesh.is_file(): raise FileNotFoundError(f'Missing mesh: {mesh}')
    native = max(args.resolutions)
    native_dir = scene_root(args.image_dataset_root, split, native, 'colmap', scene)
    if not render_complete(native_dir):
        shutil.rmtree(native_dir, ignore_errors=True)
        subprocess.run(render_command(args, mesh, native_dir, native), env=env, check=True, cwd=SCRIPT_DIR)
        if not render_complete(native_dir): raise RuntimeError(f'Invalid or incomplete render: {native_dir}')
    for resolution in args.resolutions:
        image_dir = scene_root(args.image_dataset_root, split, resolution, 'colmap', scene)
        if resolution != native and not render_complete(image_dir):
            shutil.rmtree(image_dir, ignore_errors=True)
            common.downscale_scene(native_dir, image_dir, native, resolution)
            if not render_complete(image_dir): raise RuntimeError(f'Invalid downscaled render: {image_dir}')
        if args.skip_gs: continue
        out = scene_root(args.gs_dataset_root, split, resolution, 'gsplat', scene)
        ckpt = out / 'ckpts' / checkpoint_name(args.train_steps)
        stats = out / 'stats' / metrics_name(args.train_steps)
        if ckpt.is_file() and ckpt.stat().st_size and stats.is_file() and stats.stat().st_size: continue
        command = common.build_train_command(image_dir, out, scene, max_splats=args.max_splats, max_steps=args.train_steps)
        command[0] = sys.executable
        command[2] = args.strategy
        subprocess.run(command, env=env, check=True, cwd=SCRIPT_DIR)
        if not (ckpt.is_file() and ckpt.stat().st_size and stats.is_file() and stats.stat().st_size):
            raise RuntimeError(f'Training did not produce checkpoint and metrics: {out}')


def main(argv=None):
    args = parse_args(argv)
    inventory = discover_inventory(args.shapenet_root)
    if not inventory:
        print('[ERROR] No ShapeNet model directories found', file=sys.stderr); return 1
    manifest_path = args.manifest or args.image_dataset_root / 'shapenet_splits.json'
    if args.rebuild_manifest and manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding='utf-8'))['assignments']
            new = create_manifest(inventory, args.seed)['assignments']
            for scene, old_split in old.items():
                if new.get(scene) == old_split:
                    continue
                for root, kind in ((args.image_dataset_root, 'colmap'), (args.gs_dataset_root, 'gsplat')):
                    base = root / f'{old_split}-set' / 'shapenet'
                    if base.is_dir() and any((res / kind / scene).exists() for res in base.iterdir() if res.is_dir()):
                        raise ValueError(f'Rebuilding would move {scene} from {old_split}; relocate or remove its old outputs first')
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f'[ERROR] {exc}', file=sys.stderr); return 1
    try: manifest = load_manifest(manifest_path, inventory, args.seed, args.rebuild_manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr); return 1
    scenes = [(scene, split) for scene, split in manifest['assignments'].items()
              if (not args.categories or scene.split('/')[0] in args.categories) and (not args.split or split == args.split)]
    if args.limit: scenes = scenes[:args.limit]
    if not scenes:
        print('[ERROR] No models match the requested categories and split', file=sys.stderr); return 1
    gpu = common.select_gpu(args.gpu_id)
    if gpu is None: return 1
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    failures = []
    for scene, split in scenes:
        print(f'\n=== {scene} ({split}) ===', flush=True)
        try: process_scene(args, scene, split, env)
        except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            print(f'[ERROR] {scene}: {exc}', file=sys.stderr)
            failures.append({'scene_id': scene, 'split': split, 'error': str(exc)})
    if failures:
        failure_path = args.image_dataset_root / 'shapenet_failures.jsonl'
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open('a', encoding='utf-8') as handle:
            for item in failures: handle.write(json.dumps(item) + '\n')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
