#!/usr/bin/env python3

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Local config
OBJAVERSE_GLBS_ROOT = Path('/project/ricky/objaverse/glbs')
DATASET_ROOT = Path('/project/ricky/splatformer-sr-data')
OBJAVERSE_TRAIN_SPLIT_FILE = Path('./traintest_splits/objaverse_train.txt')
OBJAVERSE_TEST_SPLIT_FILE = Path('./traintest_splits/objaverse_test.txt')
OBJAVERSE_FAILED_SPLIT_FILE = Path('./traintest_splits/objaverse_failed.txt')
OBJAVERSE_BLENDER_BIN = 'blender-3.2.2-linux-x64/blender'
RENDER_SCRIPT = 'render_full.py'
NUM_VIEWS = 128
DEFAULT_RESOLUTIONS = [512, 128]
# DEFAULT_RESOLUTIONS = [512, 256, 128]
CHECKPOINT_NAME = 'step-000015001.ckpt'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate native-resolution Objaverse data.'
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='GPU index to use. If not set, auto-select an idle GPU.',
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='000-000',
        help='Objaverse partition in format 000-XXX (e.g., 000-003).',
    )
    parser.add_argument(
        '--resolutions',
        type=int,
        nargs='+',
        default=DEFAULT_RESOLUTIONS,
        metavar='RESOLUTION',
        help='Native image resolutions to render (default: 512 256 128).',
    )
    args = parser.parse_args()

    if not re.fullmatch(r'000-\d{3}', args.partition):
        parser.error('--partition must match format 000-XXX (e.g., 000-003).')
    if any(resolution <= 0 for resolution in args.resolutions):
        parser.error('--resolutions values must all be positive integers.')
    if len(args.resolutions) != len(set(args.resolutions)):
        parser.error('--resolutions must not contain duplicate values.')

    return args


def load_ids(split_file):
    ids = set()
    with split_file.open('r', encoding='utf-8') as f:
        for line in f:
            obj_id = line.strip()
            if obj_id:
                ids.add(obj_id)
    return ids


def load_failed_ids():
    if not OBJAVERSE_FAILED_SPLIT_FILE.exists():
        return set()
    return load_ids(OBJAVERSE_FAILED_SPLIT_FILE)


def mark_failed_scene(obj_id, failed_ids):
    if obj_id in failed_ids:
        return
    OBJAVERSE_FAILED_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OBJAVERSE_FAILED_SPLIT_FILE.open('a', encoding='utf-8') as f:
        f.write(f'{obj_id}\n')
    failed_ids.add(obj_id)
    print(f'[FAILED] Added {obj_id} to {OBJAVERSE_FAILED_SPLIT_FILE}')


def select_gpu(requested_gpu_id):
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits',
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'[ERROR] Failed to query GPUs: {e}')
        return None

    if requested_gpu_id is not None:
        return str(requested_gpu_id)

    for i, line in enumerate(result.stdout.splitlines()):
        if line.strip() == '0':
            return str(i)

    print('[ERROR] No available GPU found. Exiting.')
    return None


def split_root(split_name, resolution):
    return DATASET_ROOT / f'{split_name}-set' / 'objaverse' / str(resolution)


def build_render_command(obj_path, obj_output_dir, resolution):
    return [
        OBJAVERSE_BLENDER_BIN,
        '--background',
        '--python',
        str(RENDER_SCRIPT),
        '--',
        f'--object_path={obj_path}',
        f'--output_folder={obj_output_dir}',
        f'--resolution={resolution}',
        f'--train_views={NUM_VIEWS}',
        '--test_elevation_range=0-90',
        '--train_elevation_sin_amplitude_max_levels=15',
        '--test_num_per_floor=3',
        '--use_gpu',
    ]


def build_train_command(colmap_dir, output_dir, obj_id):
    return [
        'python',
        'trainer.py',
        'default',
        '--disable_viewer',
        f'--data-dir={colmap_dir}',
        '--data-factor=1',
        f'--result-dir={output_dir}',
        '--test-every=-1',
        '--max-steps=15000',
        '--eval-steps=15000',
        '--save-steps=15000',
        '--disable-video',
        '--init-type=sfm',
        '--load-bbox',
        '--num-points-from-bbox=50000',
        '--no-normalize-world-space',
        '--sh-degree=1',
        '--tb-every=0',
        '--strategy.refine-stop-iter=10000',
        '--strategy.prune-opa=0.1',
        '--strategy.no-verbose',
    ]


def run_training(train_cmd, env):
    process = subprocess.Popen(
        train_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    full_output = []
    for line in process.stdout:
        print(line, end='')
        full_output.append(line)

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            train_cmd,
            output=''.join(full_output),
        )


def cleanup_scene_models(split_name, resolutions, obj_id):
    for resolution in resolutions:
        scene_output_dir = (
            split_root(split_name, resolution) / 'gsplat' / obj_id
        )
        if scene_output_dir.exists():
            shutil.rmtree(scene_output_dir, ignore_errors=True)


def main():
    args = parse_args()
    path_to_objaverse = OBJAVERSE_GLBS_ROOT / args.partition

    train_ids = load_ids(OBJAVERSE_TRAIN_SPLIT_FILE)
    test_ids = load_ids(OBJAVERSE_TEST_SPLIT_FILE)
    failed_ids = load_failed_ids()

    overlap = train_ids.intersection(test_ids)
    if overlap:
        preview = ', '.join(sorted(overlap)[:10])
        print(
            f'[ERROR] Found {len(overlap)} overlapping IDs in train/test splits. '
            f'Examples: {preview}'
        )
        return 1

    gpu_id = select_gpu(args.gpu_id)
    if gpu_id is None:
        return 1

    print(f'Using GPU: {gpu_id}')

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu_id

    if not path_to_objaverse.is_dir():
        print(f'[ERROR] PATH_TO_OBJAVERSE does not exist: {path_to_objaverse}')
        return 1

    glb_paths = sorted(path_to_objaverse.glob('*.glb'))

    for obj_path in glb_paths:
        obj_id = obj_path.stem

        if obj_id in failed_ids:
            print(f'[SKIP] {obj_id}: listed in failed scenes.')
            continue

        if obj_id in train_ids:
            split_name = 'train'
        elif obj_id in test_ids:
            split_name = 'test'
        else:
            print(f'[SKIP] {obj_id}: not in split.')
            continue

        print(f'\n=== Processing {obj_id} ({split_name}) ===')
        invalid_gaussians = False

        for resolution in args.resolutions:
            resolution_root = split_root(split_name, resolution)
            colmap_dir = resolution_root / 'colmap' / obj_id
            images_dir = colmap_dir / 'images'
            output_dir = resolution_root / 'gsplat' / obj_id

            print(f'\n--- Native resolution: {resolution} ---')

            image_count = 0
            if images_dir.is_dir():
                image_count = sum(
                    1 for path in images_dir.iterdir() if path.suffix == '.png'
                )

            if image_count >= NUM_VIEWS:
                print(f'[SKIP] Render exists: {image_count} images')
            else:
                render_cmd = build_render_command(
                    obj_path,
                    colmap_dir,
                    resolution,
                )
                print('Running render:')
                print(' '.join(render_cmd))
                subprocess.run(render_cmd, env=env, check=True)

            checkpoint_path = (
                output_dir
                / obj_id
                / 'splatfacto'
                / 'nerfstudio_models'
                / CHECKPOINT_NAME
            )
            print(checkpoint_path)
            if checkpoint_path.exists():
                print(
                    f'Skipping {obj_id} at resolution {resolution}, '
                    'already exists...'
                )
                continue

            train_cmd = build_train_command(colmap_dir, output_dir, obj_id)
            print(f'\n[TRAIN] {obj_id} at resolution {resolution}')
            print(' '.join(train_cmd))

            try:
                # run_training(train_cmd, env)
                subprocess.run(train_cmd, env=env, check=True)


            except subprocess.CalledProcessError as e:
                output = e.output or ''

                if 'Invalid shape for means3d' in output:
                    print(
                        f'[CLEANUP] Removing gsplat outputs for {obj_id}'
                    )
                    print(
                        f'[ERROR] Training failed for {obj_id} '
                        f'at resolution {resolution}'
                    )
                    mark_failed_scene(obj_id, failed_ids)
                    cleanup_scene_models(
                        split_name,
                        args.resolutions,
                        obj_id,
                    )
                    print(f'[SKIP] {obj_id} due to invalid Gaussians')
                    invalid_gaussians = True
                    break
            exit()

        if invalid_gaussians:
            continue

    return 0


if __name__ == '__main__':
    sys.exit(main())
