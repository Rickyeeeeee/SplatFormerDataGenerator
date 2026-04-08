#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

# Local config
OBJAVERSE_GLBS_ROOT = Path('/project/ricky/objaverse/glbs')
TRAIN_SET_ROOT = Path('/project2/ricky/splatformer-data/train-set-512/objaverse')
TEST_SET_ROOT = Path('/project2/ricky/splatformer-data/test-set-512/objaverse')
OBJAVERSE_TRAIN_SPLIT_FILE = Path('./traintest_splits/objaverse_train.txt')
OBFAVERSE_TEST_SPLIT_FILE = Path('./traintest_splits/objaverse_test.txt')
OBJAVERSE_BLENDER_BIN = 'blender-3.2.2-linux-x64/blender'
RENDER_SCRIPT = 'render_full.py'
NUM_VIEWS = 128
MAX_RESOLUTION = 512


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Objaverse data.')
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
    args = parser.parse_args()

    if not re.fullmatch(r'000-\d{3}', args.partition):
        parser.error('--partition must match format 000-XXX (e.g., 000-003).')

    return args


args = parse_args()
PATH_TO_OBJAVERSE = OBJAVERSE_GLBS_ROOT / args.partition

# Load train-test split
train_ids = set()
with OBJAVERSE_TRAIN_SPLIT_FILE.open('r', encoding='utf-8') as f:
    for line in f:
        obj_id = line.strip()
        if obj_id:
            train_ids.add(obj_id)

test_ids = set()
with OBFAVERSE_TEST_SPLIT_FILE.open('r', encoding='utf-8') as f:
    for line in f:
        obj_id = line.strip()
        if obj_id:
            test_ids.add(obj_id)

overlap = train_ids.intersection(test_ids)
if overlap:
    preview = ', '.join(sorted(list(overlap))[:10])
    print(
        f'[ERROR] Found {len(overlap)} overlapping IDs in train/test splits. '
        f'Examples: {preview}'
    )
    sys.exit(1)

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
    sys.exit(1)

gpu_id = None
if args.gpu_id is not None:
    gpu_id = str(args.gpu_id)
if gpu_id is None:
    for i, line in enumerate(result.stdout.splitlines()):
        if line.strip() == '0':
            gpu_id = str(i)
            break

if gpu_id is None:
    print('[ERROR] No available GPU found. Exiting.')
    sys.exit(1)

print(f'Using GPU: {gpu_id}')

env = dict(os.environ)
env['CUDA_VISIBLE_DEVICES'] = gpu_id

if not PATH_TO_OBJAVERSE.is_dir():
    print(f'[ERROR] PATH_TO_OBJAVERSE does not exist or is not a directory: {PATH_TO_OBJAVERSE}')
    sys.exit(1)

glb_paths = sorted(PATH_TO_OBJAVERSE.glob('*.glb'))

# Main Loop
for obj_path in glb_paths:
    obj_id = obj_path.stem

    # Train/Test split
    if obj_id in train_ids:
        split_name = 'train'
        split_root = TRAIN_SET_ROOT
    elif obj_id in test_ids:
        split_name = 'test'
        split_root = TEST_SET_ROOT
    else:
        print(f'[SKIP] {obj_id}: not in train/test split.')
        continue

    obj_output_dir = split_root / 'colmap' / obj_id
    images_dir = obj_output_dir / 'images'

    print(f'Processing {obj_id} ({split_name})')

    # Blender rendering
    image_count = 0
    if images_dir.is_dir():
        image_count = sum(1 for p in images_dir.iterdir() if p.is_file() and p.suffix == '.png')

    if image_count >= NUM_VIEWS:
        print(f'[SKIP] {obj_id}: found {image_count} images in {images_dir}, skipping render.')
    else:
        render_cmd = [
            OBJAVERSE_BLENDER_BIN,
            '--background',
            '--python',
            str(RENDER_SCRIPT),
            '--',
            f'--object_path={obj_path}',
            f'--output_folder={obj_output_dir}',
            f'--resolution={MAX_RESOLUTION}',
            f'--train_views={NUM_VIEWS}',
            '--test_elevation_range=0-90',
            '--train_elevation_sin_amplitude_max_levels=15',
            '--test_num_per_floor=3',
            '--use_gpu',
        ]
        print("Running command:")
        print(" ".join(render_cmd))
        subprocess.run(render_cmd, env=env, check=True)

    colmap_dir = split_root / 'colmap' / obj_id
    output_dir = split_root / 'nerfstudio'

    # Nerfstudio training
    for df in [1, 2, 4]:
        experiment_dir = f'{obj_id}/df-{df}'
        if os.path.exists(output_dir / experiment_dir / 'splatfacto' / 'nerfstduio_models' / 'step-000010001.ckpt'):
            print(f"Skipping {experiment_dir}, already exists...")
            continue
        train_cmd = [
            'ns-train',
            'splatfacto',
            '--logging.local-writer.enable=False',
            '--logging.profiler=none',
            f'--pipeline.datamanager.data={colmap_dir}',
            '--pipeline.model.sh_degree=1',
            '--pipeline.save_img=False',
            '--test_after_train',
            'True',
            f'--output_dir={output_dir}',
            f'--experiment-name={experiment_dir}',
            '--relative-model-dir=nerfstudio_models',
            '--vis',
            'viewer+tensorboard',
            '--steps_per_eval_image=100000',
            '--steps_per_eval_all_images=1000000',
            '--max_num_iterations=30000',
            '--save_only_latest_checkpoint',
            'False',
            '--steps_per_save=100000',
            '--save_last_checkpoint',
            'True',
            '--early_stop_steps=10000',
            '--save_only_gs_params',
            'True',
            '--viewer.quit-on-train-completion',
            'True',
            'colmap',
            f'--downscale_factor={df}',
            '--downscale-rounding-mode=floor',
            '--load_3D_points',
            'True',
            '--eval-mode',
            'all',
            '--auto_scale_poses=False',
            '--orientation_method=none',
            '--center_method=none',
            '--load_bbox',
            'True',
            '--num_points_from_bbox',
            '50000',
            '--assume_colmap_world_coordinate_convention',
            'False',
        ]
        subprocess.run(train_cmd, env=env, check=True)
