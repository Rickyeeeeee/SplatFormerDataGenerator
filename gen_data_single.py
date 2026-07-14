#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Local config
OBJAVERSE_GLBS_ROOT = Path('/project/ricky/objaverse/glbs')
TRAIN_SET_ROOT = Path('/project2/ricky/splatformer-data/train-set-512/objaverse')
TEST_SET_ROOT = Path('/project2/ricky/splatformer-data/test-set-512/objaverse')
OBJAVERSE_FAILED_SPLIT_FILE = Path(__file__).resolve().parent / 'traintest_splits' / 'objaverse_failed.txt'
OBJAVERSE_BLENDER_BIN = 'blender-3.2.2-linux-x64/blender'
RENDER_SCRIPT = Path(__file__).resolve().parent / 'render_full.py'
NUM_VIEWS = 128
MAX_RESOLUTION = 512


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Objaverse data for a single scene.')
    parser.add_argument(
        '--scene_id',
        '--scene_name',
        dest='scene_id',
        type=str,
        required=True,
        help='Objaverse scene id/name to search for under the glbs root.',
    )
    parser.add_argument(
        '--split',
        choices=('train', 'test'),
        required=True,
        help='Output split root to use for this scene.',
    )
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=None,
        help='GPU index to use. If not set, auto-select an idle GPU.',
    )
    parser.add_argument(
        '--force_render',
        action='store_true',
        help='Regenerate render outputs even if enough images already exist.',
    )
    parser.add_argument(
        '--force_gs_model',
        '--force_gs',
        dest='force_gs_model',
        action='store_true',
        help='Regenerate nerfstudio GS model outputs even if checkpoints already exist.',
    )
    return parser.parse_args()


def find_object_path(scene_id):
    obj_id = Path(scene_id).stem
    if not obj_id:
        print('[ERROR] scene_id must not be empty.')
        return None, None
    if not OBJAVERSE_GLBS_ROOT.is_dir():
        print(f'[ERROR] OBJAVERSE_GLBS_ROOT does not exist: {OBJAVERSE_GLBS_ROOT}')
        return None, None

    matches = sorted(OBJAVERSE_GLBS_ROOT.rglob(f'{obj_id}.glb'))
    if not matches:
        print(f'[ERROR] Could not find {obj_id}.glb under {OBJAVERSE_GLBS_ROOT}')
        return None, None
    if len(matches) > 1:
        print(f'[ERROR] Found multiple matches for {obj_id}.glb under {OBJAVERSE_GLBS_ROOT}:')
        for match in matches:
            print(f'  {match}')
        return None, None

    return obj_id, matches[0]


def load_failed_ids():
    failed_ids = set()
    if OBJAVERSE_FAILED_SPLIT_FILE.exists():
        with OBJAVERSE_FAILED_SPLIT_FILE.open('r', encoding='utf-8') as f:
            for line in f:
                obj_id = line.strip()
                if obj_id:
                    failed_ids.add(obj_id)
    return failed_ids


def mark_failed_scene(obj_id, failed_ids):
    if obj_id in failed_ids:
        return
    OBJAVERSE_FAILED_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OBJAVERSE_FAILED_SPLIT_FILE.open('a', encoding='utf-8') as f:
        f.write(f'{obj_id}\n')
    failed_ids.add(obj_id)
    print(f'[FAILED] Added {obj_id} to {OBJAVERSE_FAILED_SPLIT_FILE}')


def select_gpu(gpu_id_arg):
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

    if gpu_id_arg is not None:
        return str(gpu_id_arg)

    for i, line in enumerate(result.stdout.splitlines()):
        if line.strip() == '0':
            return str(i)

    print('[ERROR] No available GPU found. Exiting.')
    return None


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
    output = ''.join(full_output)

    if process.returncode != 0:
        raise subprocess.CalledProcessError(
            process.returncode,
            train_cmd,
            output=output,
        )


def main():
    args = parse_args()

    obj_id, obj_path = find_object_path(args.scene_id)
    if obj_path is None:
        return 1
    split_root = TRAIN_SET_ROOT if args.split == 'train' else TEST_SET_ROOT

    failed_ids = load_failed_ids()
    if obj_id in failed_ids:
        print(f'[SKIP] {obj_id}: listed in failed scenes.')
        return 0

    gpu_id = select_gpu(args.gpu_id)
    if gpu_id is None:
        return 1

    print(f'Using GPU: {gpu_id}')

    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = gpu_id

    obj_output_dir = split_root / 'colmap' / obj_id
    images_dir = obj_output_dir / 'images'

    print(f'\n=== Processing {obj_id} ({args.split}) ===')
    print(f'GLB root: {OBJAVERSE_GLBS_ROOT}')
    print(f'Object path: {obj_path}')

    image_count = 0
    if images_dir.is_dir():
        image_count = sum(1 for p in images_dir.iterdir() if p.suffix == '.png')

    if args.force_render and obj_output_dir.exists():
        print(f'[FORCE] Removing existing render output: {obj_output_dir}')
        shutil.rmtree(obj_output_dir, ignore_errors=True)
        image_count = 0

    if image_count >= NUM_VIEWS:
        print(f'[SKIP] Render exists: {image_count} images')
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
        print('Running render:')
        print(' '.join(render_cmd))
        try:
            subprocess.run(render_cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f'[ERROR] Rendering failed for {obj_id}: {e}')
            return e.returncode

    colmap_dir = split_root / 'colmap' / obj_id
    output_dir = split_root / 'nerfstudio'

    for df in [1, 2, 4]:
        experiment_dir = f'{obj_id}/df-{df}'
        experiment_path = output_dir / experiment_dir
        checkpoint_path = experiment_path / 'splatfacto' / 'nerfstudio_models' / 'step-000010001.ckpt'
        print(checkpoint_path)
        if args.force_gs_model and experiment_path.exists():
            print(f'[FORCE] Removing existing GS model output: {experiment_path}')
            shutil.rmtree(experiment_path, ignore_errors=True)
        if checkpoint_path.exists():
            print(f'Skipping {experiment_dir}, already exists...')
            continue

        train_cmd = [
            'ns-train',
            'splatfacto',
            '--logging.local-writer.enable=False',
            '--logging.profiler=none',
            f'--pipeline.datamanager.data={colmap_dir}',
            '--pipeline.model.sh_degree=1',
            '--pipeline.save_img=False',
            '--pipeline.datamanager.images-on-gpu=True',
            '--pipeline.datamanager.cache-images=gpu',
            '--test_after_train',
            'True',
            f'--output_dir={output_dir}',
            f'--experiment-name={experiment_dir}',
            '--relative-model-dir=nerfstudio_models',
            '--vis',
            'viewer',
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

        print(f'\n[TRAIN] {experiment_dir}')
        print(' '.join(train_cmd))

        try:
            run_training(train_cmd, env)
        except subprocess.CalledProcessError as e:
            output = e.output or ''

            if 'Invalid shape for means3d' in output:
                print(f'[CLEANUP] Removing nerfstudio outputs for {obj_id}')
                print(f'[ERROR] Training failed for {experiment_dir}')
                mark_failed_scene(obj_id, failed_ids)

                scene_output_dir = output_dir / obj_id
                if scene_output_dir.exists():
                    shutil.rmtree(scene_output_dir, ignore_errors=True)

                print(f'[SKIP] {obj_id} due to invalid Gaussians')
                return 1

            print(f'[ERROR] Training failed for {experiment_dir}')
            return e.returncode
        except FileNotFoundError as e:
            print(f'[ERROR] Failed to launch training command: {e}')
            return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
