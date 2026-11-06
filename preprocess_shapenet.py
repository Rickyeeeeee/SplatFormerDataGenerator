#!/usr/bin/env python3
"""Scan, filter, and compute statistics for ShapeNet gsplat scenes."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import pickle
import sys

import preprocess as common
from gen_data_shapenet import (IMAGE_ROOT, DEFAULT_MAX_SPLATS, DEFAULT_TRAIN_STEPS,
                              checkpoint_name, gs_dataset_root, metrics_name, scene_root)

FIELDS = ['scene_id', 'synset_id', 'image_count', 'num_gs', 'psnr', 'ssim', 'lpips', 'ellipse_time']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-dataset-root', '--image_dataset_root', type=Path, default=IMAGE_ROOT)
    p.add_argument('--gs-dataset-root', '--gs_dataset_root', type=Path,
                   help='Override the strategy- and cap-specific gsplat root')
    p.add_argument('--strategy', choices=['mcmc', 'default'], default='mcmc')
    p.add_argument('--max-splats', type=int, default=DEFAULT_MAX_SPLATS,
                   help='Splat cap used when choosing the default gsplat root')
    p.add_argument('--resolution', type=int, default=128)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    p.add_argument('--train-steps', type=int, default=DEFAULT_TRAIN_STEPS,
                   help=f'Training iterations used to produce checkpoints (default: {DEFAULT_TRAIN_STEPS})')
    p.add_argument('--psnr-threshold', '--psnr_threshold', type=float, default=common.DEFAULT_PSNR_THRESHOLD)
    p.add_argument('--scan', action='store_true')
    p.add_argument('--filter', action='store_true')
    p.add_argument('--statistics', action='store_true')
    p.add_argument('--statistics-max-gs-num', type=int, default=1_000_000)
    p.add_argument('--statistics-remove-outlier-ndevs', type=float, default=0.0)
    args = p.parse_args(argv)
    if args.resolution <= 0 or args.statistics_max_gs_num <= 0 or args.train_steps <= 0 or args.max_splats < 4: p.error('resolution, train steps, and maximum Gaussian count must be positive, and max splats at least 4')
    if not math.isfinite(args.psnr_threshold) or not math.isfinite(args.statistics_remove_outlier_ndevs) or args.statistics_remove_outlier_ndevs < 0: p.error('thresholds must be finite and outlier threshold non-negative')
    if args.gs_dataset_root is None:
        args.gs_dataset_root = gs_dataset_root(args.strategy, args.max_splats)
    return args


def discover_scenes(image_root, gs_root, split, resolution):
    result = set()
    for root, kind in ((image_root, 'colmap'), (gs_root, 'gsplat')):
        base = root / f'{split}-set' / 'shapenet' / str(resolution) / kind
        if base.is_dir():
            for category in base.iterdir():
                if category.is_dir():
                    result.update(f'{category.name}/{model.name}' for model in category.iterdir() if model.is_dir())
    return sorted(result)


def scan_scenes(image_root, gs_root, split, resolution, train_steps=DEFAULT_TRAIN_STEPS):
    rows, invalid = [], Counter()
    for scene in discover_scenes(image_root, gs_root, split, resolution):
        image_dir = scene_root(image_root, split, resolution, 'colmap', scene)
        gs_dir = scene_root(gs_root, split, resolution, 'gsplat', scene)
        try:
            count = common.count_png_images(image_dir / 'images')
            if count != common.EXPECTED_IMAGE_COUNT: raise ValueError(f'Expected 128 images, found {count}')
            checkpoint = gs_dir / 'ckpts' / checkpoint_name(train_steps)
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            metrics = common.load_render_metrics(gs_dir / 'stats' / metrics_name(train_steps))
            rows.append({'scene_id': scene, 'synset_id': scene.split('/')[0], 'image_count': count,
                         'num_gs': metrics['num_GS'], **{k: float(metrics[k]) for k in ('psnr', 'ssim', 'lpips', 'ellipse_time')}})
        except FileNotFoundError: invalid['missing_file'] += 1
        except OSError: invalid['filesystem_error'] += 1
        except ValueError: invalid['invalid_data'] += 1
    return rows, invalid


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != FIELDS: raise ValueError(f'Invalid CSV header: {path}')
        rows = list(reader)
    for row in rows:
        if len(row['scene_id'].split('/')) != 2 or row['synset_id'] != row['scene_id'].split('/')[0]: raise ValueError('Invalid scene or synset ID')
        for field in ('image_count', 'num_gs'): row[field] = int(row[field])
        for field in ('psnr', 'ssim', 'lpips', 'ellipse_time'):
            row[field] = float(row[field])
            if not math.isfinite(row[field]): raise ValueError(f'Non-finite {field}')
    return rows


def calculate_statistics(rows, root, split, resolution, max_count, outlier, train_steps=DEFAULT_TRAIN_STEPS):
    scenes, failures, reference, optional = {}, [], None, {}
    for row in rows:
        scene = row['scene_id']
        checkpoint = scene_root(root, split, resolution, 'gsplat', scene) / 'ckpts' / checkpoint_name(train_steps)
        try:
            stats, shapes = common.compute_scene_statistics(checkpoint, max_count, outlier)
            required = {k: v for k, v in shapes.items() if k not in common.OPTIONAL_SPLAT_KEYS}
            if reference is None: reference = required
            elif reference != required: raise ValueError('Parameter channel shapes differ between scenes')
            for key in common.OPTIONAL_SPLAT_KEYS:
                if key in shapes:
                    if key in optional and optional[key] != shapes[key]: raise ValueError(f'Inconsistent {key} shape')
                    optional[key] = shapes[key]
            scenes[scene] = stats
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
            failures.append({'scene_id': scene, 'stage': 'statistics', 'error': str(exc)})
    return scenes, common.aggregate_scene_statistics(scenes), failures


def main(argv=None):
    args = parse_args(argv)
    root = args.gs_dataset_root / f'{args.split}-set' / 'shapenet' / str(args.resolution)
    image_base = args.image_dataset_root / f'{args.split}-set' / 'shapenet' / str(args.resolution) / 'colmap'
    if not image_base.is_dir() or not (root / 'gsplat').is_dir():
        print('[ERROR] Image or gsplat directory does not exist', file=sys.stderr); return 1
    suffix = '' if args.train_steps == DEFAULT_TRAIN_STEPS else f'_step{args.train_steps - 1}'
    valid_path, filtered_path, stats_path = (root / name for name in
        (f'valid_scenes{suffix}.csv', f'psnr_filtered_scenes{suffix}.csv', f'gs_statistics{suffix}.json'))
    run_all = not (args.scan or args.filter or args.statistics)
    invalid = Counter()
    if args.scan or run_all or not valid_path.is_file():
        rows, invalid = scan_scenes(args.image_dataset_root, args.gs_dataset_root, args.split, args.resolution, args.train_steps)
        write_csv(rows, valid_path)
    else:
        try: rows = read_csv(valid_path)
        except (OSError, ValueError) as exc:
            print(f'[ERROR] {exc}', file=sys.stderr); return 1
    print(f'Valid scenes: {len(rows)}')
    if not (args.filter or args.statistics or run_all): return 0
    filtered = common.filter_scenes(rows, args.psnr_threshold)
    write_csv(filtered, filtered_path)
    print(f'Filtered scenes: {len(filtered)}')
    if not (args.statistics or run_all): return 0
    scenes, aggregate, failures = calculate_statistics(filtered, args.gs_dataset_root, args.split, args.resolution, args.statistics_max_gs_num, args.statistics_remove_outlier_ndevs, args.train_steps)
    result = {'generated_at': datetime.now(timezone.utc).isoformat(),
              'config': {'image_dataset_root': str(args.image_dataset_root), 'gs_dataset_root': str(args.gs_dataset_root),
                         'dataset_split': args.split, 'resolution': args.resolution, 'expected_image_count': common.EXPECTED_IMAGE_COUNT,
                         'psnr_threshold': args.psnr_threshold, 'train_steps': args.train_steps,
                         'checkpoint_name': checkpoint_name(args.train_steps),
                         'render_metrics_name': metrics_name(args.train_steps), 'raw_parameter_space': True,
                         'aggregate_weighting': 'equal_scene_average', 'valid_scenes_csv': str(valid_path), 'filtered_scenes_csv': str(filtered_path),
                         'normalized_statistics': {'feature_range': [0.0, 1.0], 'preserve_ratio': True, 'scaler_fit': 'selected_means',
                                                   'max_gs_num': args.statistics_max_gs_num, 'remove_outlier_ndevs': args.statistics_remove_outlier_ndevs,
                                                   'scale_representation': 'log_scale_plus_log_coordinate_scale', 'aggregate_weighting': 'equal_scene_mixture',
                                                   'variance': 'mean_of_population_variances_plus_squared_scene_mean_offsets'}},
              'summary': {'discovered_scenes': len(rows) + sum(invalid.values()), 'valid_scenes': len(rows), 'filtered_scenes': len(filtered),
                          'statistics_scenes': len(scenes), 'failure_count': len(failures), 'invalid_scene_count': sum(invalid.values())},
              'invalid_scene_reasons': dict(sorted(invalid.items())), 'aggregate': aggregate, 'scenes': scenes, 'failures': failures}
    common.write_json(result, stats_path)
    print(f'Wrote {stats_path}')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
