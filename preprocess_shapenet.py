#!/usr/bin/env python3
"""Validate ShapeNet scenes, create splits and compute training statistics."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import stat
import sys

import preprocess as common
from gen_data_shapenet import (IMAGE_ROOT, DEFAULT_MAX_SPLATS, DEFAULT_TRAIN_STEPS,
                              checkpoint_name, gs_dataset_root, metrics_name,
                              scene_root)

FIELDS = ['scene_id', 'synset_id', 'image_count', 'num_gs', 'psnr', 'ssim', 'lpips', 'ellipse_time']
SPLITS = ('train', 'val', 'test')


def render_complete(path):
    """Validate COLMAP metadata and non-empty images for preprocessing."""
    required = [path / 'metadata.json'] + [path / 'sparse' / '0' / name for name in ('bbox.txt', 'cameras.txt', 'images.txt')]
    try:
        for required_path in required:
            info = required_path.stat()
            if not stat.S_ISREG(info.st_mode) or not info.st_size: return False
        with os.scandir(path / 'images') as entries:
            pngs = {entry.name: entry for entry in entries if entry.name.endswith('.png')}
        if len(pngs) != common.EXPECTED_IMAGE_COUNT: return False
        metadata = json.loads(required[0].read_text())
        if 'normalization' not in metadata: return False
        images = (path / 'sparse/0/images.txt').read_text().splitlines()
        records = [line.split() for line in images if line.strip() and not line.lstrip().startswith('#')]
        if len(records) != common.EXPECTED_IMAGE_COUNT or any(len(row) != 10 or row[-1] not in pngs for row in records): return False
        if len({row[-1] for row in records}) != common.EXPECTED_IMAGE_COUNT or any(not all(math.isfinite(float(x)) for x in row[1:8]) for row in records): return False
        for entry in pngs.values():
            info = entry.stat()
            if not stat.S_ISREG(info.st_mode) or not info.st_size: return False
        bbox = [[float(x) for x in line.split()] for line in (path / 'sparse/0/bbox.txt').read_text().splitlines()]
        if len(bbox) != 2 or any(len(row) != 3 or not all(math.isfinite(x) for x in row) for row in bbox) or any(b <= a for a, b in zip(*bbox)): return False
        camera = (path / 'sparse/0/cameras.txt').read_text().split()
        if len(camera) != 7 or camera[1] != 'SIMPLE_PINHOLE' or not all(math.isfinite(float(x)) and float(x) > 0 for x in camera[2:]): return False
    except (FileNotFoundError, NotADirectoryError, ValueError, TypeError):
        return False
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image-dataset-root', '--image_dataset_root', type=Path, default=IMAGE_ROOT)
    p.add_argument('--gs-dataset-root', '--gs_dataset_root', type=Path)
    p.add_argument('--strategy', choices=['mcmc', 'default'], default='mcmc')
    p.add_argument('--max-splats', type=int, default=DEFAULT_MAX_SPLATS)
    p.add_argument('--resolutions', '--resolution', type=int, nargs='+', default=[128])
    p.add_argument('--split-ratios', type=float, nargs=3, metavar=('TRAIN', 'VAL', 'TEST'), default=[0.9, 0.05, 0.05])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-steps', type=int, default=DEFAULT_TRAIN_STEPS)
    p.add_argument('--psnr-threshold', '--psnr_threshold', type=float, default=common.DEFAULT_PSNR_THRESHOLD)
    p.add_argument('--skip-statistics', action='store_true', help='Generate splits and counts without loading checkpoints')
    p.add_argument('--statistics-max-gs-num', type=int, default=1_000_000)
    p.add_argument('--statistics-remove-outlier-ndevs', type=float, default=0.0)
    args = p.parse_args(argv)
    if min(args.resolutions) <= 0 or len(args.resolutions) != len(set(args.resolutions)):
        p.error('resolutions must be distinct positive integers')
    if args.statistics_max_gs_num <= 0 or args.train_steps <= 0 or args.max_splats < 4:
        p.error('train steps and maximum Gaussian count must be positive, and max splats at least 4')
    if any(not math.isfinite(x) or x < 0 for x in args.split_ratios) or not math.isclose(sum(args.split_ratios), 1.0, abs_tol=1e-8):
        p.error('split ratios must be finite, non-negative, and sum to 1')
    if not math.isfinite(args.psnr_threshold) or not math.isfinite(args.statistics_remove_outlier_ndevs) or args.statistics_remove_outlier_ndevs < 0:
        p.error('thresholds must be finite and outlier threshold non-negative')
    if args.gs_dataset_root is None:
        args.gs_dataset_root = gs_dataset_root(args.strategy, args.max_splats)
    return args


def discover_scenes(image_root, gs_root, resolution):
    result = set()
    for root, kind in ((image_root, 'colmap'), (gs_root, 'gsplat')):
        base = root / 'shapenet' / str(resolution) / kind
        if base.is_dir():
            for category in base.iterdir():
                if category.is_dir():
                    result.update(f'{category.name}/{model.name}' for model in category.iterdir() if model.is_dir())
    return sorted(result)


def scan_scenes(image_root, gs_root, resolution, train_steps=DEFAULT_TRAIN_STEPS):
    rows, invalid = [], Counter()
    for scene in discover_scenes(image_root, gs_root, resolution):
        image_dir = scene_root(image_root, resolution, 'colmap', scene)
        gs_dir = scene_root(gs_root, resolution, 'gsplat', scene)
        try:
            if not render_complete(image_dir): raise ValueError('Invalid or incomplete COLMAP render')
            count = common.EXPECTED_IMAGE_COUNT  # render_complete already counted the PNGs
            checkpoint = gs_dir / 'ckpts' / checkpoint_name(train_steps)
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            if not checkpoint.stat().st_size: raise ValueError('Empty checkpoint')
            metrics = common.load_render_metrics(gs_dir / 'stats' / metrics_name(train_steps))
            rows.append({'scene_id': scene, 'synset_id': scene.split('/')[0], 'image_count': count,
                         'num_gs': metrics['num_GS'], **{k: float(metrics[k]) for k in ('psnr', 'ssim', 'lpips', 'ellipse_time')}})
        except FileNotFoundError: invalid['missing_file'] += 1
        except OSError: invalid['filesystem_error'] += 1
        except ValueError: invalid['invalid_data'] += 1
    return rows, invalid


def split_scenes(scene_ids, ratios, seed):
    """Assign every scene once; stratify independently for each synset."""
    groups, assignments = {}, {}
    for scene in sorted(scene_ids): groups.setdefault(scene.split('/')[0], []).append(scene)
    for category, ids in sorted(groups.items()):
        category_seed = int.from_bytes(hashlib.sha256(f'{seed}:{category}'.encode()).digest(), 'big')
        random.Random(category_seed).shuffle(ids)
        n = len(ids)
        positive = [i for i, ratio in enumerate(ratios) if ratio > 0]
        if n < len(positive) and ratios[0] > 0:
            counts = [n, 0, 0]
        else:
            exact = [n * ratio for ratio in ratios]
            counts = [int(x) for x in exact]
            for i in sorted(range(3), key=lambda j: (-(exact[j] - counts[j]), j))[:n - sum(counts)]:
                counts[i] += 1
            if n >= len(positive):
                for i in positive:
                    if counts[i] == 0:
                        donor = max(positive, key=lambda j: counts[j])
                        counts[donor] -= 1
                        counts[i] += 1
        offset = 0
        for split, count in zip(SPLITS, counts):
            for scene in ids[offset:offset + count]: assignments[scene] = split
            offset += count
    return assignments


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def calculate_statistics(rows, root, resolution, max_count, outlier, train_steps=DEFAULT_TRAIN_STEPS):
    scenes, failures, reference, optional = {}, [], None, {}
    for row in rows:
        scene = row['scene_id']
        checkpoint = scene_root(root, resolution, 'gsplat', scene) / 'ckpts' / checkpoint_name(train_steps)
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
    rows_by_resolution, counts = {}, {}
    for resolution in args.resolutions:
        base = args.gs_dataset_root / 'shapenet' / str(resolution) / 'gsplat'
        image_base = args.image_dataset_root / 'shapenet' / str(resolution) / 'colmap'
        if not base.is_dir() or not image_base.is_dir():
            print(f'[ERROR] Image or gsplat directory does not exist for resolution {resolution}', file=sys.stderr)
            return 1
        rows, invalid = scan_scenes(args.image_dataset_root, args.gs_dataset_root, resolution, args.train_steps)
        passing = common.filter_scenes(rows, args.psnr_threshold)
        rows_by_resolution[resolution] = {row['scene_id']: row for row in passing}
        counts[str(resolution)] = {'discovered': len(rows) + sum(invalid.values()), 'valid_before_psnr': len(rows),
                                   'passing_psnr': len(passing), 'rejected_psnr': len(rows) - len(passing),
                                   'invalid': sum(invalid.values()), 'invalid_reasons': dict(sorted(invalid.items()))}
    common_ids = set.intersection(*(set(rows) for rows in rows_by_resolution.values()))
    assignments = split_scenes(common_ids, args.split_ratios, args.seed)
    split_counts = {split: sum(value == split for value in assignments.values()) for split in SPLITS}
    config = {'image_dataset_root': str(args.image_dataset_root), 'gs_dataset_root': str(args.gs_dataset_root),
              'strategy': args.strategy, 'max_splats': args.max_splats, 'train_steps': args.train_steps,
              'checkpoint_step': args.train_steps - 1, 'checkpoint_name': checkpoint_name(args.train_steps),
              'metrics_name': metrics_name(args.train_steps), 'psnr_threshold': args.psnr_threshold,
              'expected_image_count': common.EXPECTED_IMAGE_COUNT, 'resolutions': args.resolutions,
              'seed': args.seed, 'split_ratios': dict(zip(SPLITS, args.split_ratios)),
              'statistics_max_gs_num': args.statistics_max_gs_num,
              'statistics_remove_outlier_ndevs': args.statistics_remove_outlier_ndevs}
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'config': config,
              'resolutions': counts, 'passing_all_resolutions': len(common_ids), 'split_counts': split_counts}
    failures = []
    for resolution in args.resolutions:
        root = args.gs_dataset_root / 'shapenet' / str(resolution)
        rows = rows_by_resolution[resolution]
        for split in SPLITS:
            selected = [rows[scene] for scene in sorted(common_ids) if assignments[scene] == split]
            write_csv(selected, root / f'{split}_scenes.csv')
        if not args.skip_statistics:
            train_rows = [rows[scene] for scene in sorted(common_ids) if assignments[scene] == 'train']
            scenes, aggregate, errors = calculate_statistics(train_rows, args.gs_dataset_root, resolution,
                args.statistics_max_gs_num, args.statistics_remove_outlier_ndevs, args.train_steps)
            failures.extend(errors)
            stats = {'generated_at': report['generated_at'], 'config': dict(config, resolution=resolution, dataset_split='train'),
                     'status': 'complete' if not errors else 'partial',
                     'summary': {'training_scenes': len(train_rows), 'statistics_scenes': len(scenes), 'failure_count': len(errors)},
                     'aggregate': aggregate, 'scenes': scenes, 'failures': errors}
            common.write_json(stats, root / 'gs_statistics.json')
        else:
            common.write_json({'generated_at': report['generated_at'],
                               'config': dict(config, resolution=resolution, dataset_split='train'),
                               'status': 'skipped'}, root / 'gs_statistics.json')
    report['statistics_failures'] = len(failures)
    report['statistics_status'] = 'skipped' if args.skip_statistics else ('partial' if failures else 'complete')
    common.write_json(report, args.gs_dataset_root / 'shapenet' / 'gen_data.json')
    print(f'Passing all resolutions: {len(common_ids)}; splits: {split_counts}')
    return int(bool(failures))


if __name__ == '__main__':
    sys.exit(main())
