#!/usr/bin/env python3

import argparse
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent

OBJAVERSE_GLBS_ROOT = Path('/project/ricky/objaverse/glbs')
TRAIN_SET_ROOT = Path('/project2/ricky/splatformer-data/train-set-512/objaverse')
TEST_SET_ROOT = Path('/project/ricky/splatformer-data/test-set-512/objaverse')
OBJAVERSE_TRAIN_SPLIT_FILE = SCRIPT_DIR / 'traintest_splits' / 'objaverse_train.txt'
OBJAVERSE_TEST_SPLIT_FILE = SCRIPT_DIR / 'traintest_splits' / 'objaverse_test.txt'
OBJAVERSE_FAILED_SPLIT_FILE = SCRIPT_DIR / 'traintest_splits' / 'objaverse_failed.txt'
LOG_DIR = SCRIPT_DIR / 'logs'
PARTITION_PATTERN = re.compile(r'000-\d{3}$')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Track Objaverse dataset generation progress by partition.'
    )
    parser.add_argument(
        '--objaverse_root',
        type=Path,
        default=OBJAVERSE_GLBS_ROOT,
        help=f'Root containing Objaverse partition directories. Default: {OBJAVERSE_GLBS_ROOT}',
    )
    parser.add_argument(
        '--train_root',
        type=Path,
        default=TRAIN_SET_ROOT,
        help=f'Train output root. Default: {TRAIN_SET_ROOT}',
    )
    parser.add_argument(
        '--test_root',
        type=Path,
        default=TEST_SET_ROOT,
        help=f'Test output root. Default: {TEST_SET_ROOT}',
    )
    parser.add_argument(
        '--train_split_file',
        type=Path,
        default=OBJAVERSE_TRAIN_SPLIT_FILE,
        help=f'Train split file. Default: {OBJAVERSE_TRAIN_SPLIT_FILE}',
    )
    parser.add_argument(
        '--test_split_file',
        type=Path,
        default=OBJAVERSE_TEST_SPLIT_FILE,
        help=f'Test split file. Default: {OBJAVERSE_TEST_SPLIT_FILE}',
    )
    parser.add_argument(
        '--failed_split_file',
        type=Path,
        default=OBJAVERSE_FAILED_SPLIT_FILE,
        help=f'Optional failed scene file. Default: {OBJAVERSE_FAILED_SPLIT_FILE}',
    )
    parser.add_argument(
        '--partition',
        action='append',
        default=None,
        help=(
            'Partition to scan, matching 000-XXX. May be passed multiple times. '
            'If omitted, all matching partition directories are scanned.'
        ),
    )
    parser.add_argument(
        '--log_path',
        type=Path,
        default=None,
        help='Path to write the tracking log. Defaults to DataGenerator/logs/track_scenes_<timestamp>.log.',
    )
    args = parser.parse_args()

    if args.partition:
        invalid = [partition for partition in args.partition if not PARTITION_PATTERN.fullmatch(partition)]
        if invalid:
            parser.error(f'--partition must match format 000-XXX. Invalid: {", ".join(invalid)}')

    return args


def load_split_file(path, required=True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f'Required split file does not exist: {path}')
        return set()

    ids = set()
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            obj_id = line.strip()
            if obj_id:
                ids.add(obj_id)
    return ids


def discover_partitions(objaverse_root, requested_partitions):
    if requested_partitions:
        return [objaverse_root / partition for partition in requested_partitions]

    if not objaverse_root.is_dir():
        return []

    return sorted(
        path
        for path in objaverse_root.iterdir()
        if path.is_dir() and PARTITION_PATTERN.fullmatch(path.name)
    )


def empty_split_counts():
    return Counter(
        {
            'scenes': 0,
            'colmap_success': 0,
            'nerfstudio_success': 0,
            'both_success': 0,
        }
    )


def scan_scene(obj_id, split_root):
    colmap_exists = (split_root / 'colmap' / obj_id).is_dir()
    nerfstudio_exists = (split_root / 'nerfstudio' / obj_id).is_dir()
    return colmap_exists, nerfstudio_exists


def scan_partition(partition_path, train_ids, test_ids, failed_ids, train_root, test_root):
    stats = {
        'partition': partition_path.name,
        'exists': partition_path.is_dir(),
        'total_glbs': 0,
        'failed': 0,
        'not_in_split': 0,
        'train': empty_split_counts(),
        'test': empty_split_counts(),
    }

    if not partition_path.is_dir():
        return stats

    for obj_path in sorted(partition_path.glob('*.glb')):
        obj_id = obj_path.stem
        stats['total_glbs'] += 1

        if obj_id in failed_ids:
            stats['failed'] += 1
            continue

        if obj_id in train_ids:
            split_name = 'train'
            split_root = train_root
        elif obj_id in test_ids:
            split_name = 'test'
            split_root = test_root
        else:
            stats['not_in_split'] += 1
            continue

        colmap_exists, nerfstudio_exists = scan_scene(obj_id, split_root)
        split_stats = stats[split_name]
        split_stats['scenes'] += 1
        split_stats['colmap_success'] += int(colmap_exists)
        split_stats['nerfstudio_success'] += int(nerfstudio_exists)
        split_stats['both_success'] += int(colmap_exists and nerfstudio_exists)

    return stats


def format_table(rows):
    headers = [
        'partition',
        'glbs',
        'train',
        'train_colmap',
        'train_nerfstudio',
        'train_both',
        'test',
        'test_colmap',
        'test_nerfstudio',
        'test_both',
        'failed',
        'not_in_split',
    ]
    data = []
    for row in rows:
        data.append(
            [
                row['partition'],
                str(row['total_glbs']) if row['exists'] else 'missing',
                str(row['train']['scenes']),
                str(row['train']['colmap_success']),
                str(row['train']['nerfstudio_success']),
                str(row['train']['both_success']),
                str(row['test']['scenes']),
                str(row['test']['colmap_success']),
                str(row['test']['nerfstudio_success']),
                str(row['test']['both_success']),
                str(row['failed']),
                str(row['not_in_split']),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in data)) if data else len(headers[i])
        for i in range(len(headers))
    ]
    lines = [
        '  '.join(headers[i].ljust(widths[i]) for i in range(len(headers))),
        '  '.join('-' * widths[i] for i in range(len(headers))),
    ]
    lines.extend('  '.join(row[i].ljust(widths[i]) for i in range(len(headers))) for row in data)
    return lines


def build_totals(rows):
    totals = {
        'partitions': len(rows),
        'missing_partitions': sum(1 for row in rows if not row['exists']),
        'total_glbs': sum(row['total_glbs'] for row in rows),
        'failed': sum(row['failed'] for row in rows),
        'not_in_split': sum(row['not_in_split'] for row in rows),
        'train': empty_split_counts(),
        'test': empty_split_counts(),
    }
    for row in rows:
        totals['train'].update(row['train'])
        totals['test'].update(row['test'])
    return totals


def format_report(args, rows, totals):
    timestamp = datetime.now().isoformat(timespec='seconds')
    lines = [
        f'Objaverse dataset generation tracking report',
        f'Timestamp: {timestamp}',
        '',
        f'Objaverse root: {args.objaverse_root}',
        f'Train root: {args.train_root}',
        f'Test root: {args.test_root}',
        f'Train split file: {args.train_split_file}',
        f'Test split file: {args.test_split_file}',
        f'Failed split file: {args.failed_split_file}',
        '',
        'Per-partition counts:',
    ]
    lines.extend(format_table(rows))
    lines.extend(
        [
            '',
            'Overall totals:',
            f'  partitions scanned: {totals["partitions"]}',
            f'  missing partitions: {totals["missing_partitions"]}',
            f'  total GLBs scanned: {totals["total_glbs"]}',
            f'  failed scenes skipped: {totals["failed"]}',
            f'  scenes not in split: {totals["not_in_split"]}',
            f'  train scenes found: {totals["train"]["scenes"]}',
            f'  train colmap successes: {totals["train"]["colmap_success"]}',
            f'  train nerfstudio successes: {totals["train"]["nerfstudio_success"]}',
            f'  train both successes: {totals["train"]["both_success"]}',
            f'  test scenes found: {totals["test"]["scenes"]}',
            f'  test colmap successes: {totals["test"]["colmap_success"]}',
            f'  test nerfstudio successes: {totals["test"]["nerfstudio_success"]}',
            f'  test both successes: {totals["test"]["both_success"]}',
        ]
    )
    return lines


def default_log_path():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return LOG_DIR / f'track_scenes_{timestamp}.log'


def main():
    args = parse_args()

    train_ids = load_split_file(args.train_split_file, required=True)
    test_ids = load_split_file(args.test_split_file, required=True)
    failed_ids = load_split_file(args.failed_split_file, required=False)

    overlap = train_ids.intersection(test_ids)
    if overlap:
        preview = ', '.join(sorted(overlap)[:10])
        raise ValueError(
            f'Found {len(overlap)} overlapping IDs in train/test splits. Examples: {preview}'
        )

    partitions = discover_partitions(args.objaverse_root, args.partition)
    rows = [
        scan_partition(
            partition,
            train_ids,
            test_ids,
            failed_ids,
            args.train_root,
            args.test_root,
        )
        for partition in partitions
    ]
    totals = build_totals(rows)
    report_lines = format_report(args, rows, totals)

    log_path = args.log_path or default_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')

    print('\n'.join(report_lines))
    print(f'\nLog written to: {log_path}')


if __name__ == '__main__':
    main()
