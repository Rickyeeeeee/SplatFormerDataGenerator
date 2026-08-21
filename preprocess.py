#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_DATASET_ROOT = Path("/project/ricky/splatformer-sr-data")
DEFAULT_OLD_RESOLUTION = 128
DEFAULT_NEW_RESOLUTION = 512
DEFAULT_PSNR_THRESHOLD = 27.0
EXPECTED_IMAGE_COUNT = 128
SOURCE_CHECKPOINT_NAME = "ckpt_14999_rank0.pt"
SOURCE_STATS_NAME = "val_step14999.json"
REFINER_STEPS = 3_000
REFINED_CHECKPOINT_NAME = "ckpt_2999_rank0.pt"
REQUIRED_METRICS = ("num_GS", "psnr", "ssim", "lpips", "ellipse_time")
REQUIRED_SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0", "shN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan paired GSplat scenes, filter them by input PSNR, refine 128px "
            "splats on 512px images, and calculate raw parameter statistics."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root containing train-set or test-set (default: {DEFAULT_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--old_resolution",
        type=int,
        default=DEFAULT_OLD_RESOLUTION,
        help="Source GS/image resolution (default: 128).",
    )
    parser.add_argument(
        "--new_resolution",
        type=int,
        default=DEFAULT_NEW_RESOLUTION,
        help="Target image resolution (default: 512).",
    )
    parser.add_argument(
        "--psnr_threshold",
        type=float,
        default=DEFAULT_PSNR_THRESHOLD,
        help="Inclusive source-resolution PSNR threshold (default: 27).",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU index for refinement. If omitted, select the first idle GPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run refinement even when the expected final checkpoint exists.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Regenerate valid_scenes.csv even when it already exists.",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Regenerate psnr_filtered_scenes.csv even when it already exists.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Run the refinement stage for filtered scenes.",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Regenerate gs_statistics.json from available refined checkpoints.",
    )
    parser.add_argument(
        "--testset",
        action="store_true",
        help=(
            "Use test-set as input and test-set-4x-up as refinement output "
            "instead of the train-set directories."
        ),
    )
    args = parser.parse_args()

    if args.old_resolution <= 0 or args.new_resolution <= 0:
        parser.error("--old_resolution and --new_resolution must be positive.")
    if args.old_resolution == args.new_resolution:
        parser.error("--old_resolution and --new_resolution must differ.")
    if not math.isfinite(args.psnr_threshold):
        parser.error("--psnr_threshold must be finite.")
    if args.gpu_id is not None and args.gpu_id < 0:
        parser.error("--gpu_id must be non-negative.")
    return args


def input_objaverse_root(dataset_root: Path, testset: bool = False) -> Path:
    source_directory = "test-set-exp" if testset else "train-set"
    return dataset_root / source_directory / "objaverse"


def refined_objaverse_root(dataset_root: Path, testset: bool = False) -> Path:
    output_directory = "test-set-4x-up" if testset else "train-set-4x-up"
    return dataset_root / output_directory / "objaverse"


def resolution_root(
    dataset_root: Path, resolution: int, testset: bool = False
) -> Path:
    return input_objaverse_root(dataset_root, testset) / str(resolution)


def source_checkpoint_path(
    dataset_root: Path,
    resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Path:
    return (
        resolution_root(dataset_root, resolution, testset)
        / "gsplat"
        / scene_id
        / "ckpts"
        / SOURCE_CHECKPOINT_NAME
    )


def source_stats_path(
    dataset_root: Path,
    resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Path:
    return (
        resolution_root(dataset_root, resolution, testset)
        / "gsplat"
        / scene_id
        / "stats"
        / SOURCE_STATS_NAME
    )


def refined_scene_dir(
    dataset_root: Path,
    old_resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Path:
    return (
        refined_objaverse_root(dataset_root, testset)
        / str(old_resolution)
        / "gsplat"
        / scene_id
    )


def refined_checkpoint_path(
    dataset_root: Path,
    old_resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Path:
    return (
        refined_scene_dir(dataset_root, old_resolution, scene_id, testset)
        / "ckpts"
        / REFINED_CHECKPOINT_NAME
    )


def csv_fieldnames(resolutions: Sequence[int]) -> List[str]:
    fields = ["scene_id"]
    for resolution in resolutions:
        prefix = f"res_{resolution}"
        fields.extend(
            [
                f"{prefix}_image_count",
                f"{prefix}_num_gs",
                f"{prefix}_psnr",
                f"{prefix}_ssim",
                f"{prefix}_lpips",
                f"{prefix}_ellipse_time",
            ]
        )
    return fields


def _child_directory_names(path: Path) -> set:
    if not path.is_dir():
        return set()
    return {entry.name for entry in path.iterdir() if entry.is_dir()}


def discover_scene_ids(
    dataset_root: Path,
    resolutions: Iterable[int],
    testset: bool = False,
) -> List[str]:
    scene_ids = set()
    for resolution in resolutions:
        root = resolution_root(dataset_root, resolution, testset)
        scene_ids.update(_child_directory_names(root / "colmap"))
        scene_ids.update(_child_directory_names(root / "gsplat"))
    return sorted(scene_ids)


def count_png_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    return sum(
        1
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def load_render_metrics(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing render metrics: {path}") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read render metrics {path}: {exc}") from exc

    if not isinstance(metrics, dict):
        raise ValueError(f"Render metrics must be a JSON object: {path}")
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"Render metrics {path} are missing keys: {missing}")

    validated: Dict[str, Any] = {}
    for key in REQUIRED_METRICS:
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Render metric {key!r} in {path} is not numeric.")
        if not math.isfinite(float(value)):
            raise ValueError(f"Render metric {key!r} in {path} is not finite.")
        validated[key] = value

    if int(validated["num_GS"]) != validated["num_GS"] or validated["num_GS"] <= 0:
        raise ValueError(f"Render metric 'num_GS' in {path} must be a positive integer.")
    validated["num_GS"] = int(validated["num_GS"])
    return validated


def inspect_scene_resolution(
    dataset_root: Path,
    resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Dict[str, Any]:
    images_dir = (
        resolution_root(dataset_root, resolution, testset)
        / "colmap"
        / scene_id
        / "images"
    )
    image_count = count_png_images(images_dir)
    if image_count != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} PNG images at {images_dir}, found {image_count}."
        )

    checkpoint_path = source_checkpoint_path(
        dataset_root, resolution, scene_id, testset
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    metrics = load_render_metrics(
        source_stats_path(dataset_root, resolution, scene_id, testset)
    )
    return {
        "image_count": image_count,
        "num_gs": metrics["num_GS"],
        "psnr": float(metrics["psnr"]),
        "ssim": float(metrics["ssim"]),
        "lpips": float(metrics["lpips"]),
        "ellipse_time": float(metrics["ellipse_time"]),
    }


def scan_valid_scenes(
    dataset_root: Path,
    resolutions: Sequence[int],
    testset: bool = False,
) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    invalid_reasons: Counter = Counter()

    for scene_id in discover_scene_ids(dataset_root, resolutions, testset):
        row: Dict[str, Any] = {"scene_id": scene_id}
        try:
            for resolution in resolutions:
                values = inspect_scene_resolution(
                    dataset_root, resolution, scene_id, testset
                )
                prefix = f"res_{resolution}"
                for name, value in values.items():
                    row[f"{prefix}_{name}"] = value
        except FileNotFoundError:
            invalid_reasons["missing_file"] += 1
            continue
        except OSError:
            invalid_reasons["filesystem_error"] += 1
            continue
        except ValueError:
            invalid_reasons["invalid_data"] += 1
            continue
        rows.append(row)

    return rows, invalid_reasons


def write_scene_csv(
    rows: Sequence[Dict[str, Any]], path: Path, resolutions: Sequence[int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = csv_fieldnames(resolutions)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def load_scene_csv(path: Path, resolutions: Sequence[int]) -> List[Dict[str, Any]]:
    expected_fields = csv_fieldnames(resolutions)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"CSV {path} has columns {reader.fieldnames}; expected {expected_fields}."
            )
        rows = list(reader)

    integer_fields = {
        f"res_{resolution}_image_count" for resolution in resolutions
    } | {f"res_{resolution}_num_gs" for resolution in resolutions}
    numeric_fields = set(expected_fields) - {"scene_id"} - integer_fields
    parsed_rows: List[Dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        try:
            parsed: Dict[str, Any] = {"scene_id": row["scene_id"]}
            if not parsed["scene_id"]:
                raise ValueError("scene_id is empty")
            for field in integer_fields:
                parsed[field] = int(row[field])
            for field in numeric_fields:
                parsed[field] = float(row[field])
                if not math.isfinite(parsed[field]):
                    raise ValueError(f"{field} is not finite")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid row {row_number} in {path}: {exc}") from exc
        parsed_rows.append(parsed)
    return parsed_rows


def filter_scenes(
    rows: Sequence[Dict[str, Any]], old_resolution: int, psnr_threshold: float
) -> List[Dict[str, Any]]:
    psnr_key = f"res_{old_resolution}_psnr"
    return [row for row in rows if float(row[psnr_key]) >= psnr_threshold]


def select_gpu(requested_gpu_id: Optional[int]) -> Optional[str]:
    if requested_gpu_id is not None:
        return str(requested_gpu_id)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"[ERROR] Failed to query GPUs: {exc}")
        return None

    for gpu_id, line in enumerate(result.stdout.splitlines()):
        if line.strip() == "0":
            return str(gpu_id)
    print("[ERROR] No idle GPU found.")
    return None


def build_refiner_command(
    dataset_root: Path,
    old_resolution: int,
    new_resolution: int,
    scene_id: str,
    refiner_path: Path,
    testset: bool = False,
) -> List[str]:
    init_checkpoint = source_checkpoint_path(
        dataset_root, old_resolution, scene_id, testset
    )
    old_data_dir = (
        resolution_root(dataset_root, old_resolution, testset)
        / "colmap"
        / scene_id
    )
    new_data_dir = (
        resolution_root(dataset_root, new_resolution, testset)
        / "colmap"
        / scene_id
    )
    result_dir = refined_scene_dir(
        dataset_root, old_resolution, scene_id, testset
    )
    return [
        sys.executable,
        str(refiner_path),
        "default",
        "--disable_viewer",
        f"--init-ckpt={init_checkpoint}",
        f"--old-data-dir={old_data_dir}",
        f"--new-data-dir={new_data_dir}",
        f"--result-dir={result_dir}",
        "--data-factor=1",
        "--test-every=-1",
        f"--max-steps={REFINER_STEPS}",
        f"--eval-steps={REFINER_STEPS}",
        f"--save-steps={REFINER_STEPS}",
        "--alpha-aware",
        "--disable-video",
        "--init-type=sfm",
        "--load-bbox",
        "--num-points-from-bbox=50000",
        "--no-normalize-world-space",
        "--batch-size",
        "4",
        "--sh-degree=1",
        "--tb-every=0",
        "--strategy.refine-start-iter=1000000",
    ]


def run_refinements(
    rows: Sequence[Dict[str, Any]],
    dataset_root: Path,
    old_resolution: int,
    new_resolution: int,
    gpu_id: Optional[int],
    force: bool,
    refiner_path: Path,
    testset: bool = False,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    statuses: Dict[str, str] = {}
    failures: List[Dict[str, Any]] = []
    pending: List[str] = []

    for row in rows:
        scene_id = str(row["scene_id"])
        final_checkpoint = refined_checkpoint_path(
            dataset_root, old_resolution, scene_id, testset
        )
        if final_checkpoint.is_file() and not force:
            statuses[scene_id] = "reused"
        else:
            pending.append(scene_id)

    selected_gpu = select_gpu(gpu_id) if pending else None
    if pending and selected_gpu is None:
        for scene_id in pending:
            statuses[scene_id] = "refinement_failed"
            failures.append(
                {
                    "scene_id": scene_id,
                    "stage": "refinement",
                    "error": "No GPU was available for refinement.",
                }
            )
        return statuses, failures

    env = dict(os.environ)
    if selected_gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = selected_gpu

    for scene_id in pending:
        command = build_refiner_command(
            dataset_root,
            old_resolution,
            new_resolution,
            scene_id,
            refiner_path,
            testset,
        )
        print(f"\n[REFINE] {scene_id} on GPU {selected_gpu}")
        print(" ".join(command))
        try:
            subprocess.run(command, env=env, check=True)
            final_checkpoint = refined_checkpoint_path(
                dataset_root, old_resolution, scene_id, testset
            )
            if not final_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Refiner completed without producing {final_checkpoint}"
                )
        except (OSError, subprocess.CalledProcessError) as exc:
            statuses[scene_id] = "refinement_failed"
            failure: Dict[str, Any] = {
                "scene_id": scene_id,
                "stage": "refinement",
                "error": str(exc),
            }
            if isinstance(exc, subprocess.CalledProcessError):
                failure["returncode"] = exc.returncode
            failures.append(failure)
            print(f"[ERROR] Refinement failed for {scene_id}: {exc}")
            continue
        statuses[scene_id] = "refined"

    return statuses, failures


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to calculate checkpoint statistics. Run this "
            "script in the GSplat training environment."
        ) from exc
    return torch


def load_splats(checkpoint_path: Path) -> Mapping:
    torch = _import_torch()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {checkpoint_path}")
    splats = checkpoint.get("splats")
    if not isinstance(splats, Mapping):
        raise ValueError(f"Checkpoint has no 'splats' mapping: {checkpoint_path}")
    missing = [key for key in REQUIRED_SPLAT_KEYS if key not in splats]
    if missing:
        raise ValueError(f"Checkpoint {checkpoint_path} is missing splats: {missing}")
    return splats


def _validate_splat_mapping(splats: Mapping, checkpoint_path: Path) -> Tuple[int, Dict[str, Tuple[int, ...]]]:
    torch = _import_torch()
    if not splats:
        raise ValueError(f"Checkpoint contains no splat parameters: {checkpoint_path}")

    num_splats: Optional[int] = None
    trailing_shapes: Dict[str, Tuple[int, ...]] = {}
    for key, value in splats.items():
        if not torch.is_tensor(value):
            raise TypeError(f"Splat {key!r} in {checkpoint_path} is not a tensor.")
        if not torch.is_floating_point(value):
            raise TypeError(f"Splat {key!r} in {checkpoint_path} is not floating point.")
        if value.ndim < 1:
            raise ValueError(f"Splat {key!r} in {checkpoint_path} has no row dimension.")
        if num_splats is None:
            num_splats = int(value.shape[0])
        elif value.shape[0] != num_splats:
            raise ValueError(
                f"Splat {key!r} in {checkpoint_path} has {value.shape[0]} rows; "
                f"expected {num_splats}."
            )
        if not torch.isfinite(value).all().item():
            raise ValueError(f"Splat {key!r} in {checkpoint_path} is not finite.")
        trailing_shapes[str(key)] = tuple(value.shape[1:])

    assert num_splats is not None
    if num_splats <= 0:
        raise ValueError(f"Checkpoint contains no splats: {checkpoint_path}")
    return num_splats, trailing_shapes


def tensor_statistics(value: Any) -> Dict[str, Any]:
    torch = _import_torch()
    value64 = value.detach().to(device="cpu", dtype=torch.float64)
    return {
        "mean": value64.mean(dim=0).tolist(),
        "std": value64.std(dim=0, unbiased=False).tolist(),
    }


def compute_scene_statistics(
    input_checkpoint: Path, output_checkpoint: Path
) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, ...]]]:
    torch = _import_torch()
    input_splats = load_splats(input_checkpoint)
    output_splats = load_splats(output_checkpoint)
    input_keys = set(input_splats.keys())
    output_keys = set(output_splats.keys())
    if input_keys != output_keys:
        raise ValueError(
            "Input/output splat keys differ: "
            f"missing_from_output={sorted(input_keys - output_keys)}, "
            f"missing_from_input={sorted(output_keys - input_keys)}."
        )

    input_count, input_shapes = _validate_splat_mapping(input_splats, input_checkpoint)
    output_count, output_shapes = _validate_splat_mapping(output_splats, output_checkpoint)
    if input_count != output_count:
        raise ValueError(
            f"Input/output splat counts differ: {input_count} != {output_count}."
        )
    if input_shapes != output_shapes:
        raise ValueError(
            f"Input/output splat parameter shapes differ: {input_shapes} != {output_shapes}."
        )

    result: Dict[str, Any] = {
        "num_splats": input_count,
        "input_checkpoint": str(input_checkpoint),
        "output_checkpoint": str(output_checkpoint),
        "input": {},
        "output": {},
        "delta": {},
    }
    ordered_keys = [key for key in REQUIRED_SPLAT_KEYS if key in input_keys]
    ordered_keys.extend(sorted(input_keys - set(ordered_keys)))
    for key in ordered_keys:
        input_value = input_splats[key]
        output_value = output_splats[key]
        result["input"][key] = tensor_statistics(input_value)
        result["output"][key] = tensor_statistics(output_value)
        delta = output_value.detach().to(torch.float64) - input_value.detach().to(torch.float64)
        result["delta"][key] = tensor_statistics(delta)
    return result, input_shapes


def _average_nested(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("Cannot average an empty value list.")
    first = values[0]
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in values):
            raise ValueError("Scene statistic shapes do not match.")
        return [
            _average_nested([value[index] for value in values])
            for index in range(len(first))
        ]
    if any(isinstance(value, list) for value in values):
        raise ValueError("Scene statistic shapes do not match.")
    return sum(float(value) for value in values) / len(values)


def aggregate_scene_statistics(scenes: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not scenes:
        return {
            "weighting": "equal_scene_average",
            "scene_count": 0,
            "input": {},
            "output": {},
            "delta": {},
        }

    scene_values = list(scenes.values())
    result: Dict[str, Any] = {
        "weighting": "equal_scene_average",
        "scene_count": len(scene_values),
    }
    for group in ("input", "output", "delta"):
        result[group] = {}
        parameter_keys = list(scene_values[0][group].keys())
        for key in parameter_keys:
            result[group][key] = {
                "mean": _average_nested(
                    [scene[group][key]["mean"] for scene in scene_values]
                ),
                "std": _average_nested(
                    [scene[group][key]["std"] for scene in scene_values]
                ),
            }
    return result


def calculate_statistics(
    rows: Sequence[Dict[str, Any]],
    statuses: Dict[str, str],
    dataset_root: Path,
    old_resolution: int,
    testset: bool = False,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    scenes: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    reference_shapes: Optional[Dict[str, Tuple[int, ...]]] = None

    for row in rows:
        scene_id = str(row["scene_id"])
        status = statuses.get(scene_id)
        if status not in {"refined", "reused"}:
            continue
        input_checkpoint = source_checkpoint_path(
            dataset_root, old_resolution, scene_id, testset
        )
        output_checkpoint = refined_checkpoint_path(
            dataset_root, old_resolution, scene_id, testset
        )
        try:
            scene_stats, shapes = compute_scene_statistics(
                input_checkpoint, output_checkpoint
            )
            if reference_shapes is None:
                reference_shapes = shapes
            elif shapes != reference_shapes:
                raise ValueError(
                    "Parameter channel shapes differ from earlier scenes: "
                    f"{shapes} != {reference_shapes}."
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            statuses[scene_id] = "statistics_failed"
            failures.append(
                {
                    "scene_id": scene_id,
                    "stage": "statistics",
                    "error": str(exc),
                }
            )
            print(f"[ERROR] Statistics failed for {scene_id}: {exc}")
            continue
        scene_stats["status"] = status
        scenes[scene_id] = scene_stats

    aggregate = aggregate_scene_statistics(scenes)
    return scenes, aggregate, failures


def existing_refinement_statuses(
    rows: Sequence[Dict[str, Any]],
    dataset_root: Path,
    old_resolution: int,
    testset: bool = False,
) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    for row in rows:
        scene_id = str(row["scene_id"])
        if refined_checkpoint_path(
            dataset_root, old_resolution, scene_id, testset
        ).is_file():
            statuses[scene_id] = "reused"
        else:
            statuses[scene_id] = "not_run"
    return statuses


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    resolutions = (args.old_resolution, args.new_resolution)
    train_root = input_objaverse_root(dataset_root, args.testset)
    if not train_root.is_dir():
        print(f"[ERROR] Input dataset directory does not exist: {train_root}")
        return 1

    metadata_prefix = "test_" if args.testset else ""
    valid_csv = dataset_root / f"{metadata_prefix}valid_scenes.csv"
    filtered_csv = dataset_root / f"{metadata_prefix}psnr_filtered_scenes.csv"
    statistics_json = dataset_root / f"{metadata_prefix}gs_statistics.json"
    stage_flags_specified = any(
        (args.scan, args.filter, args.refine, args.statistics)
    )
    run_refinement = args.refine or not stage_flags_specified
    needs_filtered_rows = (
        args.filter
        or args.refine
        or args.statistics
        or not stage_flags_specified
    )

    invalid_reasons: Counter = Counter()
    filtered_regenerated = False
    if args.scan or not valid_csv.is_file():
        print(f"Scanning paired scenes under {train_root}")
        valid_rows, invalid_reasons = scan_valid_scenes(
            dataset_root, resolutions, args.testset
        )
        write_scene_csv(valid_rows, valid_csv, resolutions)
        print(f"Wrote {valid_csv}")
    else:
        print(f"Reusing {valid_csv}")
        try:
            valid_rows = load_scene_csv(valid_csv, resolutions)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] Could not reuse valid scene list: {exc}")
            return 1

    print(f"Valid paired scenes: {len(valid_rows)}")
    if not needs_filtered_rows:
        return 0

    if args.filter or not filtered_csv.is_file():
        filtered_regenerated = True
        filtered_rows = filter_scenes(
            valid_rows, args.old_resolution, args.psnr_threshold
        )
        write_scene_csv(filtered_rows, filtered_csv, resolutions)
        print(f"Wrote {filtered_csv}")
    else:
        print(f"Reusing {filtered_csv}")
        try:
            filtered_rows = load_scene_csv(filtered_csv, resolutions)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] Could not reuse filtered scene list: {exc}")
            return 1

    print(
        f"Scenes with {args.old_resolution}px PSNR >= {args.psnr_threshold}: "
        f"{len(filtered_rows)}"
    )
    if stage_flags_specified and not args.refine and not args.statistics:
        return 0

    failures: List[Dict[str, Any]] = []
    if run_refinement:
        refiner_path = Path(__file__).resolve().with_name("refiner.py")
        if not refiner_path.is_file():
            print(f"[ERROR] Refiner does not exist: {refiner_path}")
            return 1
        statuses, failures = run_refinements(
            filtered_rows,
            dataset_root,
            args.old_resolution,
            args.new_resolution,
            args.gpu_id,
            args.force,
            refiner_path,
            args.testset,
        )
    else:
        statuses = existing_refinement_statuses(
            filtered_rows, dataset_root, args.old_resolution, args.testset
        )

    run_statistics = args.statistics or (
        not stage_flags_specified
        and (
            not statistics_json.is_file()
            or filtered_regenerated
            or bool(failures)
            or any(status == "refined" for status in statuses.values())
        )
    )
    if not run_statistics:
        if not stage_flags_specified and statistics_json.is_file():
            print(f"Reusing {statistics_json}")
        return 1 if failures else 0

    scenes, aggregate, statistics_failures = calculate_statistics(
        filtered_rows,
        statuses,
        dataset_root,
        args.old_resolution,
        args.testset,
    )
    failures.extend(statistics_failures)

    status_counts = Counter(statuses.values())
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "dataset_root": str(dataset_root),
            "input_root": str(train_root),
            "refined_root": str(refined_objaverse_root(dataset_root, args.testset)),
            "dataset_split": "test" if args.testset else "train",
            "old_resolution": args.old_resolution,
            "new_resolution": args.new_resolution,
            "expected_image_count": EXPECTED_IMAGE_COUNT,
            "psnr_threshold": args.psnr_threshold,
            "psnr_filter_resolution": args.old_resolution,
            "source_checkpoint_name": SOURCE_CHECKPOINT_NAME,
            "refined_checkpoint_name": REFINED_CHECKPOINT_NAME,
            "refiner_steps": REFINER_STEPS,
            "refiner_batch_size": 4,
            "refiner_interpolation": False,
            "difference": "output_minus_input",
            "raw_parameter_space": True,
            "aggregate_weighting": "equal_scene_average",
            "valid_scenes_csv": str(valid_csv),
            "filtered_scenes_csv": str(filtered_csv),
        },
        "summary": {
            "discovered_scenes": len(valid_rows) + sum(invalid_reasons.values()),
            "valid_scenes": len(valid_rows),
            "filtered_scenes": len(filtered_rows),
            "statistics_scenes": len(scenes),
            "status_counts": dict(sorted(status_counts.items())),
            "failure_count": len(failures),
            "invalid_scene_count": sum(invalid_reasons.values()),
        },
        "invalid_scene_reasons": dict(sorted(invalid_reasons.items())),
        "aggregate": aggregate,
        "scenes": scenes,
        "failures": failures,
    }
    write_json(result, statistics_json)
    print(f"Wrote {statistics_json}")
    if failures:
        print(f"[ERROR] Completed with {len(failures)} failed scene stages.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
