#!/usr/bin/env python3

import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_IMAGE_DATASET_ROOT = Path("/project/ricky/splatformer-sr-data-scaled")
DEFAULT_GS_DATASET_ROOT = Path("/project/ricky/splatformer-sr-data-mcmc-4000")
DEFAULT_RESOLUTION = 128
DEFAULT_PSNR_THRESHOLD = 23.0
EXPECTED_IMAGE_COUNT = 128
SOURCE_CHECKPOINT_NAME = "ckpt_9999_rank0.pt"
SOURCE_STATS_NAME = "val_step9999.json"
REQUIRED_METRICS = ("num_GS", "psnr", "ssim", "lpips", "ellipse_time")
REQUIRED_SPLAT_KEYS = ("means", "scales", "quats", "opacities", "sh0")
OPTIONAL_SPLAT_KEYS = ("shN",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan single-resolution MCMC GSplat scenes, filter by PSNR, "
            "and calculate raw and scene-normalized checkpoint statistics."
        )
    )
    parser.add_argument(
        "--image_dataset_root",
        type=Path,
        default=DEFAULT_IMAGE_DATASET_ROOT,
        help=f"Root containing COLMAP images (default: {DEFAULT_IMAGE_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--gs_dataset_root",
        type=Path,
        default=DEFAULT_GS_DATASET_ROOT,
        help=f"Root containing MCMC checkpoints (default: {DEFAULT_GS_DATASET_ROOT}).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=DEFAULT_RESOLUTION,
        help=f"Image and GS resolution (default: {DEFAULT_RESOLUTION}).",
    )
    parser.add_argument(
        "--psnr_threshold",
        type=float,
        default=DEFAULT_PSNR_THRESHOLD,
        help="Inclusive PSNR threshold (default: 27).",
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
        "--statistics",
        action="store_true",
        help="Regenerate gs_statistics.json from filtered MCMC checkpoints.",
    )
    parser.add_argument(
        "--testset",
        action="store_true",
        help="Use test-set images and MCMC checkpoints instead of train-set.",
    )
    parser.add_argument("--statistics_max_gs_num", type=int, default=1_000_000, help="Maximum selected Gaussians for normalized statistics (default: 1000000).")
    parser.add_argument("--statistics_remove_outlier_ndevs", type=float, default=0.0, help="Position outlier threshold in standard deviations; 0 disables filtering.")
    args = parser.parse_args()

    if args.resolution <= 0:
        parser.error("--resolution must be positive.")
    if not math.isfinite(args.psnr_threshold):
        parser.error("--psnr_threshold must be finite.")
    if args.statistics_max_gs_num <= 0:
        parser.error("--statistics_max_gs_num must be positive.")
    if not math.isfinite(args.statistics_remove_outlier_ndevs) or args.statistics_remove_outlier_ndevs < 0:
        parser.error("--statistics_remove_outlier_ndevs must be finite and non-negative.")
    return args


def input_objaverse_root(dataset_root: Path, testset: bool = False) -> Path:
    split_directory = "test-set" if testset else "train-set"
    return dataset_root / split_directory / "objaverse"


def resolution_root(
    dataset_root: Path, resolution: int, testset: bool = False
) -> Path:
    return input_objaverse_root(dataset_root, testset) / str(resolution)


def source_checkpoint_path(
    gs_dataset_root: Path,
    resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Path:
    return (
        resolution_root(gs_dataset_root, resolution, testset)
        / "gsplat"
        / scene_id
        / "ckpts"
        / SOURCE_CHECKPOINT_NAME
    )


def source_stats_path(
    gs_dataset_root: Path,
    resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Path:
    return (
        resolution_root(gs_dataset_root, resolution, testset)
        / "gsplat"
        / scene_id
        / "stats"
        / SOURCE_STATS_NAME
    )


def csv_fieldnames() -> List[str]:
    return ["scene_id", "image_count", "num_gs", "psnr", "ssim", "lpips", "ellipse_time"]


def _child_directory_names(path: Path) -> set:
    if not path.is_dir():
        return set()
    return {entry.name for entry in path.iterdir() if entry.is_dir()}


def discover_scene_ids(
    image_dataset_root: Path,
    gs_dataset_root: Path,
    resolution: int,
    testset: bool = False,
) -> List[str]:
    image_root = resolution_root(image_dataset_root, resolution, testset)
    gs_root = resolution_root(gs_dataset_root, resolution, testset)
    scene_ids = _child_directory_names(image_root / "colmap")
    scene_ids.update(_child_directory_names(gs_root / "gsplat"))
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
    image_dataset_root: Path,
    gs_dataset_root: Path,
    resolution: int,
    scene_id: str,
    testset: bool = False,
) -> Dict[str, Any]:
    images_dir = (
        resolution_root(image_dataset_root, resolution, testset)
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
        gs_dataset_root, resolution, scene_id, testset
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    metrics = load_render_metrics(
        source_stats_path(gs_dataset_root, resolution, scene_id, testset)
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
    image_dataset_root: Path,
    gs_dataset_root: Path,
    resolution: int,
    testset: bool = False,
) -> Tuple[List[Dict[str, Any]], Counter]:
    rows: List[Dict[str, Any]] = []
    invalid_reasons: Counter = Counter()

    for scene_id in discover_scene_ids(
        image_dataset_root, gs_dataset_root, resolution, testset
    ):
        row: Dict[str, Any] = {"scene_id": scene_id}
        try:
            row.update(inspect_scene_resolution(
                image_dataset_root, gs_dataset_root, resolution, scene_id, testset
            ))
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
    rows: List[Dict[str, Any]], path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = csv_fieldnames()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def load_scene_csv(path: Path) -> List[Dict[str, Any]]:
    expected_fields = csv_fieldnames()
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"CSV {path} has columns {reader.fieldnames}; expected {expected_fields}."
            )
        rows = list(reader)

    integer_fields = {"image_count", "num_gs"}
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
    rows: List[Dict[str, Any]], psnr_threshold: float
) -> List[Dict[str, Any]]:
    return [row for row in rows if float(row["psnr"]) >= psnr_threshold]


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
    checkpoint_path: Path,
    statistics_max_gs_num: int = 1_000_000,
    statistics_remove_outlier_ndevs: float = 0.0,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[int, ...]]]:
    torch = _import_torch()
    splats = load_splats(checkpoint_path)
    count, shapes = _validate_splat_mapping(splats, checkpoint_path)

    result: Dict[str, Any] = {
        "num_splats": count,
        "checkpoint": str(checkpoint_path),
        "raw": {},
    }
    keys = set(splats)
    ordered_keys = [key for key in (*REQUIRED_SPLAT_KEYS, *OPTIONAL_SPLAT_KEYS) if key in keys]
    ordered_keys.extend(sorted(keys - set(ordered_keys)))
    for key in ordered_keys:
        result["raw"][key] = tensor_statistics(splats[key])
    means = splats["means"]
    mask = torch.ones(count, dtype=torch.bool)
    if statistics_remove_outlier_ndevs > 0:
        center = means.mean(dim=0)
        radius = statistics_remove_outlier_ndevs * means.std(dim=0)
        mask &= ((means >= center - radius) & (means <= center + radius)).all(dim=1)
    indices = mask.nonzero(as_tuple=False).squeeze(1)[:statistics_max_gs_num]
    if indices.numel() == 0:
        raise ValueError("Gaussian filtering removed every splat")
    selected_means = means[indices]

    # Preserve aspect ratio and center each axis inside the unit cube.
    minimum = selected_means.min(dim=0).values
    maximum = selected_means.max(dim=0).values
    extent = (maximum - minimum).max()
    scale = 1.0 / extent if extent > 0 else torch.ones_like(extent)
    scaled_means = selected_means * scale
    translation = 0.5 - (scaled_means.min(dim=0).values + scaled_means.max(dim=0).values) / 2
    normalized = {
        "selected_gaussian_count": int(indices.numel()),
        "scale": scale.tolist(),
        "translation": translation.tolist(),
        "parameters": {},
    }
    for key in ordered_keys:
        value = splats[key][indices]
        if key == "means":
            value = value * scale + translation
        elif key == "scales":
            value = value + torch.log(scale)
        summary = tensor_statistics(value)
        summary["variance"] = value.to(torch.float64).var(dim=0, unbiased=False).tolist()
        normalized["parameters"][key] = summary
    result["normalized"] = normalized
    return result, shapes


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
            "parameter_scene_counts": {},
            "raw": {},
            "normalized": {"weighting": "equal_scene_mixture", "scene_count": 0, "parameters": {}},
        }

    scene_values = list(scenes.values())
    result: Dict[str, Any] = {
        "weighting": "equal_scene_average",
        "scene_count": len(scene_values),
    }
    result["raw"] = {}
    parameter_keys = list(dict.fromkeys(
        key for scene in scene_values for key in scene["raw"]
    ))
    result["parameter_scene_counts"] = {}
    for key in parameter_keys:
        matching_scenes = [scene for scene in scene_values if key in scene["raw"]]
        result["parameter_scene_counts"][key] = len(matching_scenes)
        result["raw"][key] = {
            "mean": _average_nested([scene["raw"][key]["mean"] for scene in matching_scenes]),
            "std": _average_nested([scene["raw"][key]["std"] for scene in matching_scenes]),
        }
    # Combine within-scene variance and between-scene mean variation equally by scene.
    torch = _import_torch()
    normalized = {"weighting": "equal_scene_mixture", "scene_count": len(scene_values), "parameters": {}}
    for key in parameter_keys:
        matching_scenes = [scene for scene in scene_values if key in scene["normalized"]["parameters"]]
        means = torch.tensor([scene["normalized"]["parameters"][key]["mean"] for scene in matching_scenes], dtype=torch.float64)
        variances = torch.tensor([scene["normalized"]["parameters"][key]["variance"] for scene in matching_scenes], dtype=torch.float64)
        mean = means.mean(dim=0)
        variance = (variances + (means - mean).square()).mean(dim=0)
        normalized["parameters"][key] = {"mean": mean.tolist(), "variance": variance.tolist(), "std": variance.sqrt().tolist()}
    result["normalized"] = normalized
    return result


def calculate_statistics(
    rows: Sequence[Dict[str, Any]],
    gs_dataset_root: Path,
    resolution: int,
    testset: bool = False,
    statistics_max_gs_num: int = 1_000_000,
    statistics_remove_outlier_ndevs: float = 0.0,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    scenes: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, Any]] = []
    reference_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
    optional_shapes: Dict[str, Tuple[int, ...]] = {}

    for row in rows:
        scene_id = str(row["scene_id"])
        checkpoint = source_checkpoint_path(
            gs_dataset_root, resolution, scene_id, testset
        )
        try:
            scene_stats, shapes = compute_scene_statistics(
                checkpoint, statistics_max_gs_num, statistics_remove_outlier_ndevs
            )
            required_shapes = {
                key: shape for key, shape in shapes.items()
                if key not in OPTIONAL_SPLAT_KEYS
            }
            if reference_shapes is None:
                reference_shapes = required_shapes
            elif required_shapes != reference_shapes:
                raise ValueError(
                    "Parameter channel shapes differ from earlier scenes: "
                    f"{required_shapes} != {reference_shapes}."
                )
            for key in OPTIONAL_SPLAT_KEYS:
                if key in shapes:
                    if key in optional_shapes and shapes[key] != optional_shapes[key]:
                        raise ValueError(
                            f"Optional parameter {key} shape {shapes[key]} differs "
                            f"from earlier scenes: {optional_shapes[key]}."
                        )
                    optional_shapes[key] = shapes[key]
        except (EOFError, OSError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                {
                    "scene_id": scene_id,
                    "stage": "statistics",
                    "error": str(exc),
                }
            )
            print(f"[ERROR] Statistics failed for {scene_id}: {exc}")
            continue
        scenes[scene_id] = scene_stats

    aggregate = aggregate_scene_statistics(scenes)
    return scenes, aggregate, failures


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    image_dataset_root = args.image_dataset_root.resolve()
    gs_dataset_root = args.gs_dataset_root.resolve()
    image_root = resolution_root(image_dataset_root, args.resolution, args.testset)
    gs_root = resolution_root(gs_dataset_root, args.resolution, args.testset)
    if not (image_root / "colmap").is_dir():
        print(f"[ERROR] Image directory does not exist: {image_root / 'colmap'}")
        return 1
    if not (gs_root / "gsplat").is_dir():
        print(f"[ERROR] MCMC GS directory does not exist: {gs_root / 'gsplat'}")
        return 1

    valid_csv = gs_root / "valid_scenes.csv"
    filtered_csv = gs_root / "psnr_filtered_scenes.csv"
    statistics_json = gs_root / "gs_statistics.json"
    run_all = not any((args.scan, args.filter, args.statistics))
    run_scan = args.scan or run_all or not valid_csv.is_file()
    run_filter = args.filter or args.statistics or run_all
    run_statistics = args.statistics or run_all

    invalid_reasons: Counter = Counter()
    if run_scan:
        print(f"Scanning {args.resolution}px MCMC scenes")
        valid_rows, invalid_reasons = scan_valid_scenes(
            image_dataset_root, gs_dataset_root, args.resolution, args.testset
        )
        write_scene_csv(valid_rows, valid_csv)
        print(f"Wrote {valid_csv}")
    else:
        print(f"Reusing {valid_csv}")
        try:
            valid_rows = load_scene_csv(valid_csv)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] Could not reuse valid scene list: {exc}")
            return 1

    print(f"Valid scenes: {len(valid_rows)}")
    if not run_filter:
        return 0

    filtered_rows = filter_scenes(valid_rows, args.psnr_threshold)
    write_scene_csv(filtered_rows, filtered_csv)
    print(f"Wrote {filtered_csv}")

    print(
        f"Scenes with {args.resolution}px PSNR >= {args.psnr_threshold}: "
        f"{len(filtered_rows)}"
    )
    if not run_statistics:
        return 0

    scenes, aggregate, failures = calculate_statistics(
        filtered_rows,
        gs_dataset_root,
        args.resolution,
        args.testset,
        statistics_max_gs_num=args.statistics_max_gs_num,
        statistics_remove_outlier_ndevs=args.statistics_remove_outlier_ndevs,
    )

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "image_dataset_root": str(image_dataset_root),
            "gs_dataset_root": str(gs_dataset_root),
            "dataset_split": "test" if args.testset else "train",
            "resolution": args.resolution,
            "expected_image_count": EXPECTED_IMAGE_COUNT,
            "psnr_threshold": args.psnr_threshold,
            "checkpoint_name": SOURCE_CHECKPOINT_NAME,
            "render_metrics_name": SOURCE_STATS_NAME,
            "raw_parameter_space": True,
            "aggregate_weighting": "equal_scene_average",
            "normalized_statistics": {
                "feature_range": [0.0, 1.0],
                "preserve_ratio": True,
                "scaler_fit": "selected_means",
                "max_gs_num": args.statistics_max_gs_num,
                "remove_outlier_ndevs": args.statistics_remove_outlier_ndevs,
                "scale_representation": "log_scale_plus_log_coordinate_scale",
                "aggregate_weighting": "equal_scene_mixture",
                "variance": "mean_of_population_variances_plus_squared_scene_mean_offsets",
            },
            "valid_scenes_csv": str(valid_csv),
            "filtered_scenes_csv": str(filtered_csv),
        },
        "summary": {
            "discovered_scenes": len(valid_rows) + sum(invalid_reasons.values()),
            "valid_scenes": len(valid_rows),
            "filtered_scenes": len(filtered_rows),
            "statistics_scenes": len(scenes),
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
