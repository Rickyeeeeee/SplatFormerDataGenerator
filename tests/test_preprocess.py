import csv
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import preprocess


RESOLUTION = 64


def create_scene(
    image_root: Path,
    gs_root: Path,
    scene_id: str,
    *,
    resolution: int = RESOLUTION,
    image_count: int = 128,
    psnr: float = 30.0,
    testset: bool = False,
) -> None:
    split = "test-set" if testset else "train-set"
    images_dir = image_root / split / "objaverse" / str(resolution) / "colmap" / scene_id / "images"
    images_dir.mkdir(parents=True)
    for index in range(image_count):
        (images_dir / f"{index:04d}.png").touch()
    scene_root = gs_root / split / "objaverse" / str(resolution) / "gsplat" / scene_id
    checkpoint = scene_root / "ckpts" / preprocess.SOURCE_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    metrics = scene_root / "stats" / preprocess.SOURCE_STATS_NAME
    metrics.parent.mkdir(parents=True)
    metrics.write_text(json.dumps({
        "num_GS": 10, "psnr": psnr, "ssim": 0.9,
        "lpips": 0.1, "ellipse_time": 0.01,
    }), encoding="utf-8")


def cli_args(image_root: Path, gs_root: Path, **overrides):
    values = {
        "image_dataset_root": image_root,
        "gs_dataset_root": gs_root,
        "resolution": RESOLUTION,
        "psnr_threshold": 27.0,
        "scan": False,
        "filter": False,
        "statistics": False,
        "testset": False,
        "statistics_max_gs_num": 1_000_000,
        "statistics_remove_outlier_ndevs": 0.0,
    }
    values.update(overrides)
    return preprocess.argparse.Namespace(**values)


class ScanTests(unittest.TestCase):
    def test_separate_roots_psnr_boundary_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, gs = root / "images", root / "gs"
            create_scene(images, gs, "b_below", psnr=26.999)
            create_scene(images, gs, "a_boundary", psnr=27.0)
            create_scene(images, gs, "c_bad_images", image_count=127)
            create_scene(images, gs, "d_missing_checkpoint")
            preprocess.source_checkpoint_path(gs, RESOLUTION, "d_missing_checkpoint").unlink()
            create_scene(images, gs, "e_bad_metrics")
            preprocess.source_stats_path(gs, RESOLUTION, "e_bad_metrics").write_text('{}')

            rows, invalid = preprocess.scan_valid_scenes(images, gs, RESOLUTION)
            self.assertEqual([row["scene_id"] for row in rows], ["a_boundary", "b_below"])
            self.assertEqual(sum(invalid.values()), 3)
            self.assertEqual([row["scene_id"] for row in preprocess.filter_scenes(rows, 27.0)], ["a_boundary"])
            self.assertEqual(rows[0]["image_count"], 128)
            self.assertEqual(rows[0]["psnr"], 27.0)

            csv_path = root / "valid.csv"
            preprocess.write_scene_csv(rows, csv_path)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                self.assertEqual(csv.DictReader(handle).fieldnames, preprocess.csv_fieldnames())
            self.assertEqual(preprocess.load_scene_csv(csv_path), rows)

    def test_resolution_and_split_paths_do_not_overlap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, gs = root / "images", root / "gs"
            create_scene(images, gs, "train_scene")
            create_scene(images, gs, "test_scene", testset=True)
            create_scene(images, gs, "other_resolution", resolution=128)
            train_rows, _ = preprocess.scan_valid_scenes(images, gs, RESOLUTION)
            test_rows, _ = preprocess.scan_valid_scenes(images, gs, RESOLUTION, True)
            self.assertEqual([row["scene_id"] for row in train_rows], ["train_scene"])
            self.assertEqual([row["scene_id"] for row in test_rows], ["test_scene"])
            self.assertEqual([row["scene_id"] for row in preprocess.scan_valid_scenes(images, gs, 128)[0]], ["other_resolution"])


class WorkflowTests(unittest.TestCase):
    def test_statistics_accepts_mixed_shn_presence(self):
        rows = [{"scene_id": "without"}, {"scene_id": "with"}]
        shapes = {"means": (3,), "scales": (3,), "quats": (4,),
                  "opacities": (), "sh0": (1, 3)}
        summaries = [
            ({"raw": {"means": {"mean": [0.0], "std": [0.0]}}}, shapes),
            ({"raw": {"means": {"mean": [1.0], "std": [0.0]},
                      "shN": {"mean": [], "std": []}}},
             {**shapes, "shN": (0, 3)}),
        ]
        with mock.patch("preprocess.compute_scene_statistics", side_effect=summaries), \
             mock.patch("preprocess.aggregate_scene_statistics", return_value={}) as aggregate:
            scenes, _, failures = preprocess.calculate_statistics(rows, Path("/gs"), RESOLUTION)
        self.assertEqual(set(scenes), {"with", "without"})
        self.assertEqual(failures, [])
        aggregate.assert_called_once()

    def test_cli_defaults_to_mcmc_resolution_and_roots(self):
        with mock.patch.object(sys, "argv", ["preprocess.py"]):
            args = preprocess.parse_args()
        self.assertEqual(args.resolution, 64)
        self.assertEqual(args.image_dataset_root, preprocess.DEFAULT_IMAGE_DATASET_ROOT)
        self.assertEqual(args.gs_dataset_root, preprocess.DEFAULT_GS_DATASET_ROOT)
        self.assertFalse(hasattr(args, "refine"))

    def test_scan_only_writes_metadata_under_gs_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, gs = root / "images", root / "gs"
            create_scene(images, gs, "scene", testset=True)
            with mock.patch("preprocess.parse_args", return_value=cli_args(images, gs, scan=True, testset=True)):
                self.assertEqual(preprocess.main(), 0)
            metadata = gs / "test-set" / "objaverse" / str(RESOLUTION)
            self.assertTrue((metadata / "valid_scenes.csv").is_file())
            self.assertFalse((metadata / "psnr_filtered_scenes.csv").exists())
            self.assertFalse((metadata / "gs_statistics.json").exists())
            self.assertFalse((images / "test-set" / "objaverse" / str(RESOLUTION) / "valid_scenes.csv").exists())

    def test_filter_only_uses_cached_scan_and_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, gs = root / "images", root / "gs"
            create_scene(images, gs, "below", psnr=26.0)
            create_scene(images, gs, "above", psnr=28.0)
            metadata = gs / "train-set" / "objaverse" / str(RESOLUTION)
            rows, _ = preprocess.scan_valid_scenes(images, gs, RESOLUTION)
            preprocess.write_scene_csv(rows, metadata / "valid_scenes.csv")
            with mock.patch("preprocess.parse_args", return_value=cli_args(images, gs, filter=True)):
                self.assertEqual(preprocess.main(), 0)
            self.assertEqual([row["scene_id"] for row in preprocess.load_scene_csv(metadata / "psnr_filtered_scenes.csv")], ["above"])
            self.assertFalse((metadata / "gs_statistics.json").exists())

    def test_statistics_only_builds_prerequisites_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, gs = root / "images", root / "gs"
            create_scene(images, gs, "scene")
            with mock.patch("preprocess.parse_args", return_value=cli_args(images, gs, statistics=True)), \
                 mock.patch("preprocess.compute_scene_statistics", side_effect=ValueError("bad checkpoint")):
                self.assertEqual(preprocess.main(), 1)
            metadata = gs / "train-set" / "objaverse" / str(RESOLUTION)
            report = json.loads((metadata / "gs_statistics.json").read_text())
            self.assertEqual(report["summary"]["failure_count"], 1)
            self.assertEqual(report["summary"]["filtered_scenes"], 1)
            self.assertEqual(report["config"]["resolution"], RESOLUTION)
            self.assertNotIn("refined_root", report["config"])
            self.assertTrue((metadata / "valid_scenes.csv").exists())
            self.assertTrue((metadata / "psnr_filtered_scenes.csv").exists())


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch is not installed")
class StatisticsTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def splats(self):
        torch = self.torch
        return {
            "means": torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
            "scales": torch.zeros((2, 3)),
            "quats": torch.ones((2, 4)),
            "opacities": torch.tensor([0.0, 2.0]),
            "sh0": torch.zeros((2, 1, 3)),
            "shN": torch.zeros((2, 0, 3)),
        }

    def save_checkpoint(self, path: Path, splats) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"step": 9999, "splats": splats}, path)

    def test_single_checkpoint_raw_and_normalized_statistics(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "scene.pt"
            self.save_checkpoint(checkpoint, self.splats())
            stats, shapes = preprocess.compute_scene_statistics(checkpoint)
            self.assertEqual(stats["num_splats"], 2)
            self.assertEqual(stats["checkpoint"], str(checkpoint))
            self.assertEqual(shapes["means"], (3,))
            self.assertEqual(stats["raw"]["means"], {"mean": [2.0, 3.0, 4.0], "std": [1.0, 1.0, 1.0]})
            self.assertEqual(stats["normalized"]["parameters"]["means"]["mean"], [0.5, 0.5, 0.5])
            self.assertEqual(stats["normalized"]["parameters"]["means"]["variance"], [0.25, 0.25, 0.25])
            self.assertEqual(stats["normalized"]["selected_gaussian_count"], 2)
            self.assertNotIn("delta", stats)

    def test_checkpoint_without_shn_is_valid_but_sh0_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "scene.pt"
            splats = self.splats()
            del splats["shN"]
            self.save_checkpoint(checkpoint, splats)
            stats, shapes = preprocess.compute_scene_statistics(checkpoint)
            self.assertNotIn("shN", shapes)
            self.assertNotIn("shN", stats["raw"])
            self.assertNotIn("shN", stats["normalized"]["parameters"])
            self.assertEqual(stats["raw"]["means"]["mean"], [2.0, 3.0, 4.0])

            del splats["sh0"]
            self.save_checkpoint(checkpoint, splats)
            with self.assertRaisesRegex(ValueError, "sh0"):
                preprocess.compute_scene_statistics(checkpoint)

    def test_nonfinite_is_rejected_and_one_selected_splat_is_centered(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "scene.pt"
            nonfinite = self.splats()
            nonfinite["scales"][0, 0] = float("nan")
            self.save_checkpoint(checkpoint, nonfinite)
            with self.assertRaisesRegex(ValueError, "not finite"):
                preprocess.compute_scene_statistics(checkpoint)
            self.save_checkpoint(checkpoint, self.splats())
            stats, _ = preprocess.compute_scene_statistics(checkpoint, statistics_max_gs_num=1)
            self.assertEqual(stats["normalized"]["parameters"]["means"]["mean"], [0.5, 0.5, 0.5])
            self.assertEqual(stats["normalized"]["selected_gaussian_count"], 1)

    def test_aggregate_equal_scene_mixture(self):
        scenes = {
            "a": {"raw": {"means": {"mean": [1.0], "std": [2.0]}},
                  "normalized": {"parameters": {"means": {"mean": [0.0], "variance": [1.0]}}}},
            "b": {"raw": {"means": {"mean": [5.0], "std": [6.0]}},
                  "normalized": {"parameters": {"means": {"mean": [2.0], "variance": [1.0]}}}},
        }
        result = preprocess.aggregate_scene_statistics(scenes)
        self.assertEqual(result["raw"]["means"], {"mean": [3.0], "std": [4.0]})
        self.assertEqual(result["normalized"]["parameters"]["means"]["variance"], [2.0])
        self.assertEqual(result["normalized"]["parameters"]["means"]["std"], [math.sqrt(2.0)])

    def test_aggregate_optional_shn_uses_only_present_scenes(self):
        present = self.splats()
        absent = self.splats()
        del absent["shN"]
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first.pt"
            second = Path(temporary) / "second.pt"
            self.save_checkpoint(first, absent)
            self.save_checkpoint(second, present)
            first_stats, _ = preprocess.compute_scene_statistics(first)
            second_stats, _ = preprocess.compute_scene_statistics(second)
            aggregate = preprocess.aggregate_scene_statistics({"without": first_stats, "with": second_stats})
            self.assertEqual(aggregate["scene_count"], 2)
            self.assertEqual(aggregate["parameter_scene_counts"]["means"], 2)
            self.assertEqual(aggregate["parameter_scene_counts"]["shN"], 1)
            self.assertIn("shN", aggregate["raw"])
            self.assertIn("shN", aggregate["normalized"]["parameters"])

    def test_default_pipeline_writes_single_checkpoint_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, gs = root / "images", root / "gs"
            create_scene(images, gs, "scene")
            checkpoint = preprocess.source_checkpoint_path(gs, RESOLUTION, "scene")
            self.save_checkpoint(checkpoint, self.splats())
            with mock.patch("preprocess.parse_args", return_value=cli_args(images, gs)):
                self.assertEqual(preprocess.main(), 0)
            report_path = gs / "train-set" / "objaverse" / str(RESOLUTION) / "gs_statistics.json"
            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"]["statistics_scenes"], 1)
            self.assertEqual(report["scenes"]["scene"]["num_splats"], 2)
            self.assertEqual(report["aggregate"]["scene_count"], 1)
            self.assertNotIn("input", report["aggregate"])


if __name__ == "__main__":
    unittest.main()
