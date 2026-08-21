import csv
from contextlib import ExitStack
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import preprocess


RESOLUTIONS = (128, 512)


def create_scene(
    root: Path,
    scene_id: str,
    resolution: int,
    *,
    image_count: int = 128,
    psnr: float = 30.0,
    testset: bool = False,
) -> None:
    source_directory = "test-set" if testset else "train-set"
    resolution_root = root / source_directory / "objaverse" / str(resolution)
    images_dir = resolution_root / "colmap" / scene_id / "images"
    images_dir.mkdir(parents=True)
    for index in range(image_count):
        (images_dir / f"{index:04d}.png").touch()
    scene_root = resolution_root / "gsplat" / scene_id
    checkpoint = scene_root / "ckpts" / preprocess.SOURCE_CHECKPOINT_NAME
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    stats = scene_root / "stats" / preprocess.SOURCE_STATS_NAME
    stats.parent.mkdir(parents=True)
    stats.write_text(
        json.dumps(
            {
                "num_GS": 10 + resolution,
                "psnr": psnr,
                "ssim": 0.9,
                "lpips": 0.1,
                "ellipse_time": 0.01,
            }
        ),
        encoding="utf-8",
    )


class ScanTests(unittest.TestCase):
    def test_scan_filter_and_csv_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for resolution in RESOLUTIONS:
                create_scene(root, "b_below", resolution, psnr=26.999 if resolution == 128 else 31)
                create_scene(root, "a_boundary", resolution, psnr=27.0 if resolution == 128 else 31)
                create_scene(root, "c_invalid", resolution, image_count=127 if resolution == 512 else 128)

            rows, invalid = preprocess.scan_valid_scenes(root, RESOLUTIONS)
            self.assertEqual([row["scene_id"] for row in rows], ["a_boundary", "b_below"])
            self.assertEqual(sum(invalid.values()), 1)
            filtered = preprocess.filter_scenes(rows, 128, 27.0)
            self.assertEqual([row["scene_id"] for row in filtered], ["a_boundary"])

            output = root / "valid_scenes.csv"
            preprocess.write_scene_csv(rows, output, RESOLUTIONS)
            with output.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                csv_rows = list(reader)
                self.assertEqual(reader.fieldnames, preprocess.csv_fieldnames(RESOLUTIONS))
            self.assertEqual([row["scene_id"] for row in csv_rows], ["a_boundary", "b_below"])
            self.assertEqual(csv_rows[0]["res_128_psnr"], "27.0")

    def test_missing_metric_key_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for resolution in RESOLUTIONS:
                create_scene(root, "scene", resolution)
            stats_path = preprocess.source_stats_path(root, 512, "scene")
            metrics = json.loads(stats_path.read_text(encoding="utf-8"))
            del metrics["lpips"]
            stats_path.write_text(json.dumps(metrics), encoding="utf-8")

            rows, invalid = preprocess.scan_valid_scenes(root, RESOLUTIONS)
            self.assertEqual(rows, [])
            self.assertEqual(sum(invalid.values()), 1)


class RefinementTests(unittest.TestCase):
    def test_command_uses_established_schedule_and_paths(self):
        root = Path("/dataset")
        command = preprocess.build_refiner_command(
            root, 128, 512, "scene", Path("/repo/refiner.py")
        )
        self.assertEqual(command[:3], [preprocess.sys.executable, "/repo/refiner.py", "default"])
        self.assertIn("--max-steps=3000", command)
        self.assertIn("--eval-steps=3000", command)
        self.assertIn("--save-steps=3000", command)
        self.assertIn("--batch-size", command)
        self.assertIn("--strategy.refine-start-iter=1000000", command)
        self.assertNotIn("--interp", command)
        self.assertIn(
            "--result-dir=/dataset/train-set-4x-up/objaverse/128/gsplat/scene",
            command,
        )

    def test_resume_run_and_failure_recording(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            reused = preprocess.refined_checkpoint_path(root, 128, "reused")
            reused.parent.mkdir(parents=True)
            reused.touch()
            rows = [
                {"scene_id": "reused"},
                {"scene_id": "refined"},
                {"scene_id": "failed"},
            ]

            def fake_run(command, **kwargs):
                if command[0] == "nvidia-smi":
                    return subprocess.CompletedProcess(command, 0, stdout="0\n", stderr="")
                result_arg = next(arg for arg in command if arg.startswith("--result-dir="))
                scene_id = Path(result_arg.split("=", 1)[1]).name
                if scene_id == "failed":
                    raise subprocess.CalledProcessError(2, command)
                checkpoint = preprocess.refined_checkpoint_path(root, 128, scene_id)
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("preprocess.subprocess.run", side_effect=fake_run) as run:
                statuses, failures = preprocess.run_refinements(
                    rows,
                    root,
                    128,
                    512,
                    None,
                    False,
                    Path("/repo/refiner.py"),
                )

            self.assertEqual(statuses["reused"], "reused")
            self.assertEqual(statuses["refined"], "refined")
            self.assertEqual(statuses["failed"], "refinement_failed")
            self.assertEqual(failures[0]["returncode"], 2)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertFalse(any("reused" in " ".join(command) for command in commands[1:]))


class StageFlagTests(unittest.TestCase):
    def args(self, root: Path, **overrides):
        values = {
            "dataset_root": root,
            "old_resolution": 128,
            "new_resolution": 512,
            "psnr_threshold": 27.0,
            "gpu_id": 0,
            "force": False,
            "scan": False,
            "filter": False,
            "refine": False,
            "statistics": False,
            "testset": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def write_lists(self, root: Path):
        for resolution in RESOLUTIONS:
            create_scene(root, "scene", resolution)
        rows, _ = preprocess.scan_valid_scenes(root, RESOLUTIONS)
        preprocess.write_scene_csv(rows, root / "valid_scenes.csv", RESOLUTIONS)
        preprocess.write_scene_csv(
            rows, root / "psnr_filtered_scenes.csv", RESOLUTIONS
        )
        return rows

    def test_testset_scan_uses_separate_input_and_metadata_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for resolution in RESOLUTIONS:
                create_scene(root, "test_scene", resolution, testset=True)
            with mock.patch(
                "preprocess.parse_args",
                return_value=self.args(root, scan=True, testset=True),
            ):
                result = preprocess.main()

            self.assertEqual(result, 0)
            self.assertTrue((root / "test_valid_scenes.csv").is_file())
            self.assertFalse((root / "valid_scenes.csv").exists())
            rows = preprocess.load_scene_csv(
                root / "test_valid_scenes.csv", RESOLUTIONS
            )
            self.assertEqual([row["scene_id"] for row in rows], ["test_scene"])

    def test_testset_refiner_command_uses_test_directories(self):
        command = preprocess.build_refiner_command(
            Path("/dataset"),
            128,
            512,
            "scene",
            Path("/repo/refiner.py"),
            testset=True,
        )
        joined = " ".join(command)
        self.assertIn("/dataset/test-set/objaverse/128/", joined)
        self.assertIn("/dataset/test-set/objaverse/512/", joined)
        self.assertIn(
            "/dataset/test-set-4x-up/objaverse/128/gsplat/scene",
            joined,
        )
        self.assertNotIn("/dataset/train-set/", joined)

    def test_scan_only_does_not_run_later_stages(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            (root / "train-set" / "objaverse").mkdir(parents=True)
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "preprocess.parse_args",
                        return_value=self.args(root, scan=True),
                    )
                )
                scan = stack.enter_context(
                    mock.patch(
                        "preprocess.scan_valid_scenes",
                        return_value=([], preprocess.Counter()),
                    )
                )
                filter_stage = stack.enter_context(
                    mock.patch("preprocess.filter_scenes")
                )
                refine = stack.enter_context(
                    mock.patch("preprocess.run_refinements")
                )
                statistics = stack.enter_context(
                    mock.patch("preprocess.calculate_statistics")
                )
                result = preprocess.main()

            self.assertEqual(result, 0)
            scan.assert_called_once()
            filter_stage.assert_not_called()
            refine.assert_not_called()
            statistics.assert_not_called()
            self.assertTrue((root / "valid_scenes.csv").is_file())

    def test_refine_reuses_existing_lists_and_does_not_run_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.write_lists(root)
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "preprocess.parse_args",
                        return_value=self.args(root, refine=True),
                    )
                )
                scan = stack.enter_context(
                    mock.patch("preprocess.scan_valid_scenes")
                )
                filter_stage = stack.enter_context(
                    mock.patch("preprocess.filter_scenes")
                )
                refine = stack.enter_context(
                    mock.patch(
                        "preprocess.run_refinements",
                        return_value=({"scene": "reused"}, []),
                    )
                )
                statistics = stack.enter_context(
                    mock.patch("preprocess.calculate_statistics")
                )
                result = preprocess.main()

            self.assertEqual(result, 0)
            scan.assert_not_called()
            filter_stage.assert_not_called()
            refine.assert_called_once()
            statistics.assert_not_called()

    def test_filter_flag_regenerates_only_filtered_list(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rows = self.write_lists(root)
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "preprocess.parse_args",
                        return_value=self.args(root, filter=True),
                    )
                )
                scan = stack.enter_context(
                    mock.patch("preprocess.scan_valid_scenes")
                )
                filter_stage = stack.enter_context(
                    mock.patch("preprocess.filter_scenes", return_value=rows)
                )
                refine = stack.enter_context(
                    mock.patch("preprocess.run_refinements")
                )
                result = preprocess.main()

            self.assertEqual(result, 0)
            scan.assert_not_called()
            filter_stage.assert_called_once()
            refine.assert_not_called()

    def test_statistics_uses_existing_refined_checkpoints_without_refining(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.write_lists(root)
            checkpoint = preprocess.refined_checkpoint_path(root, 128, "scene")
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            empty_aggregate = preprocess.aggregate_scene_statistics({})
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "preprocess.parse_args",
                        return_value=self.args(root, statistics=True),
                    )
                )
                refine = stack.enter_context(
                    mock.patch("preprocess.run_refinements")
                )
                statistics = stack.enter_context(
                    mock.patch(
                        "preprocess.calculate_statistics",
                        return_value=({}, empty_aggregate, []),
                    )
                )
                result = preprocess.main()

            self.assertEqual(result, 0)
            refine.assert_not_called()
            statistics.assert_called_once()
            self.assertTrue((root / "gs_statistics.json").is_file())

    def test_no_flags_reuses_existing_statistics_when_no_scene_was_refined(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.write_lists(root)
            statistics_path = root / "gs_statistics.json"
            statistics_path.write_text('{"cached": true}\n', encoding="utf-8")
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "preprocess.parse_args",
                        return_value=self.args(root),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "preprocess.run_refinements",
                        return_value=({"scene": "reused"}, []),
                    )
                )
                statistics = stack.enter_context(
                    mock.patch("preprocess.calculate_statistics")
                )
                result = preprocess.main()

            self.assertEqual(result, 0)
            statistics.assert_not_called()
            self.assertEqual(
                json.loads(statistics_path.read_text(encoding="utf-8")),
                {"cached": True},
            )


    def test_no_flags_refreshes_statistics_after_rebuilding_filtered_list(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            self.write_lists(root)
            (root / "psnr_filtered_scenes.csv").unlink()
            statistics_path = root / "gs_statistics.json"
            statistics_path.write_text('{"cached": true}\n', encoding="utf-8")
            empty_aggregate = preprocess.aggregate_scene_statistics({})
            with ExitStack() as stack:
                stack.enter_context(
                    mock.patch(
                        "preprocess.parse_args",
                        return_value=self.args(root),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "preprocess.run_refinements",
                        return_value=({"scene": "reused"}, []),
                    )
                )
                statistics = stack.enter_context(
                    mock.patch(
                        "preprocess.calculate_statistics",
                        return_value=({}, empty_aggregate, []),
                    )
                )
                result = preprocess.main()

            self.assertEqual(result, 0)
            statistics.assert_called_once()
            self.assertNotEqual(
                json.loads(statistics_path.read_text(encoding="utf-8")),
                {"cached": True},
            )


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
            "shN": torch.zeros((2, 3, 3)),
        }

    def save_checkpoint(self, path: Path, splats) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"step": 0, "splats": splats}, path)

    def test_input_output_and_delta_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "input.pt"
            output_path = root / "output.pt"
            input_splats = self.splats()
            output_splats = {key: value.clone() for key, value in input_splats.items()}
            output_splats["means"] += 2.0
            output_splats["opacities"] += self.torch.tensor([1.0, 3.0])
            self.save_checkpoint(input_path, input_splats)
            self.save_checkpoint(output_path, output_splats)

            stats, shapes = preprocess.compute_scene_statistics(input_path, output_path)
            self.assertEqual(stats["num_splats"], 2)
            self.assertEqual(shapes["means"], (3,))
            self.assertEqual(stats["input"]["means"]["mean"], [2.0, 3.0, 4.0])
            self.assertEqual(stats["input"]["means"]["std"], [1.0, 1.0, 1.0])
            self.assertEqual(stats["delta"]["means"]["mean"], [2.0, 2.0, 2.0])
            self.assertEqual(stats["delta"]["means"]["std"], [0.0, 0.0, 0.0])
            self.assertEqual(stats["delta"]["opacities"]["mean"], 2.0)
            self.assertEqual(stats["delta"]["opacities"]["std"], 1.0)

    def test_shape_mismatch_and_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_path = root / "input.pt"
            output_path = root / "output.pt"
            self.save_checkpoint(input_path, self.splats())
            mismatched = self.splats()
            mismatched["means"] = mismatched["means"][:1]
            self.save_checkpoint(output_path, mismatched)
            with self.assertRaisesRegex(ValueError, "rows|counts"):
                preprocess.compute_scene_statistics(input_path, output_path)

            nonfinite = self.splats()
            nonfinite["scales"][0, 0] = float("nan")
            self.save_checkpoint(output_path, nonfinite)
            with self.assertRaisesRegex(ValueError, "not finite"):
                preprocess.compute_scene_statistics(input_path, output_path)

    def test_equal_scene_aggregate_averages_scene_statistics(self):
        scene_a = {
            group: {"means": {"mean": [1.0, 3.0], "std": [2.0, 4.0]}}
            for group in ("input", "output", "delta")
        }
        scene_b = {
            group: {"means": {"mean": [5.0, 7.0], "std": [6.0, 8.0]}}
            for group in ("input", "output", "delta")
        }
        aggregate = preprocess.aggregate_scene_statistics({"a": scene_a, "b": scene_b})
        self.assertEqual(aggregate["input"]["means"]["mean"], [3.0, 5.0])
        self.assertEqual(aggregate["input"]["means"]["std"], [4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
