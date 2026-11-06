import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import gen_data_shapenet as gen
import preprocess_shapenet as prep


def create_scene(images, gs, resolution, scene, psnr=30, checkpoint=True):
    image = gen.scene_root(images, resolution, 'colmap', scene)
    (image / 'images').mkdir(parents=True)
    sparse = image / 'sparse/0'
    sparse.mkdir(parents=True)
    lines = []
    for index in range(128):
        name = f'{index:03}.png'
        (image / 'images' / name).write_bytes(b'png')
        lines.append(f'{index} 1 0 0 0 0 0 3 1 {name}\n\n')
    (image / 'metadata.json').write_text('{"normalization": {}}')
    (sparse / 'images.txt').write_text(''.join(lines))
    (sparse / 'cameras.txt').write_text(f'1 SIMPLE_PINHOLE {resolution} {resolution} 100 50 50\n')
    (sparse / 'bbox.txt').write_text('-1 -1 -1\n1 1 1\n')
    output = gen.scene_root(gs, resolution, 'gsplat', scene)
    (output / 'ckpts').mkdir(parents=True)
    (output / 'stats').mkdir()
    if checkpoint: (output / 'ckpts' / gen.checkpoint_name(gen.DEFAULT_TRAIN_STEPS)).write_text('checkpoint')
    (output / 'stats' / gen.metrics_name(gen.DEFAULT_TRAIN_STEPS)).write_text(json.dumps(
        {'num_GS': 10, 'psnr': psnr, 'ssim': .9, 'lpips': .1, 'ellipse_time': 1}))


class SplitTests(unittest.TestCase):
    def test_stratification_and_ratios(self):
        ids = [f'03001627/{i:04d}' for i in range(100)] + ['02691156/a', '02691156/b']
        first = prep.split_scenes(ids, [0.9, 0.05, 0.05], 42)
        self.assertEqual(first, prep.split_scenes(reversed(ids), [0.9, 0.05, 0.05], 42))
        self.assertEqual([list(first.values()).count(x) for x in prep.SPLITS], [92, 5, 5])
        self.assertNotEqual(first, prep.split_scenes(ids, [0.9, 0.05, 0.05], 43))
        only_train = prep.split_scenes(ids, [1, 0, 0], 1)
        self.assertEqual(set(only_train.values()), {'train'})
        with self.assertRaises(SystemExit): prep.parse_args(['--split-ratios', '1', '1', '1'])

    def test_multi_resolution_counts_csvs_and_train_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); images = root / 'images'; gs = root / 'gs'
            for res in (128, 64):
                for scene in ('a', 'b', 'c', 'd'):
                    create_scene(images, gs, res, '03001627/' + scene, psnr=20 if (res == 64 and scene == 'd') else 30)
            create_scene(images, gs, 128, '03001627/e', psnr=20)
            create_scene(images, gs, 128, '03001627/x', checkpoint=False)
            create_scene(images, gs, 64, '03001627/y', checkpoint=False)
            selected = []
            def stats(rows, root, resolution, max_count, outlier, steps):
                selected.append((resolution, [r['scene_id'] for r in rows]))
                return {}, {}, []
            with mock.patch.object(prep, 'calculate_statistics', side_effect=stats):
                code = prep.main(['--image-dataset-root', str(images), '--gs-dataset-root', str(gs), '--resolutions', '128', '64'])
            self.assertEqual(code, 0)
            report = json.loads((gs / 'shapenet/gen_data.json').read_text())
            self.assertEqual(report['passing_all_resolutions'], 3)
            self.assertEqual(report['split_counts'], {'train': 1, 'val': 1, 'test': 1})
            self.assertEqual(report['resolutions']['128']['discovered'], 6)
            self.assertEqual(report['resolutions']['128']['valid_before_psnr'], 5)
            self.assertEqual(report['resolutions']['128']['passing_psnr'], 4)
            self.assertEqual(report['resolutions']['128']['rejected_psnr'], 1)
            self.assertEqual(report['resolutions']['128']['invalid'], 1)
            self.assertEqual(report['resolutions']['64']['passing_psnr'], 3)
            self.assertEqual(report['resolutions']['64']['rejected_psnr'], 1)
            self.assertEqual(report['resolutions']['64']['invalid'], 1)
            for split in prep.SPLITS:
                a = (gs / 'shapenet/128' / f'{split}_scenes.csv').read_text().splitlines()
                b = (gs / 'shapenet/64' / f'{split}_scenes.csv').read_text().splitlines()
                self.assertEqual(len(a), 2)
                self.assertEqual(a[1].split(',')[0], b[1].split(',')[0])
            self.assertEqual(selected[0][1], selected[1][1])
            self.assertEqual(len(selected[0][1]), 1)
            self.assertEqual(prep.main(['--image-dataset-root', str(images), '--gs-dataset-root', str(gs), '--resolutions', '128', '64', '--split-ratios', '1', '0', '0', '--skip-statistics']), 0)
            self.assertEqual(json.loads((gs / 'shapenet/gen_data.json').read_text())['split_counts']['train'], 3)
            self.assertEqual(json.loads((gs / 'shapenet/128/gs_statistics.json').read_text())['status'], 'skipped')
            self.assertEqual(len((gs / 'shapenet/128/val_scenes.csv').read_text().splitlines()), 1)


class GeneratorTests(unittest.TestCase):
    def test_inventory_only_scans_selected_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for category in ('03001627', '02691156'):
                (root / category / 'model').mkdir(parents=True)
            self.assertEqual(gen.discover_inventory(root, ['03001627']), ['03001627/model'])
            self.assertEqual(len(gen.discover_inventory(root)), 2)

    def test_preprocessing_counts_images_without_render_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = '03001627/a'
            create_scene(root / 'images', root / 'gs', 128, scene)
            image = gen.scene_root(root / 'images', 128, 'colmap', scene)
            self.assertTrue(gen.render_complete(image))
            (image / 'images/000.png').write_bytes(b'')
            (image / 'metadata.json').unlink()
            self.assertTrue(gen.render_complete(image))
            rows, invalid = prep.scan_scenes(root / 'images', root / 'gs', 128)
            self.assertEqual(len(rows), 1)
            self.assertEqual(invalid, {})
            (image / 'images/000.png').unlink()
            self.assertFalse(gen.render_complete(image))
            rows, invalid = prep.scan_scenes(root / 'images', root / 'gs', 128)
            self.assertEqual(rows, [])
            self.assertEqual(invalid['invalid_data'], 1)

    def test_filesystem_errors_do_not_trigger_incomplete_render(self):
        with mock.patch.object(gen.os, 'scandir', side_effect=PermissionError('unavailable')):
            with self.assertRaises(PermissionError):
                gen.render_complete(Path('/render'))

    def test_strategy_roots_and_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = gen.parse_args(['--shapenet-root', tmp, '--strategy', 'default', '--gpu-ids', '0', '2', '--scenes-per-gpu', '2'])
            self.assertEqual(args.gs_dataset_root, Path('/project/ricky/shapenet-sr-data-default-4096'))
            self.assertEqual(args.gpu_ids, [0, 2])
            self.assertEqual(args.scenes_per_gpu, 2)
            self.assertEqual(gen.scene_root(Path('/images'), 128, 'colmap', '03001627/a'), Path('/images/shapenet/128/colmap/03001627/a'))
            with self.assertRaises(SystemExit): gen.parse_args(['--shapenet-root', tmp, '--gpu-ids', '0', '0'])

    def test_gpu_selection_respects_existing_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = gen.parse_args(['--shapenet-root', tmp])
            result = subprocess.CompletedProcess([], 0, stdout='0, 0\n1, 15\n2, 0\n')
            with mock.patch.object(gen.subprocess, 'run', return_value=result), \
                 mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2'}):
                self.assertEqual(gen.select_gpus(args), [2])
                args.gpu_ids = [0]
                with self.assertRaises(ValueError): gen.select_gpus(args)

    def test_schedule_limits_and_one_gpu_per_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = gen.parse_args(['--shapenet-root', tmp, '--image-dataset-root', str(root / 'images'), '--gs-dataset-root', str(root / 'gs'), '--scenes-per-gpu', '2'])
            active = {0: 0, 1: 0}; peak = {0: 0, 1: 0}; seen = {}; lock = threading.Lock()
            def process(args, scene, env, run):
                gpu = int(env['CUDA_VISIBLE_DEVICES'])
                with lock:
                    active[gpu] += 1; peak[gpu] = max(active[gpu], peak[gpu]); seen[scene] = gpu
                time.sleep(.03)
                with lock: active[gpu] -= 1
                if scene.endswith('/fail'): raise RuntimeError('training failed')
            scenes = [f'03001627/{i}' for i in range(8)] + ['03001627/fail']
            with mock.patch.object(gen, 'process_scene', side_effect=process):
                failures = gen.run_scenes(args, scenes, [0, 1])
            self.assertEqual(set(seen), set(scenes))
            self.assertEqual(peak, {0: 2, 1: 2})
            self.assertEqual(failures[0]['scene_id'], '03001627/fail')
            self.assertEqual(failures[0]['gpu_id'], seen['03001627/fail'])

    def test_locks_prevent_concurrent_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = gen.parse_args(['--shapenet-root', tmp, '--image-dataset-root', str(root / 'images'), '--gs-dataset-root', str(root / 'gs')])
            with gen.lock_scene(args, '03001627/a'):
                with self.assertRaises(RuntimeError):
                    with gen.lock_scene(args, '03001627/a'): pass
            with gen.lock_scene(args, '03001627/a'): pass

    def test_interruption_terminates_child(self):
        children = gen.ChildProcesses()
        errors = []
        def run():
            try: children.run([sys.executable, '-c', 'import time; time.sleep(30)'])
            except subprocess.CalledProcessError as exc: errors.append(exc)
        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.time() + 5
        while not children.processes and time.time() < deadline: time.sleep(.01)
        children.stop(); thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)

    def test_resume_default_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = gen.parse_args(['--shapenet-root', tmp, '--strategy', 'default', '--image-dataset-root', str(root / 'images'), '--gs-dataset-root', str(root / 'gs')])
            scene = '03001627/a'
            mesh = root / scene / 'models/model_normalized.obj'
            mesh.parent.mkdir(parents=True); mesh.touch()
            out = gen.scene_root(args.gs_dataset_root, 128, 'gsplat', scene)
            def run(command, **kwargs):
                self.assertEqual(command[2], 'default')
                self.assertIn('--strategy.cap-max=4096', command)
                (out / 'ckpts').mkdir(parents=True)
                (out / 'ckpts' / gen.checkpoint_name(args.train_steps)).write_text('ok')
            with mock.patch.object(gen, 'render_complete', return_value=True):
                gen.process_scene(args, scene, {}, run=run)
                gen.process_scene(args, scene, {}, run=lambda *a, **k: self.fail('already complete'))


if __name__ == '__main__': unittest.main()
