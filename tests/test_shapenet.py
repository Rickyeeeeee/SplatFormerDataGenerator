import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import gen_data_shapenet as gen
import preprocess_shapenet as prep


class ManifestTests(unittest.TestCase):
    def test_determinism_stratification_and_small_categories(self):
        ids = [f'03001627/{i:04d}' for i in range(100)] + ['02691156/a', '02691156/b']
        a = gen.create_manifest(ids, 42)
        self.assertEqual(a, gen.create_manifest(list(reversed(ids)), 42))
        self.assertEqual([list(a['assignments'].values()).count(x) for x in ('train', 'val', 'test')], [92, 5, 5])
        self.assertEqual(a['assignments']['02691156/a'], 'train')
        self.assertNotEqual(a, gen.create_manifest(ids, 43))

    def test_manifest_persistence_and_inventory_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'split.json'
            ids = ['03001627/a', '03001627/b', '03001627/c']
            first = gen.load_manifest(path, ids, 42)
            self.assertEqual(first, gen.load_manifest(path, ids, 99))
            with self.assertRaises(ValueError): gen.load_manifest(path, ids + ['03001627/d'], 42)
            self.assertEqual(len(gen.load_manifest(path, ids + ['03001627/d'], 42, True)['inventory']), 4)

    def test_discovery_includes_missing_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '03001627' / 'abc').mkdir(parents=True)
            (root / 'not_category' / 'other').mkdir(parents=True)
            self.assertEqual(gen.discover_inventory(root), ['03001627/abc'])

    def test_defaults_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = gen.parse_args(['--shapenet-root', tmp])
            self.assertEqual(args.resolutions, [128])
            self.assertEqual(args.max_splats, 4096)
            command = gen.render_command(args, Path('/mesh.obj'), Path('/out'), 128)
            self.assertIn('--train_views=128', command)
            self.assertTrue(command[command.index('--python') + 1].endswith('render_shapenet.py'))
            train = gen.common.build_train_command(Path('/images'), Path('/out'), '03001627/a', args.max_splats)
            self.assertIn('--strategy.cap-max=4096', train)
            self.assertIn('--num-points-from-bbox=2048', train)

    def test_configurable_training_steps_and_names(self):
        command = gen.common.build_train_command(Path('/images'), Path('/out'), 'scene', max_steps=2000)
        for option in ('--max-steps=2000', '--save-steps=2000', '--eval-steps=2000', '--strategy.refine-stop-iter=2000'):
            self.assertIn(option, command)
        self.assertEqual(gen.checkpoint_name(2000), 'ckpt_1999_rank0.pt')
        self.assertEqual(gen.metrics_name(2000), 'val_step1999.json')
        self.assertEqual(gen.checkpoint_name(1), 'ckpt_0_rank0.pt')
        self.assertEqual(gen.metrics_name(1), 'val_step0000.json')
        self.assertEqual(gen.checkpoint_name(10000), gen.common.CHECKPOINT_NAME)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit): gen.parse_args(['--shapenet-root', tmp, '--train-steps', '0'])
            self.assertEqual(gen.parse_args(['--shapenet-root', tmp, '--train-steps', '2000']).train_steps, 2000)
        with self.assertRaises(SystemExit): prep.parse_args(['--train-steps', '0'])

    def test_strategy_roots_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ['--shapenet-root', tmp]
            self.assertEqual(gen.parse_args(base).gs_dataset_root, Path('/project/ricky/shapenet-sr-data-mcmc-4096'))
            default = gen.parse_args(base + ['--strategy', 'default'])
            self.assertEqual(default.gs_dataset_root, Path('/project/ricky/shapenet-sr-data-default-4096'))
            custom = gen.parse_args(base + ['--strategy', 'default', '--max-splats', '8000'])
            self.assertEqual(custom.gs_dataset_root, Path('/project/ricky/shapenet-sr-data-default-8000'))
            override = ['--strategy', 'default', '--gs-dataset-root', '/tmp/custom-gs']
            self.assertEqual(gen.parse_args(base + override).gs_dataset_root, Path('/tmp/custom-gs'))
            self.assertEqual(prep.parse_args(override).gs_dataset_root, Path('/tmp/custom-gs'))
            self.assertEqual(prep.parse_args(['--strategy', 'default', '--max-splats', '8000']).gs_dataset_root, custom.gs_dataset_root)

    def test_default_strategy_training_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = gen.parse_args(['--shapenet-root', tmp, '--strategy', 'default',
                                   '--max-splats', '4096', '--train-steps', '2000',
                                   '--image-dataset-root', str(root / 'images'),
                                   '--gs-dataset-root', str(root / 'gs')])
            scene = '03001627/a'
            mesh = root / scene / 'models/model_normalized.obj'
            mesh.parent.mkdir(parents=True)
            mesh.touch()
            def train(command, **kwargs):
                self.assertEqual(command[2], 'default')
                self.assertIn('--strategy.cap-max=4096', command)
                self.assertIn('--num-points-from-bbox=2048', command)
                self.assertIn('--max-steps=2000', command)
                out = gen.scene_root(args.gs_dataset_root, 'train', 128, 'gsplat', scene)
                (out / 'ckpts').mkdir(parents=True)
                (out / 'stats').mkdir()
                (out / 'ckpts' / gen.checkpoint_name(2000)).write_text('checkpoint')
                (out / 'stats' / gen.metrics_name(2000)).write_text('{}')
            with mock.patch.object(gen, 'render_complete', return_value=True), \
                 mock.patch.object(gen.subprocess, 'run', side_effect=train) as run:
                gen.process_scene(args, scene, 'train', {})
                gen.process_scene(args, scene, 'train', {})
            run.assert_called_once()


class PipelineTests(unittest.TestCase):
    def test_nested_scan_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root / 'image'; gs = root / 'gs'; scene = '03001627/abc'
            img_dir = gen.scene_root(image, 'val', 128, 'colmap', scene) / 'images'
            img_dir.mkdir(parents=True)
            for i in range(128): (img_dir / f'{i:03}.png').touch()
            gs_dir = gen.scene_root(gs, 'val', 128, 'gsplat', scene)
            (gs_dir / 'ckpts').mkdir(parents=True)
            (gs_dir / 'ckpts' / gen.checkpoint_name(gen.DEFAULT_TRAIN_STEPS)).touch()
            (gs_dir / 'stats').mkdir()
            (gs_dir / 'stats' / gen.metrics_name(gen.DEFAULT_TRAIN_STEPS)).write_text(json.dumps({'num_GS': 10, 'psnr': 25, 'ssim': .9, 'lpips': .1, 'ellipse_time': 1}))
            rows, invalid = prep.scan_scenes(image, gs, 'val', 128)
            self.assertEqual(invalid, {})
            self.assertEqual(rows[0]['synset_id'], '03001627')
            self.assertEqual(rows[0]['scene_id'], scene)
            csv_path = root / 'scenes.csv'
            prep.write_csv(rows, csv_path)
            self.assertEqual(prep.read_csv(csv_path), rows)
            self.assertEqual(prep.common.filter_scenes(rows, 26), [])
            self.assertEqual(prep.discover_scenes(image, gs, 'train', 128), [])

    def test_incomplete_render_does_not_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(gen.render_complete(root))
            (root / 'images').mkdir()
            for i in range(128): (root / 'images' / f'{i:03}.png').touch()
            self.assertFalse(gen.render_complete(root))

    def test_missing_mesh_fails_independently(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = gen.parse_args(['--shapenet-root', tmp])
            with self.assertRaises(FileNotFoundError): gen.process_scene(args, '03001627/missing', 'train', {})

    def test_resume_training_and_downscale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = gen.parse_args(['--shapenet-root', tmp, '--image-dataset-root', str(root / 'images'),
                                   '--gs-dataset-root', str(root / 'gs'), '--resolutions', '128', '64'])
            scene = '03001627/a'
            mesh = root / scene / 'models/model_normalized.obj'
            mesh.parent.mkdir(parents=True)
            mesh.touch()
            native = gen.scene_root(args.image_dataset_root, 'train', 128, 'colmap', scene)
            target = gen.scene_root(args.image_dataset_root, 'train', 64, 'colmap', scene)
            for resolution in (128, 64):
                out = gen.scene_root(args.gs_dataset_root, 'train', resolution, 'gsplat', scene)
                (out / 'ckpts').mkdir(parents=True)
                (out / 'stats').mkdir()
                (out / 'ckpts' / gen.checkpoint_name(gen.DEFAULT_TRAIN_STEPS)).write_text('checkpoint')
                (out / 'stats' / gen.metrics_name(gen.DEFAULT_TRAIN_STEPS)).write_text('{}')
            with mock.patch.object(gen, 'render_complete', side_effect=[True, False, True]), \
                 mock.patch.object(gen.common, 'downscale_scene') as downscale, \
                 mock.patch.object(gen.subprocess, 'run') as run:
                gen.process_scene(args, scene, 'train', {})
            downscale.assert_called_once_with(native, target, 128, 64)
            run.assert_not_called()

    def test_statistics_uses_nested_checkpoint(self):
        with mock.patch.object(prep.common, 'compute_scene_statistics', return_value=({'raw': {}}, {'means': (3,)})) as calc, \
             mock.patch.object(prep.common, 'aggregate_scene_statistics', return_value={}):
            scenes, _, failures = prep.calculate_statistics([{'scene_id': '03001627/a'}], Path('/gs'), 'test', 128, 4096, 0)
            self.assertEqual(failures, [])
            self.assertEqual(list(scenes), ['03001627/a'])
            self.assertEqual(calc.call_args.args[0], Path('/gs/test-set/shapenet/128/gsplat/03001627/a/ckpts') / gen.checkpoint_name(gen.DEFAULT_TRAIN_STEPS))

    def test_custom_step_scan_and_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root / 'image'; gs = root / 'gs'; scene = '03001627/abc'
            images = gen.scene_root(image, 'train', 128, 'colmap', scene) / 'images'
            images.mkdir(parents=True)
            for i in range(128): (images / f'{i:03}.png').touch()
            output = gen.scene_root(gs, 'train', 128, 'gsplat', scene)
            (output / 'ckpts').mkdir(parents=True)
            (output / 'stats').mkdir()
            (output / 'ckpts' / gen.checkpoint_name(20)).write_text('checkpoint')
            (output / 'stats' / gen.metrics_name(20)).write_text(json.dumps({'num_GS': 10, 'psnr': 25, 'ssim': .9, 'lpips': .1, 'ellipse_time': 1}))
            self.assertEqual(len(prep.scan_scenes(image, gs, 'train', 128, 20)[0]), 1)
            self.assertEqual(len(prep.scan_scenes(image, gs, 'train', 128)[0]), 0)
            with mock.patch.object(prep.common, 'compute_scene_statistics', return_value=({'raw': {}}, {'means': (3,)})) as calc, \
                 mock.patch.object(prep.common, 'aggregate_scene_statistics', return_value={}):
                prep.calculate_statistics([{'scene_id': scene}], gs, 'train', 128, 4096, 0, 20)
            self.assertEqual(calc.call_args.args[0], output / 'ckpts' / 'ckpt_19_rank0.pt')
            self.assertEqual(prep.main(['--image-dataset-root', str(image), '--gs-dataset-root', str(gs), '--train-steps', '20', '--scan']), 0)
            self.assertTrue((gs / 'train-set/shapenet/128/valid_scenes_step19.csv').is_file())


if __name__ == '__main__': unittest.main()
