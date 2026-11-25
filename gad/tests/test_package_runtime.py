"""Small export fixtures: source-only delivery never inspects local models."""
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import package_runtime as exporter


class RuntimePackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'LMOps'
        for name in ('train.yaml', 'README.md', '.gitignore', 'gad/.gitignore',
                     'gad/PIPELINE.md', 'gad/ONLINE_RUBRIC.md', 'gad/LICENSE',
                     'gad/setup.py', 'rubric/README.md', 'rubric/SMOKE_RESULTS.md',
                     'archive/README.md', 'assets/README.md', 'assets/manifest.json'):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}\n' if path.suffix == '.json' else '# fixture\n')
        source = self.root / 'third_party/verl/SOURCE.json'
        source.parent.mkdir(parents=True)
        source.write_text(json.dumps({'refs': {'pinned/gad': 'fixture'}}))
        (self.root / 'rubric/core').mkdir()
        (self.root / 'rubric/core/__init__.py').write_text('')
        (self.root / 'gad/extensions').mkdir()
        (self.root / 'gad/extensions/core').symlink_to('../../rubric/core')
        self.models = [self.root / 'models' / name for name in ('student', 'critic', 'judge')]
        data = dict(zip(('models.student', 'models.discriminator_init',
                         'rubric.judge.tokenizer_path'), map(str, self.models)))
        data.update({'rubric.enabled': True, 'rubric.mode': 'joint'})
        self.data_paths = []
        for key, name in (('data.train_file', 'train.parquet'), ('data.val_file', 'dev.parquet'),
                          ('data.teacher_index', 'index.jsonl'), ('data.manifest', 'manifest.json'),
                          ('validation.quick_file', 'quick.parquet')):
            path = self.root / 'data/math' / name
            data[key] = str(path)
            self.data_paths.append(path)
        self.cfg = SimpleNamespace(require=data.__getitem__, get=data.get)
        self.output = self.root.parent / 'export'

    def export(self, include_assets=False):
        with patch.object(exporter, 'ROOT', self.root), patch.object(exporter, 'load_config', return_value=self.cfg):
            return exporter.package(self.root / 'train.yaml', self.output, include_assets=include_assets)

    def make_assets(self):
        for model in self.models:
            model.mkdir(parents=True)
            (model / 'model.safetensors').write_bytes(b'tiny fixture weights')
        for path in self.data_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'tiny fixture data')

    def test_source_only_does_not_inspect_models_or_hash_individual_files(self):
        self.make_assets()
        original_is_dir, original_rglob = Path.is_dir, Path.rglob
        original_sha = exporter.sha256_file

        def is_dir(path):
            if path in self.models:
                raise AssertionError('source-only export inspected model directory')
            return original_is_dir(path)

        def rglob(path, *args, **kwargs):
            if path in self.models:
                raise AssertionError('source-only export enumerated model files')
            return original_rglob(path, *args, **kwargs)

        def archive_sha(path):
            self.assertEqual(self.output / 'runtime.tar.gz', path)
            return original_sha(path)

        with patch.object(Path, 'is_dir', is_dir), patch.object(Path, 'rglob', rglob), \
                patch.object(exporter, 'sha256_file', side_effect=archive_sha):
            manifest = self.export()
        self.assertFalse(manifest['assets_complete'])
        self.assertFalse(any(asset['included'] for asset in manifest['assets'].values()))
        self.assertFalse((self.output / 'assets.tar').exists())
        self.assertNotIn('imports', manifest)
        self.assertNotIn('files', manifest)
        with tarfile.open(self.output / 'runtime.tar.gz') as archive:
            link = archive.getmember('LMOps/gad/extensions/core')
            self.assertTrue(link.issym())
            self.assertEqual('../../rubric/core', link.linkname)

    def test_explicit_assets_are_packed_and_only_archives_are_hashed(self):
        self.make_assets()
        original_sha = exporter.sha256_file

        def archive_sha(path):
            self.assertIn(path, (self.output / 'runtime.tar.gz', self.output / 'assets.tar'))
            return original_sha(path)

        with patch.object(exporter, 'sha256_file', side_effect=archive_sha):
            manifest = self.export(include_assets=True)
        self.assertTrue(manifest['assets_complete'])
        with tarfile.open(self.output / 'assets.tar') as archive:
            for name in ('assets/models/student/model.safetensors',
                         'assets/initialization/critic/model.safetensors',
                         'assets/models/judge/model.safetensors'):
                self.assertEqual(b'tiny fixture weights', archive.extractfile('LMOps/' + name).read())
            self.assertEqual(b'tiny fixture data', archive.extractfile('LMOps/data/math/train.parquet').read())

    def test_missing_models_are_allowed_only_when_not_requested(self):
        self.export()
        self.output = self.root.parent / 'with-assets'
        with self.assertRaisesRegex(ValueError, 'required method asset missing'):
            self.export(include_assets=True)

    def test_requested_missing_data_is_reported(self):
        self.make_assets()
        self.data_paths[0].unlink()
        with self.assertRaisesRegex(ValueError, 'required training data missing'):
            self.export(include_assets=True)


if __name__ == '__main__':
    unittest.main()
