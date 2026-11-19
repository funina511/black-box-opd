"""The trainer and fresh Python workers must enable diagnostics before imports."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gad/tools')]
from config import Config
from pipeline import stage_env


class StageFaulthandlerTests(unittest.TestCase):
    def test_stage_environment_enables_faulthandler_at_interpreter_start(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            cfg = Config({}, path=folder / 'config.yaml')
            for stage in ('seqkd', 'warmup', 'gad'):
                with self.subTest(stage=stage):
                    env = stage_env(cfg, folder / 'run', stage, folder / stage,
                                    log_backend='console', install_guard=False)
                    self.assertEqual('1', env['PYTHONFAULTHANDLER'])
                    result = subprocess.run(
                        [sys.executable, '-c',
                         'import faulthandler,json;print(json.dumps({"enabled":faulthandler.is_enabled()}))'],
                        env={**os.environ, **env, 'CUDA_VISIBLE_DEVICES': ''},
                        capture_output=True, text=True, timeout=10, check=True)
                    self.assertTrue(json.loads(result.stdout)['enabled'])


if __name__ == '__main__':
    unittest.main()
