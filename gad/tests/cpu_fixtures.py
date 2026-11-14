"""Explicit asset identities for CPU logic tests, not loadable model weights.

Real model/tokenizer compatibility belongs to GAD_TEST_QWEN_TOKENIZER tests.
"""
from pathlib import Path
from config import compose_config as compose_real_config, load_config as load_real_config
ROOT=Path(__file__).resolve().parents[2]

def _cpu_assets(cfg):
    cfg.data['models.student']=str(ROOT/'gad/tests/fixtures/identity_model')
    cfg.data['rubric.judge.tokenizer_path']=str(ROOT/'gad/tests/fixtures/identity_model')
    cfg.data['validation.quick_file']=str(ROOT/'gad/tests/fixtures/quick.parquet')
    cfg.data['data.train_file']=str(ROOT/'gad/tests/fixtures/quick.parquet')
    cfg.data['data.val_file']=str(ROOT/'gad/tests/fixtures/quick.parquet')
    if not cfg.get('models.discriminator_init'):cfg.data['models.discriminator_init']=str(ROOT/'gad/tests/fixtures/identity_model')
    return cfg

def compose_cpu_config(*args,**kwargs):
    return _cpu_assets(compose_real_config(*args,**kwargs))

def load_cpu_config(path):
    return _cpu_assets(load_real_config(path))
