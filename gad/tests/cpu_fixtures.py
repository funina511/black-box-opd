"""Explicit asset identities for CPU logic tests, not loadable model weights.

Real model/tokenizer compatibility belongs to GAD_TEST_QWEN_TOKENIZER tests.
"""
from pathlib import Path
from config import compose_config as compose_real_config
ROOT=Path(__file__).resolve().parents[2]

def compose_cpu_config(*args,**kwargs):
    cfg=compose_real_config(*args,**kwargs)
    cfg.data['models.student']=str(ROOT/'gad/tests/fixtures/identity_model')
    cfg.data['rubric.judge.tokenizer_path']=str(ROOT/'gad/tests/fixtures/identity_model')
    cfg.data['validation.quick_file']=str(ROOT/'gad/tests/fixtures/quick.parquet')
    cfg.data['data.train_file']=str(ROOT/'gad/tests/fixtures/quick.parquet')
    cfg.data['data.val_file']=str(ROOT/'gad/tests/fixtures/quick.parquet')
    if not cfg.get('models.discriminator_init'):cfg.data['models.discriminator_init']=str(ROOT/'gad/tests/fixtures/identity_model')
    return cfg
