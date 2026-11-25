#!/usr/bin/env python3
"""Export the self-contained source/dependencies and, optionally, method assets.

Never export generated repositories, backend worktrees or historical runs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import sha256_file,write_json
from config import load_config
ROOT=Path(__file__).resolve().parents[2]


def package(config,output,evidence_dir=None,include_assets=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=load_config(config)
    directories=('scripts','configs','gad/scripts/pipeline','gad/configs','gad/tools','gad/extensions','gad/tests','gad/deepscaler','rubric/core','rubric/specs',
        'rubric/prompts','rubric/tests','rubric/configs','sample','scoring','evaluate','third_party','environments','docs')
    files={p for name in directories for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.suffix not in ('.pyc','.log') and not p.name.endswith('.local.yaml')}
    files.update(ROOT/p for p in ('train.yaml','README.md','.gitignore','gad/.gitignore','gad/PIPELINE.md','gad/ONLINE_RUBRIC.md',
                                 'gad/LICENSE','gad/setup.py','rubric/README.md','rubric/SMOKE_RESULTS.md',
                                 'archive/README.md','assets/README.md','assets/manifest.json'))
    files.update(p for p in (ROOT/'assets/datasets/math_eval').rglob('*') if p.is_file())
    # Keep the canonical core as a relative in-directory link in Git and archives.
    files.add(ROOT/'gad/extensions/core')
    files.update((ROOT/'rubric').glob('*.py'))
    with tarfile.open(output/'runtime.tar.gz','w:gz',dereference=False) as tar:
        for p in sorted(files):
            if not p.exists():raise ValueError('required source missing: '+str(p))
            name='LMOps/'+str(p.relative_to(ROOT))
            if p.is_symlink():
                if not p.resolve().is_relative_to(ROOT):raise ValueError('external source symlink: '+str(p))
            tar.add(p,arcname=name,recursive=False)
    keys=['models.student','models.discriminator_init']
    if cfg.get('rubric.enabled') and cfg.get('rubric.mode')=='joint':keys.append('rubric.judge.tokenizer_path')
    assets={};asset_files={}
    for key in keys:
        directory=Path(cfg.require(key))
        destination=Path('assets/initialization' if key=='models.discriminator_init' else 'assets/models')/directory.name
        assets[key]={'source':str(directory),'destination':str(destination),'included':include_assets}
        # A source-only export must not inspect or hash locally installed weights.
        if not include_assets:continue
        if not directory.is_dir():
            raise ValueError('required method asset missing: '+str(directory))
        items=[]
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            target=destination/p.relative_to(directory);asset_files[str(target)]=p;items.append(p)
        if not any(p.suffix in ('.safetensors','.bin') for p in items):raise ValueError('model weights missing: '+str(directory))
    data_paths=[Path(cfg.require(k)) for k in ('data.train_file','data.val_file','data.teacher_index','data.manifest','validation.quick_file')]
    data_roots=sorted({p.parent for p in data_paths})
    if include_assets:
        for p in data_paths:
            if not p.is_relative_to(ROOT/'data'):raise ValueError('copy external training data into data/ before release: '+str(p))
            if not p.is_file():raise ValueError('required training data missing: '+str(p))
        for directory in data_roots:
            for p in directory.iterdir():
                if p.is_file():asset_files[str(p.relative_to(ROOT))]=p
        with tarfile.open(output/'assets.tar','w',dereference=True) as tar:
            for name,p in sorted(asset_files.items()):tar.add(p,arcname='LMOps/'+name,recursive=False)
    status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True) if (ROOT/'.git').exists() else 'source archive without Git metadata'
    manifest={'schema_version':'gad.runtime_package.v3','runtime_sha256':sha256_file(output/'runtime.tar.gz'),
        'assets_archive_sha256':sha256_file(output/'assets.tar') if include_assets else None,
        'assets':assets,'training_data_sources':[str(p) for p in data_roots],'assets_complete':include_assets,
        'source_working_tree_status':status,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip() if (ROOT/'.git').exists() else None,
        'source_bundle':json.loads((ROOT/'third_party/verl/SOURCE.json').read_text()),
        'method':{'rubric_enabled':cfg.get('rubric.enabled'),'mode':cfg.get('rubric.mode')},
        'scope':'source export plus optional selected assets; excluded model/data contents are not inspected; target environment still requires deployment; legacy cross-path resume and resharding unsupported'}
    write_json(output/'manifest.json',manifest)
    (output/'REBUILD.md').write_text('''# 准备发布目录

1. 代码通过Git交付时固定commit即可；runtime.tar.gz仅是另一种源码交付方式。使用归档时，传输后核对归档本身的SHA256一次，再解到项目父目录；归档自带LMOps/前缀，不要求逐文件核验。
2. 只有显式传入--include-assets才会生成assets.tar，解到同一父目录。默认不读取本地模型权重或训练数据内容，只记录其来源与目标位置；模型可在目标机重新下载，现有Teacher回答数据须另外携带。
3. 按LMOps/environments/README.md安装环境并设置GAD_PYTHON，根据目标机填写机器配置。旧launch里的Warmup来源不能直接用于从头训练；单卡Judge共卡等部署要求按迁移计划处理。
4. 补齐third_party/verl/verl.bundle后，prepare_backends会重建固定后端；不复制旧worktree或历史run。训练数据与评测数据也应按交付说明补齐。
5. 目标机运行配置预检和train/evaluate的dry-run，核对训练阶段、预算及选定评测任务；必要的小规模GPU验证通过后再正式启动。

清单记录Git版本和归档校验值，不包含逐文件哈希或Python导入审计。旧run跨路径续训及任意拓扑重分片未支持；这不影响在新机创建新实验。
''')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,help='Reserved for audit callers; historical evidence is delivered separately')
    p.add_argument('--include-assets',action='store_true',help='Also create selected method model/data assets archive')
    a=p.parse_args();package(a.config,a.output,a.evidence_dir,a.include_assets);print(a.output/'manifest.json')
