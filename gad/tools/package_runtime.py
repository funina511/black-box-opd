#!/usr/bin/env python3
"""Export the self-contained source/dependencies and, optionally, method assets.

Never export generated repositories, backend worktrees or historical runs.
"""
import argparse
import ast
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
    directories=('scripts','configs','gad/scripts/pipeline','gad/configs','gad/tools','gad/extensions','gad/tests','rubric/core','rubric/specs',
        'rubric/prompts','rubric/tests','rubric/configs','sample','scoring','evaluate','third_party','environments','docs')
    files={p for name in directories for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts
           and p.suffix not in ('.pyc','.log') and not p.name.endswith('.local.yaml')}
    files.update(ROOT/p for p in ('train.yaml','README.md','.gitignore','gad/.gitignore','gad/PIPELINE.md','gad/ONLINE_RUBRIC.md','assets/README.md','assets/manifest.json'))
    files.update(p for p in (ROOT/'assets/datasets/math_eval').rglob('*') if p.is_file())
    # Keep the canonical core as a relative in-directory link in Git and archives.
    files.add(ROOT/'gad/extensions/core')
    files.update((ROOT/'rubric').glob('*.py'))
    source_hashes={};imports={}
    with tarfile.open(output/'runtime.tar.gz','w:gz',dereference=False) as tar:
        for p in sorted(files):
            if not p.exists():raise ValueError('required source missing: '+str(p))
            name='LMOps/'+str(p.relative_to(ROOT))
            if p.is_symlink():
                if not p.resolve().is_relative_to(ROOT):raise ValueError('external source symlink: '+str(p))
                source_hashes[name]={'symlink':str(p.readlink())}
            else:source_hashes[name]=sha256_file(p)
            tar.add(p,arcname=name,recursive=False)
            if p.suffix=='.py':
                tree=ast.parse(p.read_text());imports[name]=sorted({n.module or '.' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names})
    keys=['models.student','models.discriminator_init']
    if cfg.get('rubric.enabled') and cfg.get('rubric.mode')=='joint':keys.append('rubric.judge.tokenizer_path')
    assets={};asset_files={};missing=[]
    for key in keys:
        directory=Path(cfg.require(key))
        destination=Path('assets/initialization' if key=='models.discriminator_init' else 'assets/models')/directory.name
        if not directory.is_dir():
            if include_assets:raise ValueError('required method asset missing: '+str(directory))
            missing.append(key)
            assets[key]={'source':str(directory),'destination':str(destination),'present':False}
            continue
        items={}
        for p in sorted(directory.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts:continue
            target=destination/p.relative_to(directory);asset_files[str(target)]=p;items[str(p.relative_to(directory))]=sha256_file(p)
        if not any(p.endswith(('.safetensors','.bin')) for p in items):raise ValueError('model weights missing: '+str(directory))
        assets[key]={'source':str(directory),'destination':str(destination),'files':items}
    data_roots={Path(cfg.require(k)).parent for k in ('data.train_file','data.val_file','data.teacher_index','data.manifest','validation.quick_file')}
    for directory in data_roots:
        if not directory.is_relative_to(ROOT/'data'):raise ValueError('copy external training data into data/ before release: '+str(directory))
        if not directory.is_dir():
            if include_assets:raise ValueError('required training data missing: '+str(directory))
            missing.append(str(directory.relative_to(ROOT)));continue
        for p in directory.iterdir():
            if p.is_file():asset_files[str(p.relative_to(ROOT))]=p
    hashes={name:sha256_file(p) for name,p in sorted(asset_files.items())}
    if include_assets:
        with tarfile.open(output/'assets.tar','w',dereference=True) as tar:
            for name,p in sorted(asset_files.items()):tar.add(p,arcname='LMOps/'+name,recursive=False)
    status=subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain','--untracked-files=all'],text=True) if (ROOT/'.git').exists() else 'source archive without Git metadata'
    manifest={'schema_version':'gad.runtime_package.v2','runtime_sha256':sha256_file(output/'runtime.tar.gz'),
        'assets_archive_sha256':sha256_file(output/'assets.tar') if include_assets else None,
        'files':source_hashes,'asset_files':hashes,'assets':assets,'missing_assets':missing,'assets_complete':bool(include_assets and not missing),'imports':imports,'source_working_tree_status':status,'source_commit':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip() if (ROOT/'.git').exists() else None,
        'source_bundle':json.loads((ROOT/'third_party/verl/SOURCE.json').read_text()),
        'method':{'rubric_enabled':cfg.get('rubric.enabled'),'mode':cfg.get('rubric.mode')},
        'scope':'self-contained LMOps root after source+selected assets extraction; original Python environment reused for validation; legacy cross-path resume and resharding unsupported'}
    write_json(output/'manifest.json',manifest)
    (output/'REBUILD.md').write_text('''# 准备发布目录

1. 解包runtime.tar.gz（仅一个LMOps根目录）。如有assets.tar解包到同一父目录；否则按manifest的destination复制对应资产并逐文件核对SHA256。内容相同的已存在文件无需再复制。
2. 按LMOps/environments/README.md准备环境并设置GAD_PYTHON。资产完整不代表CUDA环境已验证。
3. 在LMOps执行 scripts/preflight.sh --model qwen3_0.6b --dataset dapo_math_17k_v2 --method baseline --machine 3090_4gpu --prepare-backends 。GPU规格不同先填服务器模板。
4. 源仓库缺失时prepare会从third_party/verl中的具名bundle及shallow模板重建三个固定提交；已有非空平铺目录不会被自动覆盖。默认不联网。
5. train dry-run检查三阶段，evaluate dry-run检查所选模型和四任务。正式训练由用户启动；baseline/correctness_only不需要Judge。

权重清单之外的Teacher、其它模型、历史runs及backends不包含。旧run异路径续训、任意拓扑重分片未支持。包内仅保留被正式入口/CPU回归引用的历史工具；未证明无用的源文件不自动删除。
''')
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evidence-dir',type=Path,help='Reserved for audit callers; historical evidence is delivered separately')
    p.add_argument('--include-assets',action='store_true',help='Also create selected method model/data assets archive')
    a=p.parse_args();package(a.config,a.output,a.evidence_dir,a.include_assets);print(a.output/'manifest.json')
