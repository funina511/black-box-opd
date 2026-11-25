"""Build current patches in isolated worktrees, never update an experiment backend."""
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]

def prepare(directory, profile='original', stages='seqkd,warmup,gad'):
    subprocess.run([sys.executable,str(ROOT/'gad/tools/prepare_backends.py'),
        '--repo',str(ROOT/'gad/verl'),'--backend-root',str(directory),
        '--profile',profile,'--stages',stages],check=True,cwd=ROOT)

def cleanup(directory):
    directory=Path(directory)
    for stage in ('seqkd','warmup','gad'):
        path=directory/stage
        if (path/'.git').exists():
            subprocess.run(['git','-C',str(ROOT/'gad/verl'),'worktree','remove','--force',str(path)],
                           capture_output=True,check=True)

if __name__=='__main__':
    root=Path(sys.argv[1])
    if len(sys.argv)>2 and sys.argv[2]=='cleanup':
        cleanup(root/'original');cleanup(root/'extensions')
    else:
        prepare(root/'original')
        prepare(root/'extensions','gad_extensions_v1','seqkd,warmup,gad')
