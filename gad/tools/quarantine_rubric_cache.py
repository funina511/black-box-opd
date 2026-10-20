#!/usr/bin/env python3
"""Copy a cache into a NEW namespace excluding human-confirmed bad entries.

The source and active experiments are never mutated. Review rows require a
cache_key, review_source=human, review_status=failed and a reason.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from rubric.core.contracts import file_hash


def quarantine(source, destination, reviews):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() or source==destination or source in destination.parents:raise ValueError('destination must be a separate new namespace')
    failed={}
    for row in reviews:
        if row.get('review_source')!='human' or row.get('review_status')!='failed' or not row.get('reason'):raise ValueError('only explicit human-confirmed failures may be quarantined')
        key=row.get('cache_key','')
        if not key or Path(key).name!=key or any(c not in '0123456789abcdef' for c in key):raise ValueError('invalid cache key')
        if not (source/(key+'.json')).is_file():raise ValueError('reviewed cache entry missing')
        failed[key]=row
    if not failed:raise ValueError('no confirmed failures supplied')
    destination.mkdir(parents=True)
    isolated=destination/'quarantine';isolated.mkdir()
    records=[]
    for path in sorted(source.glob('*.json')):
        target=(isolated if path.stem in failed else destination)/path.name
        shutil.copy2(path,target)
        if path.stem in failed:records.append({'key':path.stem,'sha256':file_hash(path),'review':failed[path.stem]})
    manifest={'source':str(source),'destination':str(destination),'quarantined':records,'source_modified':False,'activation':'not activated; requires explicit new-run cache selection or migration, never replace a running directory'}
    (isolated/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--reviews',required=True);a=p.parse_args()
    print(json.dumps(quarantine(a.source,a.destination,[json.loads(l) for l in Path(a.reviews).read_text().splitlines() if l.strip()]),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
