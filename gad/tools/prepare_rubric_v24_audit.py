#!/usr/bin/env python3
"""Freeze unseen training questions and prepare pending human review before tuning."""
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
import random
import re
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rubric.core.contracts import digest, file_hash
from scoring.math import score, identity


def normalized(prompt):
    return re.sub(r'\W+', '', str(prompt).casefold())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=ROOT/'data/rubric_math_v1/math_v1_512q_seed42_2631677456f2/trajectories.jsonl')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--questions', type=int, default=12)
    ap.add_argument('--seed', type=int, default=20261016)
    a = ap.parse_args()
    if a.output.exists():raise ValueError('use a new directory; frozen review materials must not be overwritten')
    prior = sorted({*ROOT.glob('reports/rubric_v2/**/inputs.jsonl'), *ROOT.glob('reports/rubric_v2/**/selected24.jsonl'), ROOT/'reports/rubric_v2/judge_samples_guarded.jsonl'})
    seen_ids, seen_text, seen_hashes, provenance = set(), set(), set(), {}
    for p in prior:
        if not p.exists():continue
        provenance[str(p.relative_to(ROOT))] = file_hash(p)
        for line in p.read_text().splitlines():
            r = json.loads(line)
            for k in ('uid', 'problem_id', 'prompt_id'):
                if r.get(k):seen_ids.add(str(r[k]))
            for k in ('prompt_hash', 'normalized_prompt_hash'):
                if r.get(k):seen_hashes.add(r[k])
            if r.get('prompt'):seen_text.add(normalized(r['prompt']))
    # Character shingles also exclude near identical wording; the fixed source
    # and cutoff are recorded, not a proof that all semantic duplicates vanish.
    def shingles(t):return {t[i:i+12] for i in range(max(1, len(t)-11))}
    prior_shingles = [shingles(t) for t in seen_text if t]
    groups = defaultdict(list)
    excluded = Counter()
    for line in a.source.read_text().splitlines():
        r = json.loads(line)
        if r['split'] != 'train' or not r['input_eligible']:continue
        n = normalized(r['prompt'])
        reason = None
        if str(r['uid']) in seen_ids or r['prompt_hash'] in seen_hashes or r['normalized_prompt_hash'] in seen_hashes or n in seen_text:reason = 'id_or_normalized_text'
        if not reason:
            s = shingles(n)
            if any(len(s & t)/max(1, len(s | t)) >= .9 for t in prior_shingles):reason = 'near_duplicate'
        if reason:excluded[reason] += 1;continue
        groups[r['uid']].append(r)
    pool = [g for _,g in sorted(groups.items()) if any(r['source_role']=='teacher' for r in g) and any(r['source_role']=='student' for r in g)]
    rng = random.Random(a.seed)
    selected = rng.sample(pool, a.questions)
    rows, blind = [], []
    for g in selected:
        for role in ('teacher', 'student'):
            r = rng.choice(sorted([r for r in g if r['source_role']==role], key=lambda r:r['response_id']))
            row = {k:r[k] for k in ('uid','prompt','prompt_hash','normalized_prompt_hash','response','response_hash','response_id','ground_truth','source_role','truncated','finish_reason','source_fingerprint')}
            row.update(id='holdout_'+r['response_id'][:20], kind='real', source_split='train', scoring=score(r['response'], r['ground_truth']), semantic_review_status='pending')
            rows.append(row)
            blind.append({'review_id': f'h{len(blind)+1:03}', 'problem':r['prompt'], 'visible_response':r['response'], 'reference_final_answer':r['ground_truth'], 'response_truncated':r['truncated'], 'reviewer':'', 'review_source':'human', 'review_status':'pending', 'verified_progress':{'value':None,'valid':None,'score_valid':None,'evidence_valid':None,'reason':''}, 'unresolved_error_severity':{'value':None,'valid':None,'score_valid':None,'evidence_valid':None,'reason':''}})
    old = ROOT/'reports/rubric_v2/simplified_8b_20261015/analysis.json'
    previous = json.loads(old.read_text())
    audit = []
    notes = {
      'real_900e7927fa39787bac81': 'evidence_attribution; invented repair and full progress; original final answer 721, not 77',
      'real_0440d331986484988ada': 'invented repair; invalid correction link; critical error understated',
      'boundary_false_repair_0': 'severity_understated; actual replacement absent',
      'boundary_partial_formula_0': 'independent prefix survives incomplete formula; error zero needs human confirmation',
      'boundary_no_math_0': 'unsupported zero error masked; progress zero plausible',
      'frozen12_illegal_division': 'unsupported positive progress masked; severe error retained',
      'boundary_pending_branch_0': 'legal unfinished branch mistaken for error; positive error masked',
      'boundary_real_repair_0': 'actual replacement and substitution visible; scores plausible'}
    for d in previous['details']:
        if not d.get('raw_record'):continue
        for dim in ('verified_progress','unresolved_error_severity'):
            audit.append({'id':d['id'], 'dimension':dim, 'raw_score':d['raw'][dim], 'final_score':d['effective'][dim], 'mask_reason':d.get('mask_reasons',{}).get(dim,[]), 'score_valid':'pending', 'evidence_valid':'pending', 'error_type':notes.get(d['id'],''), 'review_source':'model_assisted', 'review_status':'pending_human_confirmation', 'source':str(old.relative_to(ROOT)), 'source_sha256':file_hash(old)})
    a.output.mkdir(parents=True)
    jsonl(a.output/'holdout.jsonl', rows)
    jsonl(a.output/'human_review_blind.jsonl',blind)
    write(a.output/'human_review_key.json',{b['review_id']:r['id'] for b,r in zip(blind,rows)})
    jsonl(a.output/'development_audit.jsonl',audit)
    write(a.output/'acceptance.json', {'status':'pending_user_confirmation', 'allowed_mislabels':{'verified_progress':None,'unresolved_error_severity':None},'critical_error_definition':None,'human_sample_size':None,'judge_calls_limit':24,'gpu_seconds_limit':1800,'judge_model':'Qwen3-8B','thinking':True,'gpu_execution':'paused','human_review':'pending'})
    write(a.output/'freeze_manifest.json', {'version':'research.v2.4.freeze.v1','seed':a.seed,'questions':a.questions,'responses':len(rows),'source':str(a.source),'source_sha256':file_hash(a.source),'excluded_inputs':provenance,'excluded_responses':dict(excluded),'available_questions':len(pool),'selection':'uniform questions then one random existing teacher/student response, train split only; no actor sampling','near_duplicate_rule':'normalized alphanumeric text and 12-character shingle Jaccard >=0.9; semantic duplicates still require review','strata':dict(Counter(f"{r['source_role']}/truncated={r['truncated']}/correct={r['scoring']['correct']}" for r in rows)),'difficulty':'not human rated; length is not mathematical difficulty','holdout_usage':'unseen frozen holdout; demote to development if used for rule tuning','development_set':'old frozen24 and previous128; no independent acceptance claim','scorer':identity(),'files':{p.name:file_hash(p) for p in a.output.iterdir() if p.is_file()},'plan_sha256':file_hash(ROOT/'docs/Rubric监督v2.4修改方案.md')})
    print(a.output)

if __name__=='__main__':main()
