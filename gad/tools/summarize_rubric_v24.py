#!/usr/bin/env python3
"""Prepare or summarize per-dimension audits without treating models as humans.

--samples writes a review packet; --audit summarizes reviewed packets. Cohorts
and real/boundary groups are separate. No judge requests are made.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

DIMS=('verified_progress','unresolved_error_severity')


def packet(samples,cohort):
    result=[]
    for sample in samples:
        response=sample.get('response','');r=sample.get('result',{})
        parsed=r.get('parsed',{});raw=parsed.get('_raw_labels',parsed.get('_record',{}))
        if not raw:
            try:raw=json.loads(r.get('raw_output') or '{}')
            except (ValueError,TypeError):raw={}
            if not isinstance(raw,dict):raw={}
        for dim in DIMS:
            result.append({'id':sample['id'],'dimension':dim,'cohort':cohort,'kind':sample.get('kind','unknown'),'uid':sample.get('uid'),'source_role':sample.get('source_role'),'truncated':sample.get('truncated'),'answer_correct':sample.get('scoring',{}).get('correct'),'problem':sample.get('prompt'),'visible_response':response,'raw_score':raw.get(dim,{}).get('value'),'final_score':parsed.get(dim,{}).get('value'),'mask_reason':parsed.get('_mask_reasons',{}).get(dim,[]),'raw_events':parsed.get('_initial_record',parsed.get('_record',{})).get('events'),'final_events':parsed.get('_record',{}).get('events'),'source_evidence':parsed.get('_source_evidence'),'semantic_model_review':parsed.get('_semantic_review'),'technical_error':r.get('error'),'score_valid':None,'evidence_valid':None,'raw_score_valid':None,'raw_evidence_valid':None,'critical_false_acceptance':None,'error_type':[],'reviewer':'','review_source':'human','review_status':'pending'})
    return result


def summary(rows):
    groups=defaultdict(list)
    seen=set()
    for row in rows:
        identity=(row.get('cohort','unspecified'),row['id'],row['dimension'])
        if identity in seen:raise ValueError('duplicate dimension audit')
        seen.add(identity)
        if row.get('review_status')=='completed' and row.get('review_source')=='human':
            if not row.get('reviewer'):raise ValueError('completed human reviews require a named reviewer')
            for field in ('score_valid','evidence_valid','raw_score_valid','raw_evidence_valid','critical_false_acceptance'):
                if row.get(field) is not None and type(row[field]) is not bool:raise ValueError('review decisions must be boolean or null')
        groups[(row.get('cohort','unspecified'),row.get('kind','unknown'),row['dimension'])].append(row)
    result=[]
    for (cohort,kind,dim),items in sorted(groups.items()):
        kept=[r for r in items if r['final_score'] is not None]
        human=[r for r in items if r.get('review_status')=='completed' and r.get('review_source')=='human']
        judged_kept=[r for r in human if r['final_score'] is not None and type(r.get('score_valid')) is bool]
        evidence=[r for r in human if r['final_score'] is not None and type(r.get('evidence_valid')) is bool]
        reliable=[r for r in human if r['final_score'] is not None and r.get('score_valid') is True and r.get('evidence_valid') is True]
        result.append({'cohort':cohort,'kind':kind,'dimension':dim,'planned_dimensions':len(items),'kept_dimensions':len(kept),'coverage':len(kept)/len(items),'unique_responses':len({r['id'] for r in items}),'independent_questions_known':len({r['uid'] for r in items if r.get('uid')}),'human_reviewed_dimensions':len(human),'model_assisted_dimensions':sum(r.get('review_source') in ('model_assisted','targeted_model') for r in items),'reviewed_kept_score_correct':{'numerator':sum(r['score_valid'] for r in judged_kept),'denominator':len(judged_kept)},'reviewed_kept_evidence_supported':{'numerator':sum(r['evidence_valid'] for r in evidence),'denominator':len(evidence)},'human_reliable_retained_dimensions':len(reliable),'critical_false_acceptance':sum(r.get('critical_false_acceptance') is True for r in human),'correct_raw_label_wrongly_masked':sum(r['final_score'] is None and r.get('raw_score_valid') is True and r.get('raw_evidence_valid') is True for r in human),'accidentally_right_score_unsupported_evidence':sum(r['final_score'] is not None and r.get('score_valid') is True and r.get('evidence_valid') is False for r in human),'reliability':'pending' if not human else 'partial audit; inspect denominators, not deployment acceptance'})
    # Same problem, same rule-correctness; count distinct problems separately.
    by_response=defaultdict(dict)
    for r in rows:
        if r.get('uid') and r.get('answer_correct') in (True,False) and type(r.get('answer_correct')) is bool:
            by_response[(r.get('cohort','unspecified'),r.get('kind','unknown'),r['uid'],r['answer_correct'])].setdefault(r['id'],{})[r['dimension']]=r
    pairs=[]
    for (cohort,kind,uid,corr),responses in by_response.items():
        ids=sorted(responses)
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                comparable=[];different=[];reliable=[]
                for d in DIMS:
                    x,y=responses[a].get(d,{}),responses[b].get(d,{})
                    if x.get('final_score') is not None and y.get('final_score') is not None:
                        comparable.append(d)
                        if x['final_score']!=y['final_score']:different.append(d)
                        if all(r.get('review_source')=='human' and r.get('review_status')=='completed' and r.get('score_valid') is True and r.get('evidence_valid') is True for r in (x,y)):reliable.append(d)
                pairs.append({'cohort':cohort,'kind':kind,'uid':uid,'answer_correct':corr,'response_ids':[a,b],'comparable_dimensions':comparable,'different_dimensions':different,'human_reliable_dimensions':reliable})
    return {'groups':result,'same_problem_same_correctness_pairs':pairs,'discrimination_status':'descriptive paired differences; no human discrimination acceptance without reliable per-pair reviews','acceptance':'pending user thresholds and human review; no pooled development/boundary/holdout accuracy'}


def main():
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--samples',type=Path);g.add_argument('--audit',type=Path);ap.add_argument('--cohort',default='unspecified');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise ValueError('use a new output path to preserve reviewed evidence')
    rows=[json.loads(l) for l in (a.samples or a.audit).read_text().splitlines() if l.strip()]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.samples:a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in packet(rows,a.cohort)))
    else:a.output.write_text(json.dumps(summary(rows),ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
