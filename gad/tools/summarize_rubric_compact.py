#!/usr/bin/env python3
"""Audit Appendix A's fixed four responses without generating or relabeling data."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from rubric.core.process_v2 import source_spans
DIMS=('verified_progress','unresolved_error_severity')


def read_rows(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def evidence_metrics(parsed,response):
    row=parsed.get('_record',{});spans=source_spans(response);lookup={s['id']:i for i,s in enumerate(spans)}
    events={};all_ids=[]
    for eid,event in row.get('events',{}).items():
        evidence=parsed.get('_source_evidence',{}).get(eid,{})
        ids=evidence.get('supporting_span_ids',event.get('supporting_span_ids',[]))
        unique=sorted(set(ids),key=lambda x:lookup.get(x,len(lookup)))
        units=event.get('supporting_span_ranges')
        ratio=len(unique)/len(spans) if spans else 0
        events[eid]={'units':len(units) if units is not None else len(ids),'wire_kind':'ranges' if units is not None else 'ids',
                     'expanded_unique_spans':len(unique),'coverage_ratio':ratio,'broad_reference':ratio>=.8,
                     'expanded_ids':unique,'compact_ranges':units}
        all_ids.extend(unique)
    return {'events':events,'event_units':sum(e['units'] for e in events.values()),
            'expanded_union_count':len(set(all_ids)),'cross_event_duplicate_spans':len(all_ids)-len(set(all_ids)),
            'source_span_count':len(spans),'union_coverage_ratio':len(set(all_ids))/len(spans) if spans else 0}


def per_row(row,arm):
    r=row['result'];parsed=r.get('parsed',{});initial=r.get('initial_parsed',parsed)
    review=r.get('targeted_result',{});review_wire=review.get('parsed',{})
    requests=[]
    for stage,result in [('primary',r),('review',review)]:
        if not result:continue
        requests.append({'stage':stage,'attempts':result.get('attempts',0)-(review.get('attempts',0) if stage=='primary' else 0),
                         'provider_attempts':[{'seconds':a.get('seconds'),'usage':a.get('response',{}).get('usage'),
                                              'transport_error':a.get('transport_error')} for a in result.get('api_attempts',[])],
                         'output_lengths':result.get('output_lengths'), 'queue_and_wait':result.get('scheduler_timing'),
                         'error':result.get('error')})
    primary_ids={sid for e in initial.get('_source_evidence',{}).values() for sid in e.get('supporting_span_ids',[])}
    label_units=0;label_duplicate_ids=[]
    review_evidence=parsed.get('_review_source_evidence',{})
    reviewed_event_ids={sid for key,e in review_evidence.items() if key.startswith('event:') for sid in e.get('supporting_span_ids',[])}
    expanded_label_duplicates=sum(len(set(e.get('supporting_span_ids',[])) & reviewed_event_ids) for key,e in review_evidence.items() if key.startswith('label:'))
    for decision in review_wire.get('labels',{}).values():
        units=decision.get('supporting_span_ranges',decision.get('supporting_span_ids',[]));label_units+=len(units)
        if 'supporting_span_ids' in decision:label_duplicate_ids.extend(s for s in decision['supporting_span_ids'] if s in primary_ids)
    return {'arm':arm,'id':row['id'],'uid':row['uid'],'source_role':row['source_role'],'truncated':row['truncated'],
            'response_sha256':hashlib.sha256(row['response'].encode()).hexdigest(),'human_review':'pending',
            'initial':{d:initial.get('_record',{}).get(d) for d in DIMS},
            'final':{d:parsed.get(d) for d in DIMS},'mask_reasons':parsed.get('_mask_reasons'),
            'semantic_review':parsed.get('_semantic_review'),'requests':requests,'usage':r.get('usage',{}),
            'primary_evidence':evidence_metrics(initial,row['response']),
            'final_evidence':evidence_metrics(parsed,row['response']),
            'label_extra_evidence_units':label_units,'legacy_label_event_duplicate_ids':len(label_duplicate_ids),'review_label_event_duplicate_spans':expanded_label_duplicates,
            'review_expanded_evidence':parsed.get('_review_source_evidence'),
            'capacity_conflicts':parsed.get('_evidence_audit',{}).get('capacity_conflicts',[]),
            'error':r.get('error'),'pending':r.get('pending_scheduler_state'),
            'targeted_error':review.get('error'),'scheduler_stages':r.get('scheduler_stages'),
            'total_attempts':r.get('attempts',0),'targeted_attempts':review.get('attempts',0),
            'completed_chain':r.get('attempts',0)>0 and not r.get('pending_scheduler_state') and not r.get('error') and not review.get('error')}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--directory',required=True);args=parser.parse_args()
    out=Path(args.directory);base=read_rows(out/'fixed4.jsonl');rows=[];arm_summaries={}
    for arm in ('B0','B1','B2'):
        path=out/'gpu'/arm/'samples.jsonl'
        source=base if arm=='B0' else read_rows(path) if path.exists() else [dict(r,result={'error':'not_run','attempts':0}) for r in base]
        group=[per_row(r,arm) for r in source];rows.extend(group)
        summary_path=out/'gpu'/arm/'summary.json'
        actual=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        stages=actual.get('scheduler',{}).get('stages',[])
        waits=[s['api_wait_seconds'] for s in stages]
        arm_summaries[arm]={'planned':4,'attempted':sum(r['total_attempts']>0 for r in group),
            'completed_ids':[r['id'] for r in group if r['completed_chain']],
            'actual_calls':sum(r['total_attempts'] for r in group),'review_calls':sum(r['targeted_attempts'] for r in group),
            'actual_review_trigger_rate':sum(bool(r['semantic_review']) and r['semantic_review'].get('review_source')=='targeted_model' for r in group)/4,
            'batch_wall_seconds':actual.get('batch_wall_seconds'),
            'responses_per_second':actual.get('responses_per_second'),'output_tokens_per_second':actual.get('output_tokens_per_second'),
            'observed_api_peak':actual.get('actual_api_peak'),
            'api_seconds_median':statistics.median(waits) if waits else None,'api_seconds_max':max(waits) if waits else None,
            'usage':{k:sum(r['usage'].get(k,0) for r in group) for k in ('prompt_tokens','completion_tokens','total_tokens')},
            'primary_evidence_units':sum(r['primary_evidence']['event_units'] for r in group),
            'label_extra_evidence_units':sum(r['label_extra_evidence_units'] for r in group),
            'final_valid':{role:{d:sum(r['final'][d] is not None and r['final'][d]['valid'] for r in group if role=='all' or r['source_role']==role) for d in DIMS} for role in ('all','teacher','student')},
            'errors':dict(Counter(r['error'] or r['targeted_error'] for r in group if r['error'] or r['targeted_error'])),
            'thinking_budget_exhausted':sum(r['error']=='judge_reasoning_budget_exhausted' or r['targeted_error']=='judge_reasoning_budget_exhausted' for r in group),
            'human_process_accuracy':None,'human_review':'pending'}
    shared=set(arm_summaries['B1']['completed_ids']) & set(arm_summaries['B2']['completed_ids'])
    summary={'arms':arm_summaries,'same_completed_cohort':sorted(shared),'planned_ids':[r['id'] for r in base],
             'historical_B0_timing':'Different startup/window; fixed subset batch wall was not measured; no strict speedup claim.',
             'quality_status':'pending_human_review','reliable_labels_per_minute':None,
             'default_concurrency':1,'protocol':'rubric.events.v2.6','training_started':False}
    if len(shared)==4:
        b1,b2=arm_summaries['B1'],arm_summaries['B2']
        summary['B2_over_B1_wall_speedup']=b1['batch_wall_seconds']/b2['batch_wall_seconds']
        summary['workload_warning']='Actual outputs/lengths/review triggers may differ. Wall ratio does not isolate pure scheduling.'
    (out/'comparison.rows.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    (out/'comparison.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    blind=[];key=[]
    for i,row in enumerate(sorted(base,key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()),1):
        blind.append({'blind_id':f'A{i}','problem':row['prompt'],'visible_response_spans':source_spans(row['response']),
                      'review_fields':['progress_score_correct','error_score_correct','source_entailment','self_attribution','fake_repair','critical_error_missed'],
                      'reviewer':None,'status':'pending'})
        key.append({'blind_id':f'A{i}','id':row['id'],'role':row['source_role']})
    (out/'human_review_blind.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in blind))
    (out/'human_review_key.json').write_text(json.dumps(key,ensure_ascii=False,indent=2))
    lines=['# 附录 A 固定四条 Judge 微基准','', '| 组 | 首轮+复核调用 | 整批秒 | API输出token | 证据单元 | 最终Progress/Error | Student Progress/Error |', '|---|---:|---:|---:|---:|---|---|']
    for arm,s in arm_summaries.items():
        v=s['final_valid'];lines.append(f"| {arm} | {s['actual_calls']-s['review_calls']}+{s['review_calls']} | {s['batch_wall_seconds']} | {s['usage']['completion_tokens']} | {s['primary_evidence_units']} | {v['all'][DIMS[0]]}/{v['all'][DIMS[1]]} | {v['student'][DIMS[0]]}/{v['student'][DIMS[1]]} |")
    lines+=['','B0 为历史时段参考，未测得该四条子集的整批墙钟，不能计算严格同窗口加速比。',
            'B1/B2 共用一次服务和总预算，缓存及复核去重状态按组隔离；逐条调用和未完成项均保留。',
            '新版本仍检查完整可见轨迹；Thinking预算、数学标准、复核触发函数及训练损失未更改。',
            '人工审查尚未完成，模型复核一致不等于正确；不计算可靠标签/分钟，不据此启动正式训练。',
            '', '逐条结果见 `comparison.rows.jsonl`；盲审原文见 `human_review_blind.jsonl`，匹配键单独保存。']
    if 'B2_over_B1_wall_speedup' in summary:lines+=['',f"B2/B1 整批墙钟比值对应 {summary['B2_over_B1_wall_speedup']:.3f} 倍；输出和复核工作量变化须另外检查。"]
    (out/'benchmark_report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
