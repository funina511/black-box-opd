"""Optional, bounded model review. This module never performs network calls.

A supported verdict is model agreement, not human truth or calibrated confidence.
Only evidence locations may be changed, once, in the single targeted request.
"""
from copy import deepcopy
import json
from .contracts import V2_DIMENSIONS, digest
from .process_v2 import compact_response, source_spans, parse_events, ProtocolError

VERSION = 'rubric.targeted_review.v4'
from .evidence_ranges import RANGES, expand, EvidenceError, audit, values
DIMS = V2_DIMENSIONS[1:]
DECISION = {'type':'object', 'additionalProperties':False,
    'required':['verdict','supporting_span_ranges','refuting_span_ranges','event_ids','reason'],
    'properties':{'verdict':{'enum':['supported','refuted','unresolved','not_requested']},
      'supporting_span_ranges':RANGES,
      'refuting_span_ranges':RANGES, 'event_ids':{'type':'array','items':{'enum':['e1','e2','e3','e4']}}, 'reason':{'type':'string'}}}
EVENT_DECISION=deepcopy(DECISION)
EVENT_DECISION['required'].remove('event_ids')
EVENT_DECISION['properties'].pop('event_ids')
SCHEMA = {'type':'object','additionalProperties':False,'required':['version','labels','events'],
    'properties':{'version':{'const':VERSION},
      'labels':{'type':'object','additionalProperties':False,'required':list(DIMS),'properties':{d:DECISION for d in DIMS}},
      'events':{'type':'object','additionalProperties':False,'properties':{f'e{i}':EVENT_DECISION for i in range(1,5)}}}}
INSTRUCTION = '''你只核查指定断言的数学正确性和原文归属，不重新生成Rubric或改分。
区分trajectory_claim（原文实际说过）和judge_verification（首轮Judge自行计算）。
检查指定分数是否有充分原文支持；progress=1必须整题完成，error=0须检查全部相关可见数学。
修复须原文给出实际替代并修复关联错误，计划不是结果，不能把你的推导归给原文。
允许本次一次性在完整原文中重新定位证据；用supporting_span_ranges/refuting_span_ranges输出支持/反证闭区间对象；结构为仅允许r1至r8的对象，例如{"r1":{"start":"s1","end":"s3"}}；未用槽省略，空证据用{}，禁止数组或r9。每槽start/end必须取自本次原文的实际ID，单片段起止相同。通常1–4单元，最多8，不列举全文或跨越未引用间隙。
事件须有原文区间证据，不含event_ids字段；只有标签优先用event_ids复用本次明确核查的事件，额外证据才填区间。标签必须自行核查并明确verdict和简短理由，不能仅因事件支持而自动同意标签。
逐个核查指定事件及标签，supported/refuted/unresolved；未请求维度用not_requested。
无法核实、上下文不足或有未解决分歧用unresolved，不能为了保留标签而默认同意。
保留独立Thinking，final只输出目标schema JSON。你的结论是模型复核，不能声称人工核验。'''


def targets(parsed):
    row = parsed['_record']
    repairs = {e for e,v in row['events'].items() if v.get('type')=='corrected_error'}
    result = []
    for d in DIMS:
        raw = row[d]
        if not raw['valid']:continue
        # Also check positive partial progress: the known self-attribution
        # counterexample was accepted at partial as well as full progress.
        if repairs or (d=='verified_progress' and raw['value']>0) or (d=='unresolved_error_severity' and raw['value']==0) or parsed.get('_mask_reasons',{}).get(d):result.append(d)
    return result


def messages(problem, response, parsed):
    dims = targets(parsed)
    event_ids = set()
    for d in dims:event_ids.update(parsed['_record'][d]['event_ids'])
    # Full/global judgments and error status require the error/repair context.
    event_ids.update(e for e,v in parsed['_record']['events'].items() if v['type'] in ('explicit_error','corrected_error'))
    for _ in parsed['_record']['events']:
        for eid in list(event_ids):
            e = parsed['_record']['events'].get(eid,{})
            event_ids.update(e.get('related_event_ids', []))
            if e.get('error_event_id'):event_ids.add(e['error_event_id'])
    payload = {'review_version':VERSION,'problem':problem,'target_dimensions':dims,
        'assertions':{d:parsed['_record'][d] for d in dims},
        'events':{e:v for e,v in parsed['_record']['events'].items() if e in event_ids},
        'judge_reported_scope':parsed.get('_judge_reported_scope'),
        'visible_response_spans':json.loads(compact_response(response))}
    return [{'role':'system','content':INSTRUCTION},{'role':'user','content':json.dumps(payload,ensure_ascii=False)}]


def parse_review(raw):
    def unique(pairs):
        row={}
        for k,v in pairs:
            if k in row:raise ProtocolError('semantic_schema_error','duplicate key')
            row[k]=v
        return row
    try:r=json.loads(raw,object_pairs_hook=unique)
    except (ValueError,TypeError) as exc:raise ProtocolError('semantic_schema_error',str(exc)) from exc
    if not isinstance(r,dict) or set(r)!=set(SCHEMA['required']) or r['version']!=VERSION:raise ProtocolError('semantic_schema_error','version/root mismatch')
    if not isinstance(r['labels'],dict) or set(r['labels'])!=set(DIMS) or not isinstance(r['events'],dict) or any(e not in SCHEMA['properties']['events']['properties'] for e in r['events']):raise ProtocolError('semantic_schema_error','invalid targets')
    for kind in ('labels','events'):
        branch=DECISION if kind=='labels' else EVENT_DECISION
        for item in r[kind].values():
            if not isinstance(item,dict) or set(item)!=set(branch['required']) or item['verdict'] not in DECISION['properties']['verdict']['enum'] or not isinstance(item['reason'],str) or not item['reason'].strip():raise ProtocolError('semantic_schema_error','decision missing')
            for key in ('supporting_span_ranges','refuting_span_ranges'):
                units=item[key]
                try:
                    items=values(units)
                    if any(not isinstance(u,dict) or set(u)!={'start','end'} or any(not isinstance(v,str) for v in u.values()) for u in items):raise EvidenceError('event_spans_malformed')
                except EvidenceError as exc:raise ProtocolError(exc.code,'invalid finite evidence slots') from exc
            if kind=='labels':
                ids=item['event_ids']
                if not isinstance(ids,list) or any(e not in ('e1','e2','e3','e4') for e in ids) or len(set(ids))!=len(ids):raise ProtocolError('semantic_schema_error','invalid event refs')
    return r


def apply_review(parsed, response, review=None, failure=None):
    """A single relocation, fresh structural parse, then per-dimension masks."""
    result=deepcopy(parsed)
    dims=targets(parsed)
    spans=source_spans(response)
    evidence_audit={}
    def evidence(key, decision):
        try:
            support=expand(decision['supporting_span_ranges'],spans,allow_empty=True)
            refutes=expand(decision['refuting_span_ranges'],spans,allow_empty=True)
            evidence_audit[key]={**audit(decision['supporting_span_ranges'],support,spans), 'supporting_span_ids':support, 'refuting_span_ids':refutes}
            return support,refutes,True
        except EvidenceError as exc:
            evidence_audit[key]={'error':exc.code,'raw_decision':decision}
            return [],[],False
    row=deepcopy(parsed['_record'])
    event_ok={}
    if review:
        for eid,decision in review['events'].items():
            ids,refutes,located=evidence('event:'+eid,decision)
            ok=decision['verdict']=='supported' and bool(ids) and located and not refutes and eid in row['events']
            event_ok[eid]=ok
            if ok and eid in row['events']:row['events'][eid]['supporting_span_ranges']=decision['supporting_span_ranges']
        result=parse_events(json.dumps(row,ensure_ascii=False),response)
        event_ok={e:ok and result['_event_validity'].get(e,{}).get('valid',False) for e,ok in event_ok.items()}
        result['_initial_record']=deepcopy(parsed['_record'])
        result['_raw_labels']=deepcopy(parsed['_raw_labels'])
        result['_raw_output']=parsed['_raw_output']
    statuses={}
    for d in dims:
        decision=review['labels'].get(d) if review else None
        ids,refutes,located=evidence('label:'+d,decision) if decision else ([],[],False)
        refs=decision['event_ids'] if decision else []
        # Own explicit supported decision plus located evidence OR explicitly reviewed events.
        ok=bool(decision and decision['verdict']=='supported' and located and not refutes and (ids or refs) and all(event_ok.get(e,False) for e in refs))
        dependencies=set(row[d]['event_ids'])
        dependencies.update(e for e,v in row['events'].items() if v['type'] in ('explicit_error','corrected_error'))
        # Independent partial progress need not establish all unrelated errors.
        if d=='verified_progress' and row[d]['value']==.5:
            dependencies=set(result[d]['event_ids'])
        for _ in row['events']:
            for e in list(dependencies):
                v=row['events'].get(e,{})
                dependencies.update(v.get('related_event_ids',[]))
                if v.get('error_event_id'):dependencies.add(v['error_event_id'])
        ok=ok and all(event_ok.get(e,False) for e in dependencies)
        status='model_supported' if ok and result[d]['valid'] else 'refuted' if decision and decision['verdict']=='refuted' else 'unresolved'
        statuses[d]=status
        if status!='model_supported':
            reason='semantic_'+(failure or status)
            result[d].update(value=None,valid=False,contract_violation=reason)
            if reason not in result['_mask_reasons'][d]:result['_mask_reasons'][d].append(reason)
            result['_violations'].append({'dimension':d,'reason':reason})
    result['_review_source_evidence']=evidence_audit
    result['_semantic_review']={'status':'model_reviewed' if review else 'unresolved','review_source':'targeted_model','human_verified':False,'version':VERSION,'dimensions':statuses,'result':review,'failure':failure,'request_limit_per_trajectory':1,'independent_truth':False}
    result['_training_eligible']={d:result[d]['valid'] for d in DIMS}
    return result
