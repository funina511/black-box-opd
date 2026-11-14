"""Coordinator-owned recovery state. Never silently resend an uncertain review."""
import json
import time
from .common import atomic_json,digest
from .semantic_annotation import finish_review


class AnnotationJournal:
    def __init__(self,directory,identity,step=None, *, max_attempts=2, unknown_policy='abort', primary_requests=1, base_request_count=0):
        self.directory=directory/'annotation_state'/digest(identity)/('step_'+str(step) if step is not None else 'standalone')
        self.identity=identity
        self.max_attempts=max_attempts;self.unknown_policy=unknown_policy;self.primary_requests=primary_requests
        self.recovered_request_count=0
        self.base_request_count=base_request_count
    def path(self,rid):
        key=digest(rid)
        return self.directory/key[:2]/(key+'.json')
    def save(self,rid,state,result):
        path=self.path(rid)
        previous=json.loads(path.read_text()) if path.exists() else {}
        attempt=result.get('annotation_attempt',previous.get('annotation_attempt',1))
        prior=previous.get('prior_requests',0)
        reserved=previous.get('reserved_requests',0)
        if attempt!=previous.get('annotation_attempt',attempt):
            prior=previous.get('requests_used',0);reserved=0
        if state=='primary_submitted':reserved+=self.primary_requests
        if state=='review_submitted':reserved+=1
        unknown=(state.endswith('_submitted') or any(s.get('state') not in ('not_submitted','cancelled_before_start') for s in result.get('pending_scheduler_state',[])) or
                 result.get('submission_unknown') or result.get('targeted_result',{}).get('submission_unknown') or
                 result.get('historical_submission_unknown'))
        used=prior+(reserved if unknown else result.get('attempts',0))
        prior_usage=previous.get('prior_usage',{})
        if attempt!=previous.get('annotation_attempt',attempt):prior_usage=previous.get('usage_used',{})
        usage_used={k:prior_usage.get(k,0)+result.get('usage',{}).get(k,0)
                    for k in ('prompt_tokens','completion_tokens','total_tokens')}
        history=previous.get('history',[])+[{'state':state,'annotation_attempt':attempt,
            'time':time.time(),'error':result.get('error'),'requests_used':used}]
        atomic_json(path,{'version':2,'request_id':rid,'protocol_identity':self.identity,'state':state,'result':result,
            'annotation_attempt':attempt,'prior_requests':prior,'reserved_requests':reserved,'requests_used':used,
            'base_request_count':previous.get('base_request_count',self.base_request_count),'history':history,
            'prior_usage':prior_usage,'usage_used':usage_used})
    def recover(self,contexts):
        recovered={}
        for rid,(_,response) in contexts.items():
            path=self.path(rid)
            if not path.exists():continue
            row=json.loads(path.read_text())
            self.base_request_count=min(self.base_request_count,row.get('base_request_count',self.base_request_count))
            if row['request_id']!=rid or row['protocol_identity']!=self.identity:raise ValueError('annotation recovery identity mismatch')
            result=row['result']
            historical_usage=row.get('usage_used',result.get('usage',{}))
            uncertain=(row['state'] in ('primary_submitted','review_submitted') or
                       any(s.get('state') not in ('not_submitted','cancelled_before_start') for s in result.get('pending_scheduler_state',[])) or result.get('submission_unknown') or
                       result.get('targeted_result',{}).get('submission_unknown'))
            if row['state']=='primary_submitted':
                result={'request_id':rid,'error':'primary_submission_state_unknown_after_recovery','failure_kind':'technical',
                        'cacheable':False,'attempts':0,'usage':{},'prior_send_cost':'unknown_not_zero','server_cancellation_confirmed':False}
            if row['state']=='review_submitted':
                failure={'request_id':rid+':targeted','error':'review_submission_state_unknown_after_recovery',
                         'attempts':0,'usage':{},'cacheable':False,'prior_send_cost':'unknown_not_zero',
                         'server_cancellation_confirmed':False}
                result=finish_review(result,response,None,failure)
            self.recovered_request_count+=row.get('requests_used',result.get('attempts',0))
            error=result.get('targeted_result',{}).get('error') or result.get('error')
            retryable=error in ('judge_transport_error','judge_deadline_exceeded','judge_batch_stopped')
            if uncertain and self.unknown_policy=='retry':retryable=True
            if uncertain and self.unknown_policy!='retry':retryable=False
            if retryable:
                if row.get('version')!=2:raise ValueError('legacy failed journal has no attempt ledger; use a new run, do not delete the journal')
                attempt=row['annotation_attempt']
                not_submitted=bool(result.get('pending_scheduler_state')) and all(
                    s.get('state') in ('not_submitted','cancelled_before_start') for s in result['pending_scheduler_state'])
                if attempt>=self.max_attempts and not not_submitted:
                    raise RuntimeError(f'annotation attempt limit reached: request={rid}, attempts={attempt}, requests_used={row["requests_used"]}; fix service and start a new run or explicitly raise the configured bound')
                if 'initial_parsed' in result:
                    result={**result,'parsed':result['initial_parsed'],'attempts':0,'usage':{}}
                    for key in ('targeted_result','targeted_request','initial_parsed','error','pending_scheduler_state','submission_unknown'):
                        result.pop(key,None)
                else:result={'request_id':rid,'attempts':0,'usage':{}}
                result={**result,'annotation_attempt':attempt if not_submitted else attempt+1,'_retry_stage':'review' if 'parsed' in result else 'primary',
                        'possible_duplicate_remote_cost':bool(uncertain)}
            result={**result,'historical_requests_used':row.get('requests_used',0),
                    'historical_usage':historical_usage,
                    'historical_submission_unknown':bool(uncertain)}
            recovered[rid]=result
        self.recovered_request_count+=self.base_request_count
        return recovered
