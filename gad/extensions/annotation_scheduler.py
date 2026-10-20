"""One fair bounded queue for primary -> targeted chains, shared by all callers.

Only the coordinator advances dependencies and publishes final results. A cancelled
future does not cancel server generation; unfinished sends remain explicit.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import time
from .core.semantic_v24 import targets, messages, SCHEMA, parse_review
from .semantic_annotation import finish_review


def annotate_chains(client, requests, contexts, deadline, *, semantic=True,
                    recovered=None, on_result=None, on_state=None, stop=None):
    started=time.monotonic();limit=client.options['concurrency']
    results={};primary=deque();review=deque();inflight={};initial={};queued={}
    metrics={'version':1,'limit':limit,'peak_scheduled_inflight':0,'stages':[],
             'unfinished':[],'future_cancel_is_server_cancel':False}
    recovered=recovered or {};prefer_review=True
    def publish(rid,result):
        results[rid]=result
        if on_state:on_state(rid,'final',result)
        if on_result:on_result(rid,result)
    def needs_review(result):
        return semantic and 'parsed' in result and 'targeted_result' not in result and bool(targets(result['parsed']))
    def enqueue_review(rid,result):
        initial[rid]=result;queued[('review',rid)]=time.monotonic();review.append(rid)
    for rid in requests:
        if rid in recovered:
            r=recovered[rid]
            if needs_review(r):enqueue_review(rid,r)
            else:publish(rid,r)
        else:primary.append(rid);queued[('primary',rid)]=started
    def invoke(stage,rid,enqueued):
        before=time.monotonic()
        try:
            if stage=='primary':r=client.annotate_one(rid,requests[rid],deadline)[1]
            else:
                problem,response=contexts[rid]
                request=messages(problem,response,initial[rid]['parsed'])
                r=client.targeted(rid+':targeted',request,deadline)
                r['_targeted_request']=request
        except Exception as exc:
            r={'request_id':rid,'error':getattr(exc,'code',type(exc).__name__),
               'error_detail':str(exc),'cacheable':False,'attempts':0,'usage':{}}
        timing={'stage':stage,'request_id':rid,'queue_seconds':before-enqueued,
                'worker_seconds':time.monotonic()-before,
                'api_wait_seconds':sum(a.get('seconds',0) for a in r.get('api_attempts',[])),
                'attempts':r.get('attempts',0),'error':r.get('error')}
        r['scheduler_timing']=timing
        return r,timing
    pool=ThreadPoolExecutor(max_workers=limit,thread_name_prefix='rubric-chain')
    try:
        while primary or review or inflight:
            if time.monotonic()>=deadline or (stop and stop()):break
            while len(inflight)<limit and (primary or review):
                stage='review' if review and (prefer_review or not primary) else 'primary'
                prefer_review=stage!='review'
                rid=(review if stage=='review' else primary).popleft()
                if on_state:
                    on_state(rid,'review_submitted' if stage=='review' else 'primary_submitted',
                             initial[rid] if stage=='review' else {'request_id':rid,'attempts':0,'usage':{}})
                future=pool.submit(invoke,stage,rid,queued[(stage,rid)])
                inflight[future]=(stage,rid)
                metrics['peak_scheduled_inflight']=max(metrics['peak_scheduled_inflight'],len(inflight))
            done,_=wait(inflight,timeout=max(0,deadline-time.monotonic()),return_when=FIRST_COMPLETED)
            if not done:break
            # Stable publication for simultaneous completions; IDs never depend on completion order.
            for future in sorted(done,key=lambda f:inflight[f][1]):
                stage,rid=inflight.pop(future);result,timing=future.result();metrics['stages'].append(timing)
                if stage=='primary':
                    if on_state:on_state(rid,'primary_done',result)
                    if needs_review(result):enqueue_review(rid,result)
                    else:publish(rid,result)
                else:
                    request=result.pop('_targeted_request',None)
                    final=finish_review(initial[rid],contexts[rid][1],request,result)
                    final['scheduler_stages']=[initial[rid].get('scheduler_timing'),timing]
                    final['review_wait_seconds']=timing['queue_seconds']
                    publish(rid,final)
    finally:
        for future,(stage,rid) in list(inflight.items()):
            cancelled=future.cancel()
            metrics['unfinished'].append({'request_id':rid,'stage':stage,
                'state':'cancelled_before_start' if cancelled else 'client_deadline_server_may_continue',
                'server_cancellation_confirmed':False})
        pool.shutdown(wait=False,cancel_futures=True)
        for stage,queue in [('primary',primary),('review',review)]:
            for rid in queue:metrics['unfinished'].append({'request_id':rid,'stage':stage,'state':'not_submitted'})
        for rid in requests:
            if rid in results:continue
            failure={'request_id':rid,'error':'judge_deadline_exceeded' if time.monotonic()>=deadline else 'judge_batch_stopped',
                     'failure_kind':'technical','cacheable':False,'attempts':0,'usage':{}}
            if rid in initial:
                failure=finish_review(initial[rid],contexts[rid][1],None,failure)
            failure['pending_scheduler_state']=[s for s in metrics['unfinished'] if s['request_id']==rid]
            publish(rid,failure)
        metrics['batch_wall_seconds']=time.monotonic()-started
        client.last_schedule=metrics
    return {rid:results[rid] for rid in requests}
