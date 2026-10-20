"""Coordinator-owned recovery state. Never silently resend an uncertain review."""
import json
from .common import atomic_json,digest
from .semantic_annotation import finish_review


class AnnotationJournal:
    def __init__(self,directory,identity,step=None):
        self.directory=directory/'annotation_state'/digest(identity)/('step_'+str(step) if step is not None else 'standalone')
        self.identity=identity
    def path(self,rid):
        key=digest(rid)
        return self.directory/key[:2]/(key+'.json')
    def save(self,rid,state,result):
        atomic_json(self.path(rid),{'request_id':rid,'protocol_identity':self.identity,'state':state,'result':result})
    def recover(self,contexts):
        recovered={}
        for rid,(_,response) in contexts.items():
            path=self.path(rid)
            if not path.exists():continue
            row=json.loads(path.read_text())
            if row['request_id']!=rid or row['protocol_identity']!=self.identity:raise ValueError('annotation recovery identity mismatch')
            result=row['result']
            if row['state']=='primary_submitted':
                result={'request_id':rid,'error':'primary_submission_state_unknown_after_recovery','failure_kind':'technical',
                        'cacheable':False,'attempts':0,'usage':{},'prior_send_cost':'unknown_not_zero','server_cancellation_confirmed':False}
            if row['state']=='review_submitted':
                failure={'request_id':rid+':targeted','error':'review_submission_state_unknown_after_recovery',
                         'attempts':0,'usage':{},'cacheable':False,'prior_send_cost':'unknown_not_zero',
                         'server_cancellation_confirmed':False}
                result=finish_review(result,response,None,failure)
            recovered[rid]=result
        return recovered
