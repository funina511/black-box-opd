"""One visible trajectory, two direct process labels, no evidence/review protocol."""
import json
from .contracts import digest,file_hash

PROTOCOL='direct_v1'
SCHEMA_VERSION='rubric.direct.v1'
INPUT_VERSION='rubric.direct.input.v1'
DIMS=('verified_progress','unresolved_error_severity')
SCHEMA={'type':'object','additionalProperties':False,'required':list(DIMS),
        'properties':{d:{'enum':[0,.5,1,None]} for d in DIMS}}


class DirectProtocolError(ValueError):
    code='judge_direct_schema_error'


def parse_direct(text):
    def unique(pairs):
        result={}
        for k,v in pairs:
            if k in result:raise DirectProtocolError('duplicate label key')
            result[k]=v
        return result
    try:row=json.loads(text,object_pairs_hook=unique,parse_constant=lambda _:(_ for _ in ()).throw(DirectProtocolError('nonfinite JSON value')))
    except (ValueError,TypeError) as exc:raise DirectProtocolError(str(exc)) from exc
    if not isinstance(row,dict) or set(row)!=set(DIMS):raise DirectProtocolError('exactly two direct label fields required')
    result={}
    for d,value in row.items():
        if value is not None and (type(value) not in (int,float) or value not in (0,.5,1)):
            raise DirectProtocolError('labels must be 0, 0.5, 1 or null')
        result[d]={'value':value,'valid':value is not None,'source':'judge_direct_v1'}
    result.update(_process_protocol=PROTOCOL,_schema_version=SCHEMA_VERSION,
        _raw_output=text,_raw_labels=row,
        _semantic_review={'status':'not_requested','review_source':None,'human_verified':False,'independent_truth':False},
        _training_eligible={d:result[d]['valid'] for d in DIMS})
    return result


def messages(problem,response,ground_truth,spec,instruction,visibility=None):
    return [{'role':'system','content':instruction.strip()+'\n\n固定判据：\n'+spec.strip()},
            {'role':'user','content':json.dumps({'problem':problem,'response':response,
                'reference_final_answer':ground_truth,'visibility':visibility or {}},ensure_ascii=False,separators=(',',':'))}]


def protocol_identity(spec,instruction):
    from . import contracts
    return {'process_protocol':PROTOCOL,'schema_version':SCHEMA_VERSION,'input_protocol':INPUT_VERSION,
            'schema_hash':digest(SCHEMA),'spec_hash':digest(spec),'instructions_hash':digest(instruction),
            'parser_hash':file_hash(__file__),'request_builder_hash':file_hash(contracts.__file__)}
