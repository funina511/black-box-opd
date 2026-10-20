import hashlib
import json

DIMENSIONS = ("answer_correctness", "reasoning_validity", "condition_coverage")
LAYOUT = ("adv", *DIMENSIONS)
ANNOTATION_SCHEMA = "rubric.annotation.v1"
V2_DIMENSIONS = ('answer_correctness', 'verified_progress', 'unresolved_error_severity')
V2_LAYOUT = ('adv', *V2_DIMENSIONS)
V2_SCHEMA_VERSION = 'rubric.events.v2.7'
V2_INPUT_VERSION = 'rubric.input.v2.5'


def layout_for(version):
    if version == 'math_v1':return LAYOUT
    if version == 'math_process_v2':return V2_LAYOUT
    raise ValueError('unsupported rubric version: ' + str(version))


def schema_for(version):
    if version == 'math_v1':return JUDGE_SCHEMA
    if version == 'math_process_v2':
        from .process_v2 import JUDGE_SCHEMA_V2
        return JUDGE_SCHEMA_V2
    raise ValueError('unsupported rubric version')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_label(item, dimension):
    value, valid = item.get("value"), item.get("valid")
    if type(valid) is not bool:
        raise ValueError("label valid must be boolean")
    if not valid and value is not None:
        raise ValueError("unknown label must be null")
    allowed = (0, 1) if dimension == DIMENSIONS[0] else (0, .5, 1)
    if valid and (isinstance(value, bool) or value not in allowed):
        raise ValueError(f"invalid {dimension} label: {value}")


def parse_judge(text, version='math_v1', response=None):
    if version == 'math_process_v2':
        from .process_v2 import parse_events
        return parse_events(text, response)
    row = json.loads(text)
    if set(row) != set(DIMENSIONS[1:]):
        raise ValueError("judge must return exactly two process dimensions")
    for dim, item in row.items():
        validate_label(item, dim)
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            raise ValueError("judge requires evidence or an unknown reason")
    return row


JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": list(DIMENSIONS[1:]),
                "properties": {dim: {"type": "object", "additionalProperties": False,
                                     "required": ["value", "valid", "evidence"],
                                     "properties": {"value": {"enum": [0, .5, 1, None]},
                                                    "valid": {"type": "boolean"},
                                                    "evidence": {"type": "string", "minLength": 1}}}
                               for dim in DIMENSIONS[1:]}}


def judge_messages(prompt, response, ground_truth, spec, instruction, version='math_v1', truncation=None):
    if version == 'math_process_v2':
        from .process_v2 import compact_response, render_instructions, MARKER
        return [{'role': 'system', 'content': render_instructions(spec, instruction)},
                {'role': 'user', 'content': json.dumps({'input_protocol': V2_INPUT_VERSION, 'problem': prompt,
                    'reference_final_answer': ground_truth, 'visibility': truncation or {},
                    'response_characters': len(response), 'response_hash': digest(response),
                    'rubric_version': version}, ensure_ascii=False, separators=(',', ':'))
                 + MARKER + compact_response(response)}]
    return [{"role": "system", "content": instruction + "\n固定 rubric:\n" + spec},
            {"role": "user", "content": "待评材料 JSON:\n" + json.dumps(
                {"problem": prompt, "response": response, "reference_answer": ground_truth},
                ensure_ascii=False)}]
