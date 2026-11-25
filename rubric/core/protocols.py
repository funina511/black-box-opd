"""Dispatch the current process protocols while preserving frozen legacy contracts."""
from . import contracts, process_brief, process_direct

DIRECT_PROTOCOLS = (process_direct.PROTOCOL, process_brief.PROTOCOL)


def protocol_module(protocol):
    if protocol == process_brief.PROTOCOL:
        return process_brief
    if protocol == process_direct.PROTOCOL:
        return process_direct
    from . import process_v2
    return process_v2


def schema_for(version, process_protocol='events_v27'):
    if version == 'math_process_v2' and process_protocol == process_brief.PROTOCOL:
        return process_brief.SCHEMA
    return contracts.schema_for(version, process_protocol)


def parse_judge(text, version='math_v1', response=None, process_protocol='events_v27'):
    if version == 'math_process_v2' and process_protocol == process_brief.PROTOCOL:
        return process_brief.parse_brief(text)
    return contracts.parse_judge(text, version, response, process_protocol)


def judge_messages(prompt, response, ground_truth, spec, instruction, version='math_v1', truncation=None, process_protocol='events_v27'):
    if version == 'math_process_v2' and process_protocol == process_brief.PROTOCOL:
        return process_direct.messages(prompt, response, ground_truth, spec, instruction, truncation)
    return contracts.judge_messages(prompt, response, ground_truth, spec, instruction, version, truncation, process_protocol)
