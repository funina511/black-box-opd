"""Run direct_brief_v1 through the existing transport, budgets and error handling."""
from .core import process_brief
from .judge_client import JudgeClient as BaseJudgeClient, guidance_schema as base_guidance_schema


def guidance_schema(version, schema_override=None, process_protocol='events_v27'):
    if version == 'math_process_v2' and process_protocol == process_brief.PROTOCOL and schema_override is None:
        schema_override = process_brief.SCHEMA
    return base_guidance_schema(version, schema_override, process_protocol)


class JudgeClient(BaseJudgeClient):
    def annotate_one(self, request_id, messages, deadline, *, schema_override=None, parse_override=None, max_retries=None):
        brief = self.options.get('process_protocol') == process_brief.PROTOCOL
        if brief:
            if schema_override is not None or parse_override is not None:
                raise ValueError('direct_brief_v1 uses a single primary annotation')
            schema_override, parse_override = process_brief.SCHEMA, process_brief.parse_brief
        key, result = super().annotate_one(request_id, messages, deadline, schema_override=schema_override,
                                          parse_override=parse_override, max_retries=max_retries)
        if brief:
            result.update(input_protocol=process_brief.INPUT_VERSION, schema_version=process_brief.SCHEMA_VERSION)
        return key, result
