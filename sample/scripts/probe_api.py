#!/usr/bin/env python3
"""Send ONE teacher request built from a sampling YAML and show whether thinking was really off.

The request matches teacher_engines.OpenAIChatEngine (model, messages, temperature, top_p,
max_tokens, extra_body).  Nothing is written to the dataset directories; the API key is read
from the environment variable named by teacher.api_key_env and never printed.

  export DEEPSEEK_API_KEY=...
  python sample/scripts/probe_api.py --config sample/configs/sample_deepseek_api.yaml
  python sample/scripts/probe_api.py --config ... --row 5 --out /tmp/probe.json
"""
import argparse
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--row', type=int, default=0, help='row of prompts.input_parquet to send')
    parser.add_argument('--out', type=Path, help='save the full response JSON here')
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    base = args.config.resolve().parent
    teacher, prompts = cfg['teacher'], cfg['prompts']
    key = os.environ.get(teacher['api_key_env'])
    if not key:
        raise SystemExit(f"set {teacher['api_key_env']} first")
    row = pq.read_table(base / prompts['input_parquet']).slice(args.row, 1).to_pylist()[0]
    request = {'model': teacher['model'], 'messages': row['prompt'], 'temperature': teacher['temperature'],
               'top_p': teacher['top_p'], 'max_tokens': teacher['max_tokens']}
    if teacher.get('extra_body'):
        request['extra_body'] = teacher['extra_body']

    from openai import OpenAI
    client = OpenAI(base_url=teacher['base_url'], api_key=key, timeout=teacher.get('timeout', 600))
    print(f"POST {teacher['base_url']} model={teacher['model']} extra_body={teacher.get('extra_body')}")
    start = time.time()
    response = client.chat.completions.create(**request)
    elapsed = time.time() - start

    data = response.model_dump()
    choice = data['choices'][0]
    message = choice.get('message') or {}
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or message.get('reasoning') or ''
    usage = data.get('usage') or {}
    reasoning_tokens = (usage.get('completion_tokens_details') or {}).get('reasoning_tokens')

    student_tokens = None
    tokenizer_path = (cfg.get('student') or {}).get('tokenizer_path')
    if tokenizer_path and content:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base / tokenizer_path)
        student_tokens = len(tokenizer(content, add_special_tokens=False)['input_ids'])

    lines = [line for line in content.strip().splitlines() if line.strip()]
    print(f"elapsed            {elapsed:.1f}s")
    print(f"returned model     {data.get('model')}")
    print(f"finish_reason      {choice.get('finish_reason')}")
    print(f"usage              prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')} "
          f"reasoning={reasoning_tokens}")
    print(f"reasoning_content  {len(reasoning)} chars")
    print(f"content            {len(content)} chars, {student_tokens} student tokens")
    print(f"last line          {lines[-1] if lines else ''}")
    print(f"ground truth       {row['reward_model']['ground_truth']}")
    print('--- content head ---')
    print(content[:800])
    if reasoning:
        print('--- reasoning head ---')
        print(reasoning[:400])
    non_thinking = not reasoning and not reasoning_tokens and '<think>' not in content
    print(f"\nnon-thinking: {'YES' if non_thinking else 'NO'}"
          + ('' if non_thinking else ' (reasoning text/tokens or <think> present)'))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'response -> {args.out}')


if __name__ == '__main__':
    main()
