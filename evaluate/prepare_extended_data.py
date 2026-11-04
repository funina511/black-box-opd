#!/usr/bin/env python3
"""Import the four agreed OPRD test sets without changing prompts or answer order."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
EXTENDED_TASKS = ('Minerva', 'Olympiad-Bench', 'GPQA', 'MMLU-Pro-Math')
CHOICE_TASKS = ('GPQA', 'MMLU-Pro-Math')
DAPO_VERSION = 'extended_dapo_v2'
# v2 adds one sentence to choice questions only; v1 (no sentence) is kept for reproduction.
CHOICE_INSTRUCTIONS = {
    'extended_dapo_v1': None,
    'extended_dapo_v2': 'The last line must contain only the uppercase letter of the corresponding option '
                        'in the question, for example: Answer: B',
}
# Same instruction as the AIME/AMC/MATH-500 evaluation files and the training prompts.
DAPO_PREFIX = ('Solve the following math problem step by step. The last line of your response should be '
               'of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n')
BOXED_SUFFIX = ' Please reason step by step, and put your final answer within \\boxed{}.'
_LOWERCASE_LABEL = re.compile(r'(?m)^\s*(?:\(?[a-z]\)|[a-z]\.)\s')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source_root, destination):
    source_root, destination = Path(source_root), Path(destination)
    manifest = {'version': 1, 'tasks': {}, 'protocol': 'OPRD stored zero-shot prompts; original option order',
                'note': 'Subset names describe this local OPRD snapshot, not the complete upstream benchmarks.'}
    for source_name, task in [('Minerva', 'Minerva'), ('Olympiad-Bench', 'Olympiad-Bench'),
                              ('GPQA', 'GPQA'), ('MMLU-Pro', 'MMLU-Pro-Math')]:
        source = source_root / source_name / 'test.parquet'
        table = pq.read_table(source)
        records = table.to_pylist()
        selected = [(i, row) for i, row in enumerate(records)
                    if source_name != 'MMLU-Pro' or row['category'] == 'math']
        output = destination / task / 'test.parquet'
        if output.exists():
            raise FileExistsError(f'preserve existing evaluation asset: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        if source_name in ('GPQA', 'MMLU-Pro'):
            rows = []
            for source_index, row in selected:
                row = dict(row)
                count = 4 if source_name == 'GPQA' else int(row['option_num'])
                truth = row['reward_model']['ground_truth']
                if truth not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count] or len(truth) != 1:
                    raise ValueError(f'{task}: invalid source option key at row {source_index}')
                row['extra_info'] = {**row['extra_info'], 'option_num': count,
                                     'source_row_index': source_index}
                row['data_source'] = task
                rows.append(row)
            pq.write_table(pa.Table.from_pylist(rows), output)
        else:
            shutil.copyfile(source, output)
        manifest['tasks'][task] = {'source_file': f'{source_name}/test.parquet',
            'source_sha256': sha256(source), 'source_rows': len(records),
            'rows': len(selected), 'selection': 'category == math' if source_name == 'MMLU-Pro' else 'all',
            'source_row_indices': [i for i, _ in selected],
            'path': f'{task}/test.parquet', 'sha256': sha256(output),
            'scorer': 'choice_exact_v1' if source_name in ('GPQA', 'MMLU-Pro') else 'math_unified_v1',
            'prompts_and_ground_truth_preserved': True}
    (destination / 'extended_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def to_dapo(content):
    """Replace the boxed instruction with the DAPO answer-line instruction (OPRD test_data_dapo rule)."""
    if not content.endswith(BOXED_SUFFIX):
        raise ValueError('prompt does not end with the expected boxed instruction')
    return DAPO_PREFIX + content[:-len(BOXED_SUFFIX)].strip()


def parse_options(question, task, count):
    """Option texts in label order, or None when the split is not unique."""
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:count]
    if task == 'GPQA':
        start = question.rfind('\n\nA. ')
        pattern = r'\n\n' + r'\n'.join(rf'{label}\. (.*?)' for label in labels) + r'\s*'
        match = re.fullmatch(pattern, question[start:], re.S) if start >= 0 else None
        stem = question[:start] if start >= 0 else question
        if not match:
            return None, 'unparsed', stem
        return [text.strip() for text in match.groups()], 'ok', stem
    marker = '\nAnswer Choices: '
    start = question.rfind(marker)
    if start < 0:
        return None, 'unparsed', question
    tail, stem = question[start + len(marker):], question[:start]
    # A label repeated inside option text, e.g. "(I) and (IV)", makes the split ambiguous.
    if not tail.startswith('(A) ') or any(tail.count(f' ({label}) ') != (label != 'A') for label in labels):
        return None, 'ambiguous', stem
    bounds = [0] + [tail.index(f' ({label}) ') for label in labels[1:]] + [len(tail)]
    texts = [tail[begin:end].strip()[4:].strip() for begin, end in zip(bounds, bounds[1:])]
    rebuilt = ' '.join(f'({label}) {text}' for label, text in zip(labels, texts))
    same = rebuilt.split() == tail.split()  # the source has irregular double spaces between options
    return (texts, 'ok', stem) if same and all(texts) else (None, 'unparsed', stem)


def prepare_dapo(source_root=ROOT / 'assets/datasets/math_eval', destination=None, oprd_dapo_root=None,
                 version=DAPO_VERSION):
    """Write the extended tasks with the DAPO prompt; questions, answers and IDs are unchanged."""
    source_root = Path(source_root)
    destination = Path(destination or source_root / version)
    instruction = CHOICE_INSTRUCTIONS[version]
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'preserve existing evaluation asset: {manifest_path}')
    manifest = {'version': version, 'prompt_prefix': DAPO_PREFIX, 'removed_suffix': BOXED_SUFFIX,
                'choice_instruction': instruction,
                'rule': 'DAPO_PREFIX + question.removesuffix(BOXED_SUFFIX).strip()'
                        + (' + "\\n\\n" + choice_instruction (choice tasks only)' if instruction else '')
                        + '; ground truth, option order, extra_info.index and row order unchanged; '
                          'choice tasks add extra_info.options',
                'tasks': {}}
    for task in EXTENDED_TASKS:
        source = source_root / task / 'test.parquet'
        output = destination / task / 'test.parquet'
        if output.exists():
            raise FileExistsError(f'preserve existing evaluation asset: {output}')
        rows, stats = [], {}
        for row in pq.read_table(source).to_pylist():
            row = dict(row)
            if len(row['prompt']) != 1 or row['prompt'][0]['role'] != 'user':
                raise ValueError(f'{task}: expected one user message')
            content = row['prompt'][0]['content']
            prompt = to_dapo(content)
            if task in CHOICE_TASKS and instruction:
                prompt += '\n\n' + instruction
            row['prompt'] = [{**row['prompt'][0], 'content': prompt}]
            if task in CHOICE_TASKS:
                question = content[:-len(BOXED_SUFFIX)].strip()
                options, status, stem = parse_options(question, task, row['extra_info']['option_num'])
                lowercase = bool(_LOWERCASE_LABEL.search(stem))
                row['extra_info'] = {**row['extra_info'], 'options': options, 'options_parse': status,
                                     'lowercase_labels_in_stem': lowercase}
                stats[status] = stats.get(status, 0) + 1
                stats['lowercase_labels_in_stem'] = stats.get('lowercase_labels_in_stem', 0) + lowercase
            rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), output)
        entry = {'source_file': str(source.relative_to(source_root)), 'source_sha256': sha256(source),
                 'rows': len(rows), 'path': f'{task}/test.parquet', 'sha256': sha256(output),
                 'scorer': 'choice_exact_v2' if task in CHOICE_TASKS else 'math_unified_v1',
                 'options_parse': stats or None}
        reference = Path(oprd_dapo_root) / task / 'test.parquet' if oprd_dapo_root else None
        if reference and reference.is_file():
            # GPQA and Olympiad-Bench already exist in OPRD test_data_dapo; record agreement.
            theirs = pq.read_table(reference).to_pylist()
            strip = (lambda p: p[0]['content'].removesuffix('\n\n' + instruction)) if instruction else (lambda p: p[0]['content'])
            entry['oprd_test_data_dapo'] = {'sha256': sha256(reference), 'rows': len(theirs),
                'prompt_equal': sum(a['prompt'][0]['content'] == strip(b['prompt']) for a, b in zip(theirs, rows)),
                'prompt_equal_note': 'excluding the v2 choice instruction' if instruction and task in CHOICE_TASKS else None,
                'ground_truth_equal': sum(a['reward_model']['ground_truth'] == b['reward_model']['ground_truth']
                                          for a, b in zip(theirs, rows))}
        manifest['tasks'][task] = entry
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--dapo', action='store_true',
                        help='Write the DAPO-prompt version from an existing local extended evaluation root')
    parser.add_argument('--oprd-dapo-root', type=Path, help='Optional OPRD test_data_dapo root for agreement checks')
    parser.add_argument('--dapo-version', choices=sorted(CHOICE_INSTRUCTIONS), default=DAPO_VERSION)
    args = parser.parse_args()
    if args.dapo:
        result = prepare_dapo(args.source_root, args.output_root, args.oprd_dapo_root, args.dapo_version)
    else:
        result = prepare(args.source_root, args.output_root or ROOT / 'assets/datasets/math_eval')
    print(json.dumps({task: item['rows'] for task, item in result['tasks'].items()}, indent=2))
