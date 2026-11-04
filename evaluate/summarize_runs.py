#!/usr/bin/env python3
"""Accuracy table across runs: every task found under runs/<run>/evaluation/*, plus task means.

One row per (run, method, generation length).  Profiles with the same max_tokens are merged,
so the original four tasks and the extended tasks of a run appear in one row.  When a task
exists in several merged profiles the newest metrics file wins (listed under "notes").

Accuracy is sample accuracy (mean over all samples).  If a task still has unresolved samples
the lower bound C/N is used and the cell is marked "*".  Avg is the equal-weight mean over the
tasks present in the row; Core4/Ext4 are shown only when all of their tasks are present.
Scorer versions are not compared.

  python evaluate/summarize_runs.py baseline_06b_17k correctness_06b_17k_20261021 --csv out.csv
  python evaluate/summarize_runs.py --all --max-tokens 8192
"""
import argparse
import csv
import fnmatch
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CORE = ['AIME24', 'AIME25', 'AMC23', 'MATH-500']
EXT = ['Minerva', 'Olympiad-Bench', 'MMLU-Pro-Math', 'GPQA']


def load(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def metrics_path(task_dir):
    state = load(task_dir / 'state.json') or {}
    recorded = Path(state['metrics_path']) if state.get('metrics_path') else None
    if recorded and recorded.is_file():
        return recorded
    candidates = sorted(task_dir.glob('scores*/metrics.json'), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def max_tokens(profile_dir, task_dir):
    plan = load(profile_dir / 'plan.json') or {}
    value = (plan.get('protocol') or {}).get('max_tokens')
    if value is None:
        value = (load(task_dir / 'generation' / 'sampling_config.json') or {}).get('max_new_tokens')
    if value is None:
        match = re.search(r'_(\d+)k(?:_|$)', profile_dir.name)
        value = int(match.group(1)) * 1024 if match else None
    return int(value) if value is not None else None


def accuracy(metrics):
    partial = bool(metrics.get('unresolved_samples')) or metrics.get('evaluation_complete') is False
    for key in ('sample_accuracy', 'sample_accuracy_lower', 'pass_at_1', 'sample_accuracy_avg_at_n'):
        if metrics.get(key) is not None:
            return float(metrics[key]), partial
    # Early eval-fix metrics keep the lower bound only under provisional_metrics.
    value = (metrics.get('provisional_metrics') or {}).get('sample_accuracy')
    return (float(value), partial) if value is not None else (None, False)


def collect(run_dirs, profiles=None, methods=None, lengths=None):
    rows, notes = {}, {}
    for run_dir in run_dirs:
        for profile_dir in sorted(p for p in (run_dir / 'evaluation').glob('*') if p.is_dir()):
            if profiles and not any(fnmatch.fnmatch(profile_dir.name, pattern) for pattern in profiles):
                continue
            for task_dir in sorted(profile_dir.glob('*/*')):
                method, task = task_dir.parent.name, task_dir.name
                path = metrics_path(task_dir) if task_dir.is_dir() else None
                metrics = load(path) if path else None
                if not metrics or (methods and method not in methods):
                    continue
                value, partial = accuracy(metrics)
                tokens = max_tokens(profile_dir, task_dir)
                if value is None or (lengths and tokens not in lengths):
                    continue
                row = rows.setdefault((run_dir.name, method, tokens), {})
                cell = {'acc': value, 'partial': partial, 'profile': profile_dir.name, 'mtime': path.stat().st_mtime,
                        'unresolved': metrics.get('unresolved_samples') or 0, 'upper': metrics.get('sample_accuracy_upper'),
                        'trunc': metrics.get('truncated_rate'), 'tokens': metrics.get('output_tokens_mean')}
                old = row.get(task)
                if old:
                    keep, drop = (cell, old) if cell['mtime'] >= old['mtime'] else (old, cell)
                    notes.setdefault((run_dir.name, tokens, keep['profile'], drop['profile']), set()).add(f'{method}/{task}')
                    cell = keep
                row[task] = cell
    return rows, [f'{run} ({tokens}): {keep} replaces {drop} for {", ".join(sorted(items))}'
                  for (run, tokens, keep, drop), items in sorted(notes.items())]


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarize(rows):
    tasks = [t for t in CORE + EXT if any(t in r for r in rows.values())]
    tasks += sorted({t for r in rows.values() for t in r} - set(tasks))
    table = []
    for (run, method, tokens), cells in sorted(rows.items(), key=lambda x: (x[0][0], x[0][2] or 0, x[0][1])):
        entry = {'run': run, 'method': method, 'max_tokens': tokens, 'tasks': len(cells),
                 'partial_tasks': sorted(t for t, c in cells.items() if c['partial'])}
        for task in tasks:
            entry[task] = cells[task]['acc'] if task in cells else None
        entry['Avg'] = mean(c['acc'] for c in cells.values())
        for name, group in (('Core4', CORE), ('Ext4', EXT)):
            entry[name] = mean(cells[t]['acc'] for t in group) if all(t in cells for t in group) else None
        entry['mean_trunc'] = mean(c['trunc'] for c in cells.values())
        entry['mean_tokens'] = mean(c['tokens'] for c in cells.values())
        table.append(entry)
    return tasks, table


def fmt(value, width=6):
    return f"{'-' if value is None else f'{100 * value:.1f}':>{width}}"


def render(tasks, table):
    short = {'Olympiad-Bench': 'Olymp', 'MMLU-Pro-Math': 'MMLU-P', 'MATH-500': 'MATH5', 'Minerva': 'Minrv'}
    head = ['run'.ljust(30), 'method'.ljust(9), 'len'.rjust(5)] + [f'{short.get(t, t)[:6]:>6}' for t in tasks]
    head += ['   Avg', ' Core4', '  Ext4', ' nTask', ' trunc', 'tokens']
    lines = [' '.join(head)]
    for e in table:
        cells = [e['run'][:30].ljust(30), e['method'][:9].ljust(9), f"{e['max_tokens'] or '?':>5}"]
        for t in tasks:
            text = fmt(e[t])
            cells.append(text[1:] + '*' if t in e['partial_tasks'] else text)
        cells += [fmt(e['Avg']), fmt(e['Core4']), fmt(e['Ext4']), f"{e['tasks']:>6}", fmt(e['mean_trunc']),
                  f"{'-' if e['mean_tokens'] is None else round(e['mean_tokens']):>6}"]
        lines.append(' '.join(cells))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', help='run ids under --runs-root, or run directories')
    parser.add_argument('--all', action='store_true', help='every run under --runs-root with an evaluation/ directory')
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'runs')
    parser.add_argument('--profile', action='append', help='profile name or glob; may be repeated (default: all)')
    parser.add_argument('--methods', help='comma-separated methods (default: all)')
    parser.add_argument('--max-tokens', type=int, action='append', help='keep only these generation lengths')
    parser.add_argument('--csv', type=Path)
    args = parser.parse_args()
    if args.all:
        run_dirs = sorted(p for p in args.runs_root.iterdir() if (p / 'evaluation').is_dir())
    else:
        run_dirs = [Path(r) if Path(r).is_dir() else args.runs_root / r for r in args.runs]
    missing = [str(d) for d in run_dirs if not (d / 'evaluation').is_dir()]
    if not run_dirs or missing:
        raise SystemExit(f'no evaluation directory: {missing or "give run ids or --all"}')
    rows, notes = collect(run_dirs, args.profile, set(args.methods.split(',')) if args.methods else None,
                          set(args.max_tokens) if args.max_tokens else None)
    if not rows:
        raise SystemExit('no evaluation metrics found')
    tasks, table = summarize(rows)
    print(render(tasks, table))
    print('\naccuracy in %; * = task has unresolved samples, lower bound C/N used; Avg = mean over present tasks')
    for note in notes:
        print('note:', note)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fields = ['run', 'method', 'max_tokens', *tasks, 'Avg', 'Core4', 'Ext4', 'tasks', 'partial_tasks', 'mean_trunc', 'mean_tokens']
        with args.csv.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for entry in table:
                writer.writerow({**entry, 'partial_tasks': ';'.join(entry['partial_tasks'])})
        print(f'CSV -> {args.csv}')


if __name__ == '__main__':
    main()
