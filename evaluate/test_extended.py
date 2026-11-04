"""CPU checks for final-option grading and the exact agreed dataset subsets."""
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from evaluate import answer_forms, choice, minerva
from evaluate.prepare_extended_data import BOXED_SUFFIX, CHOICE_INSTRUCTIONS, DAPO_PREFIX, prepare, prepare_dapo
from evaluate.rescore import run
from evaluate.summarize_eval import collect
from scoring.math import evaluation_value, identity as math_identity


class ChoiceTests(unittest.TestCase):
    def row(self, response, truth='B', count=4, **extra):
        return {'response': response, 'ground_truth': truth, 'extra_info': {'option_num': count}, **extra}

    def test_explicit_final_choice_not_reasoning_letters(self):
        for text in [r'\boxed{B}', r'\boxed{\text{B}}', 'Answer: (B)',
                     '**Final answer:** B', 'The answer is B.', 'b',
                     'I considered A.\nAnswer: B', '<think>Answer: A</think>\nAnswer: B',
                     r'First \boxed{A}.' + '\nAnswer: B']:
            with self.subTest(text=text):
                self.assertEqual(choice.score(self.row(text), 'GPQA')['evaluation_value'], 1)

    def test_invalid_final_never_falls_back_to_earlier_correct_choice(self):
        for text in ['A and B are candidates.', 'Answer: A or B', 'Answer: E',
                     r'Answer: \boxed{A} or \boxed{B}', r'Answer: \boxed{B} or A',
                     '\\boxed{B}\nAnswer: A or B', 'Answer: B+C', 'No conclusion yet.']:
            with self.subTest(text=text):
                self.assertEqual(choice.score(self.row(text), 'GPQA')['evaluation_value'], 0)

    def test_actual_option_count_and_invalid_metadata(self):
        self.assertEqual(choice.score(self.row('Answer: J', 'J', 10), 'MMLU-Pro-Math')['evaluation_value'], 1)
        self.assertEqual(choice.score(self.row('Answer: J', 'B', 9), 'MMLU-Pro-Math')['evaluation_value'], 0)
        self.assertIsNone(choice.score(self.row('Answer: J', 'J', 9), 'MMLU-Pro-Math')['evaluation_value'])
        self.assertIsNone(choice.score({'response': 'B', 'ground_truth': 'B'}, 'MMLU-Pro-Math')['evaluation_value'])
        self.assertEqual(choice.score({'response': 'B', 'ground_truth': 'B'}, 'GPQA')['evaluation_value'], 1)

    def test_rescore_fixed_denominator_identity_and_summary(self):
        before = math_identity()['fingerprint']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / 'gad/GPQA'
            task.mkdir(parents=True)
            raw = task / 'raw.jsonl'
            rows = [self.row(r'\boxed{B}', global_index=0, rollout_index=0, finish_reason='stop', output_tokens=20),
                    self.row('Still working', global_index=1, rollout_index=0, finish_reason='length', output_tokens=8192),
                    self.row('Answer: Z', truth='Z', global_index=2, rollout_index=0)]
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            metrics = run(raw, task / 'scores', expected_questions=[0, 1, 2, 3], n=1, task='GPQA')
            self.assertEqual((metrics['confirmed_correct'], metrics['confirmed_incorrect'], metrics['unresolved_samples']), (1, 1, 2))
            self.assertEqual(metrics['sample_accuracy_lower'], .25)
            self.assertEqual(metrics['sample_accuracy_upper'], .75)
            self.assertEqual(metrics['scorer_version'], choice.VERSION)
            self.assertEqual(collect(root)[0]['truncated_rate'], .25)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                run(raw, task / 'other', config={'choice_fingerprint': 'stale'}, task='GPQA')
        self.assertEqual(before, math_identity()['fingerprint'])

    def test_common_explicit_final_forms(self):
        for text in ['**Final Answer**: B', '### Final Answer\n\nB', '**Final Answer**\n\n$$\n\\boxed{B}\n$$',
                     'The correct answer is B.', 'The correct option is (B).', 'Therefore, the answer is **B**.',
                     'Thus the answer is B, since 7 > 3.', r'\boxed{\textbf{B}}', r'Answer: \boxed{\mathbf{B}}',
                     'Answer: B\nWe see the answer is not A.', 'Option A is too large.\nAnswer: B',
                     'The answer is B because A is too small.', 'Answer: B\nThe answer is a bit tricky to verify.']:
            with self.subTest(text=text):
                self.assertEqual(choice.score(self.row(text), 'GPQA')['evaluation_value'], 1)

    def test_declaration_naming_several_options_is_ambiguous(self):
        # A later ambiguous declaration must neither pick one letter nor fall back to an earlier answer.
        for text in ['Therefore, the answer is B and A.', 'Answer: B\nTherefore, the final answer is A or B.',
                     'The answer is B, A.', 'the answer is (B) or (C)', 'Answer: B and C', 'Answer: B or C']:
            with self.subTest(text=text):
                result = choice.score(self.row(text), 'GPQA')
                self.assertEqual((result['evaluation_value'], result['parse_status']), (0, 'ambiguous'))

    def test_letter_with_its_own_option_text(self):
        values = ['42', '7', '$0.50', 'True, False']
        for text, expected in [(r'\boxed{B. 7}', 1), ('Answer: (B) 7', 1), (r'\boxed{(D) True, False}', 0),
                               ('Answer: (B) 42', 0), (r'\boxed{B. 8}', 0)]:
            with self.subTest(text=text):
                row = {**self.row(text), 'extra_info': {'option_num': 4, 'options': values}}
                self.assertEqual(choice.score(row, 'MMLU-Pro-Math')['evaluation_value'], expected)
        # Without option texts a letter followed by text is never guessed.
        self.assertEqual(choice.score(self.row(r'\boxed{B. 7}'), 'GPQA')['evaluation_value'], 0)

    def test_option_content_is_diagnostic_only(self):
        row = {**self.row(r'\boxed{0.5}', truth='C'), 'extra_info': {'option_num': 4, 'options': ['42', '7', '$0.50', '1']}}
        result = choice.score(row, 'MMLU-Pro-Math')
        self.assertEqual((result['evaluation_value'], result['evaluation_status']), (0, 'format_failure'))
        self.assertEqual((result['content_option'], result['content_option_correct']), ('C', True))

    def test_lowercase_letter_ambiguous_when_stem_has_lowercase_labels(self):
        row = {**self.row(r'\boxed{d}', truth='D'), 'extra_info': {'option_num': 4, 'lowercase_labels_in_stem': True}}
        self.assertEqual(choice.score(row, 'GPQA')['evaluation_value'], 0)
        self.assertEqual(choice.score(self.row(r'\boxed{d}', truth='D'), 'GPQA')['evaluation_value'], 1)

    def test_conflicting_truth_stays_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw.jsonl'
            rows = [self.row('B', truth=truth, global_index=0, rollout_index=i)
                    for i, truth in enumerate(('A', 'B'))]
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            result = run(raw, root / 'scores', expected_questions=[0], n=2, task='GPQA')
            self.assertEqual(result['unresolved_samples'], 2)


class MinervaTests(unittest.TestCase):
    def test_trailing_unit_does_not_change_number(self):
        for truth, text in [('1.6', 'Answer: 1.6 cm'), ('1.6', r'Answer: $1.6 \, \text{cm}$'), ('100', r'Answer: 100 \text{ K}'),
                            ('41.8', r'Answer: 41.8^\circ'), ('4.5e33', r'Answer: $4.5\times10^{33}$ erg/s'),
                            ('3e8', r'Answer: $3 \times 10^8$ m/s'), ('1.6', r'\boxed{1.6 \mathrm{~cm}}')]:
            with self.subTest(text=text):
                result = minerva.score(text, truth)
                self.assertEqual((result['grade_status'], result['unit_stripped']), ('correct', True))

    def test_only_units_are_removed(self):
        for truth, text in [('1.6', 'Answer: 1.58 cm'), ('1.6', 'Answer: 16 mm'), ('5', 'Answer: 5n'),
                            ('5', r'Answer: 5 \pi'), ('5', 'Answer: 5/2'), ('5', 'Answer: 5 + 2 m'),
                            ('5', r'Answer: 5 \text{or 6}'), ('5', r'Answer: 5 \mathrm{n}'), ('5', r'Answer: 5 \text{+ 2}'),
                            ('5', r'Answer: 5 \text{ or } 6'), ('5', r'Answer: 5 \text{ approximately}')]:
            with self.subTest(text=text):
                self.assertEqual(minerva.score(text, truth)['grade_status'], 'incorrect')

    def test_rescore_records_post_process(self):
        before = math_identity()['fingerprint']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw.jsonl'
            rows = [{'response': 'Answer: 1.6 cm', 'ground_truth': '1.6', 'global_index': 0, 'rollout_index': 0},
                    {'response': 'Answer: 2 cm', 'ground_truth': '1.6', 'global_index': 1, 'rollout_index': 0}]
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            metrics = run(raw, root / 'scores', expected_questions=[0, 1], n=1, task='Minerva')
            self.assertEqual((metrics['confirmed_correct'], metrics['unit_stripped_samples'], metrics['unit_stripped_correct']), (1, 2, 1))
            self.assertEqual(metrics['eval_postprocess']['version'], minerva.VERSION)
            with self.assertRaisesRegex(ValueError, 'post-process fingerprint mismatch'):
                run(raw, root / 'other', config={'minerva_fingerprint': 'stale'}, task='Minerva')
        self.assertEqual(before, math_identity()['fingerprint'])


class AnswerFormTests(unittest.TestCase):
    def status(self, truth, text, task='Olympiad-Bench'):
        return answer_forms.score(text, truth, task=task)['grade_status']

    def test_previously_unresolved_forms_become_correct(self):
        for truth, text in [
                ('$(6,3),(9,3),(9,5),(54,5)$', 'Answer: (9, 3), (6, 3), (9, 5), (54, 5)'),
                ('$(3,0),(-3,1)$', 'Answer: (3, 0) and (-3, 1)'),
                ('$(4, \\log _{10} 2)$', 'Answer: $(4, \\log_{10} 2)$'),
                ('$\\quad(2+\\sqrt{2}, 1+\\sqrt{2})$', 'Answer: (2 + \\sqrt{2}, 1 + \\sqrt{2})'),
                ('$f(x)=2 x$', 'Answer: f(x) = 2x'), ('$f(x)=2 x$', 'Answer: 2x'),
                ('$f(n)=n^{2}$', 'Answer: f(m) = m^2'), ('$k=1$', 'Answer: 1'), ('$k=1$', 'Answer: k = 1'),
                ('$M=2^{N-2}+1$', 'Answer: M = 2^{N-2} + 1'), ('$m_{\\max }=n^{2}-n-1$', 'Answer: n^2 - n - 1'),
                ('$f(x)=-1$,$f(x)=x+1$', 'Answer: f(x) = x + 1 and f(x) = -1'),
                ('$a=2$, $a=-6-4 \\sqrt{2}$', 'Answer: a = -6 - 4\\sqrt{2}, a = 2'),
                ('$90^{\\circ}$,$60^{\\circ}$', 'Answer: 60^\\circ, 90^\\circ'),
                ('$m n-\\lfloor m / 2\\rfloor$', 'Answer: mn - \\left\\lfloor \\frac{m}{2} \\right\\rfloor'),
                ('$\\lceil n / 2\\rceil+1$', 'Answer: \\lceil \\frac{n}{2} \\rceil + 1'),
                ('$\\binom{2n}{n}$', 'Answer: \\binom{2n}{n}'), ('$\\frac{N}{2}$', 'Answer: N/2'),
                ('$2RC+R+C$', 'Answer: 2 R C + R + C'), ('$\\frac{d !}{2 k(d-k) !}$', 'Answer: \\frac{d!}{2k(d-k)!}')]:
            with self.subTest(truth=truth, text=text):
                self.assertEqual(self.status(truth, text), 'correct')

    def test_minerva_forms_keep_unit_rule(self):
        for truth, text in [('np.arcsin(10/13)', 'Answer: \\arcsin\\left(\\frac{1}{1.3}\\right)'),
                            ('\\arcsin{1.3 \\sin{\\theta_w}}', 'Answer: \\theta_a = \\arcsin(1.3 \\sin(\\theta_w))'),
                            ('\\arcsin{1.3 \\sin{\\theta_w}}', 'Answer: \\arcsin(1.3 \\sin\\theta_w)'),
                            ('\\frac{d x}{d t}=k x-a', 'Answer: \\frac{dx}{dt} = kx - a'),
                            ('\\frac{2}{3} \\frac{C}{S}', 'Answer: \\frac{2C}{3S}'), ('E_{1},E_{2}', 'Answer: E_1, E_2'),
                            ('MR=SRMC', 'Answer: SRMC = MR'), ('1.6', 'Answer: 1.6 cm'),
                            ('np.arcsin(10/13)', 'Answer: \\sin^{-1}\\left(\\frac{1}{1.3}\\right)'),
                            ('E_{1},E_{2}', 'Answer: E₁ and E₂'), ('1+\\sqrt{3} i', 'Answer: 1 + i\\sqrt{3}'),
                            ('\\frac{v_{0}}{r_{0}} \\frac{1}{\\left(1+r / r_{0}\\right)^{3 / 2}}',
                             'Answer: \\omega(r) = \\frac{v_0}{r_0 (1 + \\frac{r}{r_0})^{3/2}}'),
                            ('3', 'Answer: \\log_{\\frac{3}{2}}\\left(\\frac{27}{8}\\right)')]:
            with self.subTest(truth=truth, text=text):
                self.assertEqual(self.status(truth, text, 'Minerva'), 'correct')

    def test_no_false_positives(self):
        # Wrong answers score 0 either as incorrect or, for multi-letter products the extractor
        # rejects ("mn", "MR"), as a format failure; a malformed answer can only become correct.
        value = lambda truth, text, task='Olympiad-Bench': evaluation_value(answer_forms.score(text, truth, task=task))
        for truth, text in [('$(6,3),(9,3),(9,5),(54,5)$', 'Answer: (9, 3), (6, 3), (9, 5)'),
                            ('$(3,0),(-3,1)$', 'Answer: (0, 3), (-3, 1)'), ('$f(x)=x+1$', 'Answer: f(x) = x - 1'),
                            ('$f(x)=x+1$', 'Answer: g(x) = x + 1'), ('$k=1$', 'Answer: k = 2'),
                            ('$m n-\\lfloor m / 2\\rfloor$', 'Answer: mn - \\lfloor m \\rfloor / 2'),
                            ('$\\lceil n / 2\\rceil+1$', 'Answer: \\frac{n}{2} + 1'), ('$\\frac{N}{2}$', 'Answer: N/3'),
                            ('$2RC+R+C$', 'Answer: 2RC + R'), ('$f(x)=-1$,$f(x)=x+1$', 'Answer: f(x) = -1')]:
            with self.subTest(truth=truth, text=text):
                self.assertEqual(value(truth, text), 0)
        self.assertEqual(value('MR=SRMC', 'Answer: MR = MC', 'Minerva'), 0)
        self.assertEqual(self.status('$k=1$', 'Answer: k = 2'), 'incorrect')
        # Tiny magnitudes compare relatively; I(0) is a value, not I*0.
        for truth, text in [('4.6e-24', 'Answer: \\lambda = 4.57 \\times 10^{-27} \\, \\mathrm{m}'),
                            ('2.88e-19', 'Answer: approximately **2.91 × 10⁻¹⁹ J**'),
                            ('I(0)', 'Answer: $ I(0) e^{-\\frac{t}{RC}} $')]:
            with self.subTest(truth=truth, text=text):
                self.assertEqual(value(truth, text, 'Minerva'), 0)
        # On a supported truth an undecided answer is never turned into a wrong one.
        self.assertEqual(self.status('36', 'Answer: S + 1'), 'unsupported')

    def test_prefix_and_connectors_on_supported_truths(self):
        self.assertEqual(self.status('6', 'Answer: x = 6'), 'correct')
        self.assertEqual(self.status('5,1', 'Answer: 5 and 1'), 'correct')
        self.assertEqual(self.status('1', 'Answer: 1 or -4'), 'incorrect')
        self.assertEqual(self.status('6', 'Answer: x = 7'), 'incorrect')

    def test_unsupported_forms_stay_unresolved(self):
        for truth, text in [('$k \\geqslant 2$', 'Answer: k \\ge 2'),
                            ('-\\sin (3 t) / 5+ C_{1} \\sin (2 t)+C_{2} \\cos (2 t)', 'Answer: -\\sin(3t)/5 + A\\sin(2t) + B\\cos(2t)'),
                            ('m \\frac{d v}{d t}+b v=f', 'Answer: m \\frac{dv}{dt} + bv = f')]:
            with self.subTest(truth=truth):
                self.assertEqual(self.status(truth, text), 'gt_unparseable')
        self.assertEqual(answer_forms.score('No answer here', '$k=1$', task='Olympiad-Bench')['grade_status'], 'not_attempted')

    def test_rescore_opt_in(self):
        before = math_identity()['fingerprint']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw.jsonl'
            rows = [{'response': 'Answer: f(x) = 2x', 'ground_truth': '$f(x)=2 x$', 'global_index': 0, 'rollout_index': 0},
                    {'response': 'Answer: 3x', 'ground_truth': '$f(x)=2 x$', 'global_index': 0, 'rollout_index': 1}]
            raw.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            legacy = run(raw, root / 'legacy', expected_questions=[0], n=2, task='Olympiad-Bench')
            self.assertEqual(legacy['unresolved_samples'], 2)
            config = {'answer_forms_fingerprint': answer_forms.identity()['fingerprint']}
            metrics = run(raw, root / 'forms', config=config, expected_questions=[0], n=2, task='Olympiad-Bench')
            self.assertEqual((metrics['confirmed_correct'], metrics['confirmed_incorrect']), (1, 1))
            self.assertEqual(metrics['answer_forms']['version'], answer_forms.VERSION)
            with self.assertRaisesRegex(ValueError, 'answer-form fingerprint mismatch'):
                run(raw, root / 'stale', config={'answer_forms_fingerprint': 'stale'}, task='Olympiad-Bench')
        self.assertEqual(before, math_identity()['fingerprint'])


class SummarizeRunsTests(unittest.TestCase):
    def test_profiles_with_same_length_merge_and_newest_wins(self):
        from evaluate import summarize_runs
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / 'run_a'
            entries = [('core_8k', 'AIME24', {'sample_accuracy': .1}), ('ext_8k', 'GPQA', {'sample_accuracy': .3}),
                       ('ext_8k', 'Minerva', {'sample_accuracy_lower': .2, 'unresolved_samples': 3}),
                       ('ext_af1_8k', 'Minerva', {'sample_accuracy': .25}), ('core_16k', 'AIME24', {'sample_accuracy': .5})]
            for profile, task, metrics in entries:
                path = run / 'evaluation' / profile / 'gad' / task / 'scores' / 'metrics.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(metrics))
                (run / 'evaluation' / profile / 'plan.json').write_text(
                    json.dumps({'protocol': {'max_tokens': 16384 if '16k' in profile else 8192}}))
            rows, notes = summarize_runs.collect([run])
            tasks, table = summarize_runs.summarize(rows)
            short = {e['max_tokens']: e for e in table}
            self.assertAlmostEqual(short[8192]['Avg'], (.1 + .3 + .25) / 3)
            self.assertEqual(short[8192]['partial_tasks'], [])
            self.assertEqual(short[16384]['Avg'], .5)
            self.assertEqual(len(notes), 1)


class RawCompleteTests(unittest.TestCase):
    def test_unicode_line_separator_inside_response(self):
        from evaluate.evaluate import raw_complete
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw_samples.jsonl'
            rows = [{'global_index': q, 'rollout_index': r, 'response': 'a\u2028b\u2029c'} for q in (0, 1) for r in (0, 1)]
            path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
            self.assertTrue(raw_complete(path, [0, 1], 2))
            self.assertFalse(raw_complete(path, [0, 1, 2], 2))


class DatasetTests(unittest.TestCase):
    def test_dapo_version_rewrites_only_instruction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            questions = {'Minerva': 'Find x (in cm).', 'Olympiad-Bench': '\nProve y.',
                         'GPQA': 'Which?\n\na) one\nb) two\n\nA. b\nB. a\nC. none\nD. both',
                         'MMLU-Pro-Math': 'Pick.\nAnswer Choices: (A) 1  (B) $2 (C) (I) and (II)'}
            for task, question in questions.items():
                path = root / 'src' / task / 'test.parquet'
                path.parent.mkdir(parents=True)
                extra = {'index': f'{task}-0', 'option_num': 4 if task == 'GPQA' else 3}
                pq.write_table(pa.Table.from_pylist([{'prompt': [{'content': question + BOXED_SUFFIX, 'role': 'user'}],
                    'reward_model': {'ground_truth': 'B'}, 'extra_info': extra}]), path)
            manifest = prepare_dapo(root / 'src', root / 'out')
            rows = {task: pq.read_table(root / 'out' / task / 'test.parquet').to_pylist()[0] for task in questions}
            self.assertEqual(rows['Olympiad-Bench']['prompt'], [{'content': DAPO_PREFIX + 'Prove y.', 'role': 'user'}])
            # The letter instruction is appended to choice questions only.
            instruction = CHOICE_INSTRUCTIONS['extended_dapo_v2']
            self.assertEqual(rows['GPQA']['prompt'][0]['content'], DAPO_PREFIX + questions['GPQA'] + '\n\n' + instruction)
            self.assertNotIn(instruction, rows['Minerva']['prompt'][0]['content'])
            prepare_dapo(root / 'src', root / 'v1', version='extended_dapo_v1')
            v1 = pq.read_table(root / 'v1/GPQA/test.parquet').to_pylist()[0]
            self.assertEqual(v1['prompt'][0]['content'], DAPO_PREFIX + questions['GPQA'])
            self.assertEqual(rows['Minerva']['reward_model'], {'ground_truth': 'B'})
            self.assertEqual(rows['GPQA']['extra_info']['options'], ['b', 'a', 'none', 'both'])
            self.assertTrue(rows['GPQA']['extra_info']['lowercase_labels_in_stem'])
            self.assertEqual(rows['MMLU-Pro-Math']['extra_info']['options'], ['1', '$2', '(I) and (II)'])
            self.assertEqual(manifest['tasks']['GPQA']['scorer'], choice.VERSION)
            with self.assertRaises(FileExistsError):
                prepare_dapo(root / 'src', root / 'out')

    def test_math_subset_keeps_prompt_answer_and_source_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['Minerva', 'Olympiad-Bench', 'GPQA', 'MMLU-Pro']:
                path = root / 'source' / name / 'test.parquet'
                path.parent.mkdir(parents=True)
                rows = [{'prompt': [{'role': 'user', 'content': f'question {i}'}],
                         'reward_model': {'ground_truth': 'B'}, 'extra_info': {'index': f'{name}-{i}'},
                         'data_source': name, 'option_num': 5, 'category': category}
                        for i, category in enumerate(['law', 'math', 'math'])]
                pq.write_table(pa.Table.from_pylist(rows), path)
            manifest = prepare(root / 'source', root / 'out')
            source = pq.read_table(root / 'source/MMLU-Pro/test.parquet').to_pylist()
            selected = pq.read_table(root / 'out/MMLU-Pro-Math/test.parquet').to_pylist()
            self.assertEqual(manifest['tasks']['MMLU-Pro-Math']['source_row_indices'], [1, 2])
            for original, output in zip(source[1:], selected):
                self.assertEqual(original['prompt'], output['prompt'])
                self.assertEqual(original['reward_model'], output['reward_model'])
                self.assertEqual(original['extra_info']['index'], output['extra_info']['index'])
                self.assertEqual(output['extra_info']['option_num'], 5)
            with self.assertRaises(FileExistsError):
                prepare(root / 'source', root / 'out')


if __name__ == '__main__':
    unittest.main()
