import json
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[2];sys.path.insert(0, str(ROOT))
from scoring.math import score, correctness_label, compute_score, validate_ground_truth
from evaluate.rescore import summarize
from rubric.core.process_v2 import parse_events
from rubric.core.contracts import V2_SCHEMA_VERSION


def event_record(response='2x=4', progress=.5, error=0):
    return {'schema_version':V2_SCHEMA_VERSION,'rubric_version':'math_process_v2','completion':'prefix',
        'analysis_complete':True,'events':{'e1':{'type':'verified_progress','supporting_span_ranges':{'r1': {'start': 's1', 'end': 's1'}},
        'trajectory_claim':'valid independent algebra', 'judge_verification': 'valid independent algebra','related_event_ids':[]}},
        'verified_progress':{'value':progress,'valid':True,'reason':'partial valid progress','event_ids':['e1']},
        'unresolved_error_severity':{'value':error,'valid':True,'reason':'checked visible prefix','event_ids':[]}}


class ExtractionTests(unittest.TestCase):
    def test_explicit_candidates(self):
        for response in ('Answer: 42','Final Answer:\nAnswer: 42','Answer:\n\n42','Answer:\r\n42',
                         '**Final Answer:** $42$', '# Answer:\n42','\\boxed{42}',
                         '\\boxed{12}\nAnswer: 42','Answer: \\boxed{42}', 'Answer: 42\n'+'work '*100):
            with self.subTest(response=response):self.assertTrue(score(response,'42')['correct'])
        for response in ('Answer: 42\nFinal Answer:', 'Answer: 42\n\\boxed{', 'Answer: \\boxed{1} and \\boxed{2}'):
            self.assertIn(score(response,'42')['parse_status'], ('malformed','ambiguous'))
        self.assertEqual('no_answer',score('The intermediate number is 42.','42')['parse_status'])
        self.assertFalse(score('Answer: 41','42')['correct'])

    def test_math_structure(self):
        for pred,gt in [('142.0','142'),('0.5','1/2'),('\\frac{2}{4}','0.5'),('(x+1)^2','x^2+2*x+1'),
                        ('{1,2}','{2,1}'),('42 cm','42.0 cm')]:self.assertTrue(score('Answer: '+pred,gt)['correct'],(pred,gt))
        for pred,gt in [('0.3333','1/3'),('(1,2)','(2,1)'),('(0,1]','[0,1]'),('{1,2}','1'),
                        ('42 cm','42 m'),('42 cm','42'),('x=2','2')]:self.assertFalse(score('Answer: '+pred,gt)['correct'],(pred,gt))

    def test_consumer_status_and_errors(self):
        for response in ('Answer: 2','Answer: 3','no final','Answer: \\boxed{'):
            result=score(response,'2');adapter=compute_score(response,'2')
            for key in result:self.assertEqual(result[key],adapter[key])
        self.assertEqual(0,correctness_label(score('no final','2'))['value'])
        self.assertFalse(correctness_label(score('Answer: \\boxed{','2'))['valid'])
        self.assertFalse(correctness_label(score('Answer: 2','2'),True)['valid'])
        for exception,status in ((TimeoutError('timeout'),'timeout'),(RuntimeError('tool failure'),'error')):
            with patch('scoring.math.grade_pair',side_effect=exception):
                result=score('Answer: 2','2');self.assertEqual(status,result['grade_status']);self.assertIsNone(result['correct'])
                self.assertFalse(correctness_label(result)['valid'])
        for gt in (None,'',False,'nan'):
            with self.assertRaises(ValueError):validate_ground_truth(gt)


class MetricTests(unittest.TestCase):
    def rows(self):
        return [{**score('Answer: '+answer,'2'),'global_index':q,'rollout_index':r}
                for q,r,answer in [(0,0,'2'),(0,1,'3'),(1,0,'2'),(1,1,'2')]]
    def test_denominators(self):
        result=summarize(self.rows(),expected_questions=[0,1],n=2)
        self.assertTrue(result['evaluation_complete']);self.assertEqual(.75,result['pass_at_1']);self.assertEqual(.75,result['sample_accuracy'])
        self.assertEqual(1,result['first_rollout_accuracy']);self.assertEqual(1,result['any_success_at_n'])
        for rows in (self.rows()[:-1],self.rows()+[self.rows()[0]]):
            result=summarize(rows,expected_questions=[0,1],n=2);self.assertFalse(result['evaluation_complete']);self.assertNotIn('pass_at_1',result)
        rows=self.rows();rows[0].update(grade_status='timeout',correct=None)
        self.assertFalse(summarize(rows,expected_questions=[0,1],n=2)['evaluation_complete'])


class EventTests(unittest.TestCase):
    def test_prefix_and_partial_tail(self):
        row=event_record();parsed=parse_events(json.dumps(row),'2x=4\nx=4/')
        self.assertEqual(.5,parsed['verified_progress']['value']);self.assertEqual(0,parsed['unresolved_error_severity']['value'])
        row['events']['e2']={'type':'unjudgeable','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}},
            'trajectory_claim':'incomplete formula', 'judge_verification': 'incomplete formula','related_event_ids':[]}
        self.assertEqual(.5,parse_events(json.dumps(row),'2x=4\nx=4/')['verified_progress']['value'])
    def test_invalid_evidence_and_repair(self):
        row=event_record();row['events']['e1']['supporting_span_ranges']={'r1': {'start': 's999', 'end': 's999'}}
        self.assertFalse(parse_events(json.dumps(row),'2x=4')['verified_progress']['valid'])
        row=event_record();row['events']['e2']={'type':'explicit_error','supporting_span_ranges':{'r1': {'start': 's2', 'end': 's2'}},
            'trajectory_claim':'wrong division', 'judge_verification': 'wrong division','related_event_ids':[]}
        guarded=parse_events(json.dumps(row),'2x=4\nx=3')
        self.assertFalse(guarded['unresolved_error_severity']['valid']);self.assertTrue(guarded['verified_progress']['valid'])
        row['unresolved_error_severity'].update(value=1,event_ids=['e2'])
        self.assertEqual(1,parse_events(json.dumps(row),'2x=4\nx=3')['unresolved_error_severity']['value'])
        row['events']['e3']={'type':'corrected_error','supporting_span_ranges':{'r1': {'start': 's3', 'end': 's3'}},
            'trajectory_claim':'correct replacement', 'judge_verification': 'correct replacement','error_event_id':'e2'}
        row['unresolved_error_severity'].update(value=0,event_ids=['e3'])
        self.assertEqual(0,parse_events(json.dumps(row),'2x=4\nx=3\nreplace x=3 with x=2')['unresolved_error_severity']['value'])
        row['verified_progress'].update(value=1);row['completion']='prefix'
        self.assertFalse(parse_events(json.dumps(row),'2x=4\nx=3\nreplace x=3 with x=2')['verified_progress']['valid'])


if __name__=='__main__':unittest.main()
