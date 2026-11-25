"""Fixed evaluation slots and interruption recovery without sampling a model."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from evaluate.rescore import summarize,run
from evaluate.evaluate import execute, build_plan, training_active
from scoring.math import score,DEFAULT_CONFIG,identity


class EvaluationTests(unittest.TestCase):
    def test_training_active_matches_output_run_not_name_prefix_or_model_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / 'runs/baseline_06b_17k'
            other = root / 'runs/baseline_06b_17kcnt'
            process = root / '123'
            process.mkdir()
            trainer = ['python', '-m', 'verl.trainer.main_ppo']
            cases = [
                (trainer + [f'trainer.default_local_dir={run / stage}'], True)
                for stage in ('seqkd', 'warmup', 'gad')
            ] + [
                (trainer + [f'trainer.default_local_dir={other / "gad"}'], False),
                (trainer + [f'trainer.default_local_dir={other / "gad"}',
                            f'actor_rollout_ref.model.path={run / "warmup/global_step_271/actor"}'], False),
                (['bash', '-c', f'python -m verl.trainer.main_ppo trainer.default_local_dir={run / "gad"}'], False),
                (['python', 'evaluate.py', '--run', str(run)], False),
            ]
            for argv, expected in cases:
                with self.subTest(argv=argv):
                    (process / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
                    with patch('evaluate.evaluate.Path.iterdir', return_value=iter([process])):
                        self.assertEqual(training_active(run), expected)

    def test_training_active_resolves_relative_output_from_process_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            workspace = root / 'workspace'
            run = workspace / 'runs/baseline_06b_17kcnt'
            (run / 'gad').mkdir(parents=True)
            process = root / '123'
            process.mkdir()
            (process / 'cwd').symlink_to(workspace, target_is_directory=True)
            argv = ['python', '-m', 'verl.trainer.main_ppo',
                    'trainer.default_local_dir=runs/baseline_06b_17kcnt/gad']
            (process / 'cmdline').write_bytes(('\0'.join(argv) + '\0').encode())
            with patch('evaluate.evaluate.Path.iterdir', return_value=iter([process])):
                self.assertTrue(training_active(run))

    def rows(self):
        return [{**score('Answer: 2','2'),'global_index':q,'rollout_index':r} for q in (10,40) for r in (0,1)]

    def test_actual_expected_ids_and_anomalies_use_fixed_slots(self):
        good=self.rows();result=summarize(good,expected_questions=[10,40],n=2)
        self.assertTrue(result['evaluation_complete']);self.assertEqual(1,result['pass_at_1'])
        variants=[good+[dict(good[0],rollout_index=v)] for v in (-1,2,1.5,True,'1')]
        variants += [good+[dict(good[0],global_index=99)],good+[good[0]],good[:-1],[],
            good+[dict(good[0],correct=False,grade_status='incorrect')]]
        for rows in variants:
            result=summarize(rows,expected_questions=[10,40],n=2)
            self.assertFalse(result['evaluation_complete'])
            for value in result['provisional_metrics'].values():
                if value is not None:self.assertTrue(0<=value<=1)
        empty=summarize([],expected_questions=[],n=2)
        self.assertFalse(empty['evaluation_complete']);self.assertIsNone(empty['provisional_metrics']['sample_accuracy'])

    def plan(self,root):
        return {'run_dir':str(root/'run'),'output_root':str(root/'eval'),'oprd_root':str(root),'python_bin':sys.executable,
            'models':{'gad':{'path':str(root/'model'),'target_step':16,'selection':'configured_final','content':{'config':'fixture'},
                'checkpoint_ready':True,'ready':False,'needs_export':True}},
            'tasks':{task:{'path':str(root/(task+'.parquet')),'rows':1,'sha256':task,'expected_question_ids':[uid]} for task,uid in [('A',10),('B',40)]},
            'max_prompts':0,'gpu_ids':'1','protocol':{'n':1,'math500_n':1,'temperature':1.,'top_p':.95,'top_k':-1,
                'repetition_penalty':1.,'max_tokens':8,'max_model_len':32,'gpu_memory_utilization':.5,'seed':42,'enforce_eager':True},
            'scoring_math':dict(DEFAULT_CONFIG),'scorer_identity':identity()}

    def test_export_idempotence_task_resume_and_raw_only_scoring(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.plan(root);calls=[];interrupt=[True]
            def export(plan,stage,model):
                directory=Path(model['path']);directory.mkdir(exist_ok=True)
                (directory/'model.safetensors').write_bytes(b'fake-HF-weights')
                (directory/'config.json').write_text('{}')
                (directory/'tokenizer.json').write_text('{}')
                (directory/'chat_template.jinja').write_text('fixture template')
                model.update(ready=True,needs_export=False,export_manifest='derived-export')
            def subprocess_run(argv,**kwargs):
                if '-m' in argv:
                    task=Path(argv[argv.index('--input-parquet')+1]).stem;calls.append(('generate',task))
                    if task=='B' and interrupt[0]:raise RuntimeError('interrupted generator')
                    directory=Path(argv[argv.index('--output-dir')+1]);directory.mkdir(parents=True,exist_ok=True)
                    uid=plan['tasks'][task]['expected_question_ids'][0]
                    (directory/'raw_samples.jsonl').write_text(json.dumps({'global_index':uid,'rollout_index':0,'response':'Answer: 2','ground_truth':'2'})+'\n')
                else:
                    task=argv[argv.index('--task')+1];calls.append(('score',task))
                    run(argv[argv.index('--raw-jsonl')+1],argv[argv.index('--output-dir')+1],
                        json.loads(argv[argv.index('--config-json')+1]),json.loads(Path(argv[argv.index('--expected-ids')+1]).read_text()),1,task)
            with patch('evaluate.evaluate.training_active',return_value=False),patch('evaluate.evaluate.export_stage',export),patch('evaluate.evaluate.subprocess.run',subprocess_run):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):execute(plan)
                interrupt[0]=False;calls.clear();execute(deepcopy(plan))
                self.assertEqual([('generate','B'),('score','B')],calls)
                calls.clear();execute(deepcopy(plan));self.assertEqual([],calls)
                state=root/'eval/gad/B/state.json';state.unlink()
                metrics=root/'eval/gad/B/scores/metrics.json';before=metrics.read_bytes()
                calls.clear();execute(deepcopy(plan));self.assertEqual([('score','B')],calls)
                self.assertEqual(before,metrics.read_bytes())
                changed=deepcopy(plan);changed['protocol']['seed']=43
                with self.assertRaisesRegex(RuntimeError,'settings changed'):execute(changed)
                original=(root/'eval/plan.json').read_bytes()
                for name in ('model.safetensors','config.json','tokenizer.json','chat_template.jinja'):
                    path=root/'model'/name;data=path.read_bytes();path.write_bytes(data+b'changed');calls.clear()
                    with self.assertRaisesRegex(RuntimeError,'HF content changed'):execute(deepcopy(plan))
                    self.assertEqual([],calls);self.assertEqual(original,(root/'eval/plan.json').read_bytes());path.write_bytes(data)
                from evaluate.evaluate import model_content
                nested=root/'model/nested';nested.mkdir();(nested/'shard.safetensors').write_bytes(b'shard')
                self.assertIn('nested/shard.safetensors',model_content(root/'model'))
                with self.assertRaisesRegex(RuntimeError,'HF content changed'):execute(deepcopy(plan))

    def test_new_scorer_plan_preserves_frozen_training_identity(self):
        from config import Config, DEFAULTS
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frozen=root/'train.yaml'
            frozen.write_text('scoring:\n  math:\n    fingerprint: old-training-scorer\n')
            before=frozen.read_bytes()
            cfg=Config({**DEFAULTS,'run.output_root':str(root),'run.run_id':'run',
                'models.student':str(root/'model'),'run.python_bin':sys.executable,
                'eval.tensor_parallel_size':1,
                'data.train_file':str(ROOT/'gad/tests/fixtures/quick.parquet'),'data.train_batch_size':1,
                'rollout.enable_thinking':False,'rollout.thinking_control':True,
                'scoring.math.version':'math_unified_v1','scoring.math.fingerprint':'old-training-scorer',
                'scoring.math.entrypoint':str(ROOT/'scoring/math.py')},path=frozen)
            original=deepcopy(cfg.data)
            (root/'oprd/sft').mkdir(parents=True)
            (root/'oprd/sft/sample_teacher.py').write_text('')
            dataset=root/'test.parquet'
            pq.write_table(pa.Table.from_pylist([{'prompt':[{'role':'user','content':'1+1?'}],
                'reward_model':{'ground_truth':'2'},'extra_info':{'index':10}}]),dataset)
            kwargs=dict(methods=['base'],tasks=['A'],gpu_ids='0',max_prompts=0,
                oprd_root=root/'oprd',dataset_root=root,profile='new-eval',seed=42,top_k=-1,
                repetition_penalty=1.,enforce_eager=True,resolved_config=cfg,
                dataset_overrides={'A':dataset})
            with self.assertRaisesRegex(ValueError,'evaluation scorer differs'):
                build_plan(frozen,**kwargs)
            with self.assertRaisesRegex(ValueError,'requires --allow-new-eval-scorer'):
                build_plan(frozen,**kwargs,generate_missing=True)
            with self.assertRaisesRegex(ValueError,'cannot be combined'):
                build_plan(frozen,**kwargs,allow_new_eval_scorer=True,generate_missing=True,
                           reuse_generation_from='old-profile')
            with patch('evaluate.evaluate.sampler_question_ids',return_value=[10]):
                plan=build_plan(frozen,**kwargs,allow_new_eval_scorer=True,generate_missing=True)
            self.assertEqual('old-training-scorer',plan['training_scorer']['fingerprint'])
            self.assertEqual(identity()['fingerprint'],plan['evaluation_scorer']['fingerprint'])
            self.assertEqual(identity()['fingerprint'],plan['scoring_math']['fingerprint'])
            self.assertTrue(plan['generate_missing'])
            self.assertEqual(original,cfg.data);self.assertEqual(before,frozen.read_bytes())
            self.assertFalse((root/'run').exists())

    def ready_new_scorer_plan(self,root):
        plan=self.plan(root);plan['tasks'].pop('B')
        model=root/'model';model.mkdir()
        (model/'model.safetensors').write_bytes(b'fixture-weights')
        (model/'config.json').write_text('{}')
        plan['models']['gad'].update(ready=True,needs_export=False)
        plan.update(allow_new_eval_scorer=True,generate_missing=False,
            training_scorer={'fingerprint':'old-training-scorer'},
            evaluation_scorer={'fingerprint':identity()['fingerprint']})
        return plan

    def test_new_scorer_first_evaluation_and_resume_without_regeneration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.ready_new_scorer_plan(root);calls=[]
            with patch('evaluate.evaluate.training_active',return_value=False),patch('evaluate.evaluate.subprocess.run') as command:
                with self.assertRaisesRegex(RuntimeError,'--generate-missing'):execute(deepcopy(plan))
                command.assert_not_called()
            plan['generate_missing']=True
            def command(argv,**kwargs):
                if '-m' in argv:
                    calls.append('generate')
                    target=Path(argv[argv.index('--output-dir')+1]);target.mkdir(parents=True)
                    (target/'raw_samples.jsonl').write_text(json.dumps({'global_index':10,'rollout_index':0,
                        'response':'Answer: 2','ground_truth':'2'})+'\n')
                else:
                    calls.append('score')
                    run(argv[argv.index('--raw-jsonl')+1],argv[argv.index('--output-dir')+1],
                        json.loads(argv[argv.index('--config-json')+1]),[10],1,'A')
            with patch('evaluate.evaluate.training_active',return_value=False),patch('evaluate.evaluate.subprocess.run',command):
                execute(deepcopy(plan));self.assertEqual(['generate','score'],calls)
                manifest=json.loads((root/'eval/scoring_manifest.json').read_text())
                self.assertEqual('old-training-scorer',manifest['training_scorer']['fingerprint'])
                self.assertEqual(identity()['fingerprint'],manifest['evaluation_scorer']['fingerprint'])
                self.assertTrue(manifest['generate_missing'])
                calls.clear();execute(deepcopy(plan));self.assertEqual([],calls)
                # A missing score is repaired from the same raw, without sampling again.
                state=root/'eval/gad/A/state.json';state.unlink()
                raw=root/'eval/gad/A/generation/raw_samples.jsonl';before=raw.read_bytes()
                plan['generate_missing']=False
                execute(deepcopy(plan));self.assertEqual(['score'],calls)
                self.assertEqual(before,raw.read_bytes())

    def test_reuse_old_raw_with_new_scorer_remains_read_only(self):
        from evaluate.evaluate import model_content, task_state_identity, verified_source_raw
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);plan=self.ready_new_scorer_plan(root)
            source=root/'old/gad/A/generation/raw_samples.jsonl';source.parent.mkdir(parents=True)
            source.write_text(json.dumps({'global_index':10,'rollout_index':0,
                'response':'Answer: 2','ground_truth':'2'})+'\n')
            old = deepcopy(plan)
            old['models']['gad']['hf_content'] = model_content(root/'model')
            old['tasks']['A']['evaluation_scorer'] = {'fingerprint':'historical-grader'}
            old['protocol']['task_n'] = {'B': 8}
            (root/'old/plan.json').write_text(json.dumps(old))
            source_state = source.parent.parent/'state.json'
            source_state.write_text(json.dumps({'identity':task_state_identity(old, 'gad', 'A'),
                                                'raw_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}))
            before=source.read_bytes();plan['reuse_generation_from']=str(root/'old');calls=[]
            def command(argv,**kwargs):
                self.assertNotIn('-m',argv);calls.append('score')
                self.assertEqual(str(source),argv[argv.index('--raw-jsonl')+1])
                run(str(source),argv[argv.index('--output-dir')+1],
                    json.loads(argv[argv.index('--config-json')+1]),[10],1,'A')
            with patch('evaluate.evaluate.training_active',return_value=False),patch('evaluate.evaluate.subprocess.run',command):
                execute(plan)
            self.assertEqual(['score'],calls);self.assertEqual(before,source.read_bytes())
            self.assertFalse((root/'eval/gad/A/generation').exists())
            changes = [(('protocol','temperature'), .5), (('protocol','task_n'), {'A':2}),
                       (('tasks','A','expected_question_ids'), [40]), (('tasks','A','sha256'), 'other-prompts'),
                       (('models','gad','hf_content'), {'model.safetensors':'other-model'}),
                       (('models','gad','hf_content'), {**model_content(root/'model'),'chat_template.jinja':'other'})]
            for keys, value in changes:
                with self.subTest(keys=keys):
                    changed = deepcopy(old)
                    field = changed
                    for key in keys[:-1]:
                        field = field[key]
                    field[keys[-1]] = value
                    with self.assertRaisesRegex(ValueError, 'source generation differs'):
                        verified_source_raw(plan, changed, 'gad', 'A')
            source.write_bytes(before.replace(b'Answer: 2', b'Answer: 7'))
            with self.assertRaisesRegex(ValueError, 'source state'):
                verified_source_raw(plan, old, 'gad', 'A')
            source.write_bytes(before)
            source_state.unlink()
            with self.assertRaisesRegex(ValueError, 'provenance'):
                verified_source_raw(plan, old, 'gad', 'A')
            # Sampling finished, but the scoring state was never written.
            config = {k:v for k,v in old['protocol'].items() if k not in ('n','math500_n','task_n','max_tokens')}
            config.update(max_new_tokens=8, num_rollouts=1, input_sha256='A',
                          model_config_sha256=old['models']['gad']['hf_content']['config.json'],
                          model_path=str(root/'model'), gpu_ids=[1],
                          basic_rejection=False, regenerate_until_correct=False)
            manifest = {'raw_samples_sha256': hashlib.sha256(before).hexdigest(), 'config': config}
            manifest_path = source.with_name('manifest.json')
            manifest_path.write_text(json.dumps(manifest))
            self.assertEqual((source, manifest['raw_samples_sha256']), verified_source_raw(plan, old, 'gad', 'A'))
            config['seed'] = 11
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'sampler manifest'):
                verified_source_raw(plan, old, 'gad', 'A')

    def test_reuse_published_interval_and_retry_transient_grade_failures(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            plan = self.ready_new_scorer_plan(root)
            task = root / 'eval/gad/A'
            raw = task / 'generation/raw_samples.jsonl'
            raw.parent.mkdir(parents=True)
            original = json.dumps({'global_index':10, 'rollout_index':0,
                                   'response':'Answer: 2', 'ground_truth':None})
            raw.write_text(original)
            execute(plan)
            state_path = task / 'state.json'
            state = json.loads(state_path.read_bytes())
            path = Path(state['metrics_path'])
            metrics = json.loads(path.read_bytes())
            self.assertEqual('incomplete', state['status'])
            self.assertEqual({'gt_unparseable':1}, metrics['unresolved_reason_counts'])
            self.assertEqual((0., 1.), (metrics['sample_accuracy_lower'], metrics['sample_accuracy_upper']))
            with patch('evaluate.evaluate.subprocess.run') as command:
                execute(plan)
                command.assert_not_called()

            # Reproduce a legacy interval whose input digest is only in state.
            metrics.pop('raw_sha256')
            metrics['ground_truth_audit'].pop('inspection_errors')
            path.write_text(json.dumps(metrics))
            state['metrics_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            state_path.write_text(json.dumps(state))
            with patch('evaluate.evaluate.subprocess.run') as command:
                execute(plan)
                command.assert_not_called()

            for changes in ({'unresolved_reason_counts': {'timeout': 1}, 'grader_error_count': 1},
                            {'unresolved_reason_counts': {'internal_error': 1}},
                            {'ground_truth_audit': {'inspection_errors': 1}},
                            {'sample_index_complete': False}):
                with self.subTest(changes=changes):
                    path.write_text(json.dumps({**metrics, **changes}))
                    state['metrics_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                    state_path.write_text(json.dumps(state))
                    execute(plan)
                    retried = json.loads(state_path.read_bytes())
                    self.assertNotEqual(str(path), retried['metrics_path'])
                    self.assertEqual('incomplete', retried['status'])
            path = Path(retried['metrics_path'])
            path.with_name('scored_samples.jsonl').unlink()
            execute(plan)
            repaired = Path(json.loads(state_path.read_bytes())['metrics_path'])
            self.assertTrue(repaired.with_name('scored_samples.jsonl').is_file())
            self.assertNotEqual(repaired, path)
            self.assertEqual(original, raw.read_text())

    def test_failed_ground_truth_inspection_is_recorded_for_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            raw = root / 'raw.jsonl'
            raw.write_text(json.dumps({'global_index':10, 'rollout_index':0,
                                       'response':'Answer: 2', 'ground_truth':'2'}))
            with patch('evaluate.rescore.ground_truth_semantics', side_effect=TimeoutError('inspector')):
                metrics = run(raw, root/'scores', expected_questions=[10], n=1)
            self.assertEqual({'gt_unparseable': 1}, metrics['unresolved_reason_counts'])
            self.assertEqual(1, metrics['ground_truth_audit']['inspection_errors'])

    def test_scoring_binds_consumed_bytes_and_checks_raw_before_publishing(self):
        for change in ('after_scoring', 'during_read_then_restored'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                plan = self.ready_new_scorer_plan(root)
                task = root / 'eval/gad/A'
                raw = task / 'generation/raw_samples.jsonl'
                raw.parent.mkdir(parents=True)
                original = b'{"global_index":10,"rollout_index":0,"response":"Answer: 2","ground_truth":"2"}\r\n'
                changed = original.replace(b'Answer: 2', b'Answer: 7')
                raw.write_bytes(original)

                def score_and_change(argv, **kwargs):
                    if change == 'during_read_then_restored':
                        raw.write_bytes(changed)
                    run(raw, argv[argv.index('--output-dir') + 1], expected_questions=[10], n=1, task='A')
                    raw.write_bytes(changed if change == 'after_scoring' else original)

                with patch('evaluate.evaluate.subprocess.run', side_effect=score_and_change):
                    with self.assertRaisesRegex(RuntimeError, 'changed during scoring'):
                        execute(plan)
                self.assertEqual(json.loads((task / 'state.json').read_text())['status'], 'scoring')
                self.assertFalse(list(task.glob('scores*/metrics.json')))

                execute(plan)
                state = json.loads((task / 'state.json').read_text())
                metrics = json.loads(Path(state['metrics_path']).read_text())
                self.assertEqual(state['status'], 'complete')
                self.assertEqual(metrics['confirmed_correct'], int(change == 'during_read_then_restored'))
                self.assertEqual(metrics['raw_sha256'], hashlib.sha256(raw.read_bytes()).hexdigest())
                self.assertEqual(state['raw_sha256'], metrics['raw_sha256'])

                # A historical successful run has the raw digest in state only.
                metrics.pop('raw_sha256')
                legacy_metrics = json.dumps(metrics).encode()
                Path(state['metrics_path']).write_bytes(legacy_metrics)
                state['metrics_sha256'] = hashlib.sha256(legacy_metrics).hexdigest()
                (task / 'state.json').write_text(json.dumps(state))
                with patch('evaluate.evaluate.subprocess.run') as command:
                    execute(plan)
                    command.assert_not_called()
                self.assertEqual(Path(state['metrics_path']).read_bytes(), legacy_metrics)


if __name__=='__main__':unittest.main()
