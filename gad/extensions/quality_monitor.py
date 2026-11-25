"""Warnings from cached/logged labels only; no extra judge requests."""
from .common import append_json
from .core.contracts import V2_DIMENSIONS


def monitor(owner, stats, values, step):
    previous = getattr(owner,'quality_monitor_state',{})
    state, warnings = {}, []
    for d in V2_DIMENSIONS[1:]:
        for role in ('teacher','student'):
            key=f'rubric/coverage_{d}_{role}'
            coverage=stats.get(key,0)
            if key in previous and abs(previous[key]-coverage)>=.5:warnings.append({'type':'coverage_shift','dimension':d,'role':role,'before':previous[key],'after':coverage})
            state[key]=coverage
            valid=stats.get(f'rubric/valid_{d}_{role}',0)
            if valid>=4 and stats.get(f'rubric/same_score_{d}_{role}',0):warnings.append({'type':'score_collapse','dimension':d,'role':role,'labels':valid})
        teacher=stats.get(f'rubric/coverage_{d}_teacher',0);student=stats.get(f'rubric/coverage_{d}_student',0)
        stats[f'rubric/teacher_student_coverage_gap_{d}']=teacher-student
        stats[f'rubric/teacher_student_coverage_ratio_{d}']=teacher/student if student else 0.
        stats[f'rubric/teacher_student_coverage_ratio_defined_{d}']=int(student>0)
    # Rule-correctness and process supervision have separate semantics. This
    # counter flags a full-solution contradiction for review, not proof by regex.
    stats['rubric/critical/full_progress_wrong_final']=sum(
        r['raw_scorer_result'].get('correct') is False and r['labels']['verified_progress']['valid'] and r['labels']['verified_progress']['value']==1 for r in values.values())
    stats['rubric/human_verified_process_records']=sum(r.get('semantic_review',{}).get('human_verified',False) for r in values.values())
    stats['rubric/quality_warning_count']=len(warnings)
    for mode in ('legacy_active','fixed_dimension','fixed_dimension_gated'):
        stats['rubric/loss_mode/'+mode]=int(owner.options.get('loss_mode','legacy_active')==mode)
    owner.quality_monitor_state=state
    append_json(owner.directory/'quality_monitor.jsonl',{'step':step,'warnings':warnings,'critical_full_progress_wrong_final':stats['rubric/critical/full_progress_wrong_final'],'semantic_reliability':'not_established','thresholds':'engineering alerts, not semantic acceptance'})
