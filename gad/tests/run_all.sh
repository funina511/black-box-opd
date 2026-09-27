#!/usr/bin/env bash
# Run every CPU verification suite for the GAD pipeline.
#
#   bash LMOps/gad/tests/run_all.sh [PYTHON_BIN]
#
# Suites:
#   1. LMOps/sample/tests/test_data_contract.py     data contract (16 tests)
#   2. LMOps/sample/tests/test_teacher_sampling.py  sampler against a mock endpoint (7 tests)
#   3. LMOps/sample/tests/test_eval_data.py         evaluation-set builder (5 tests)
#   4. LMOps/sample/tests/test_split_no_leak.py     prompt-grouped split + run_sample.sh stages (13 tests)
#   4b LMOps/sample/tests/test_offline_engine.py    in-process vLLM engine + engine fingerprint (21 tests)
#   5. LMOps/gad/tests/test_pipeline_cpu.py         backends, dry-run, handoff, resume, guard, skip, hydra, R5 critic batching, per-stage rollout.n, vLLM KV budget (63 tests)
#   6. LMOps/gad/tests/test_resume_patch.py         same-stage resume patch behaviour (4 tests)
#   7. LMOps/gad/tests/test_algorithm_parity.py     frozen loss/advantage formulas (5 tests)
#   7b LMOps/gad/tests/test_microbatch_equivalence.py  budget cap + accumulation weights (9 tests)
#   8. LMOps/gad/tests/test_math_validation.py      math accuracy in validation (10 tests)
#   9. LMOps/gad/tests/test_handoff_export.py       real DTensor merge, final-model export, tied embeddings (16 tests)
#  10. LMOps/gad/tools/math_metrics.py --self-test  CARD scorer parity fixtures
#
# None of these need a GPU. The GPU smoke test is a separate manual step; see
# LMOps/gad/PIPELINE.md section 4.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${1:-${GAD_PYTHON:-python3}}"

echo "=== repository: ${REPO_ROOT}"
echo "=== python    : ${PYTHON_BIN}"

failures=0

run_suite() {
  local name="$1"; shift
  echo
  echo "=== ${name}"
  if "$@"; then
    echo "=== ${name}: PASSED"
  else
    echo "=== ${name}: FAILED"
    failures=$((failures + 1))
  fi
}

run_suite "data contract" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/sample/tests/test_data_contract.py"
run_suite "teacher sampling" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/sample/tests/test_teacher_sampling.py"
run_suite "eval data" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/sample/tests/test_eval_data.py"
run_suite "split/no-leak + sample cli" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/sample/tests/test_split_no_leak.py"
run_suite "offline engine" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/sample/tests/test_offline_engine.py"
run_suite "pipeline cpu" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tests/test_pipeline_cpu.py"
run_suite "resume patch" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tests/test_resume_patch.py"
run_suite "microbatch equivalence" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tests/test_microbatch_equivalence.py"
run_suite "algorithm parity" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tests/test_algorithm_parity.py"
run_suite "math validation" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tests/test_math_validation.py"
run_suite "handoff export" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tests/test_handoff_export.py"
run_suite "math scorer parity" "${PYTHON_BIN}" "${REPO_ROOT}/LMOps/gad/tools/math_metrics.py" --self-test

echo
if [[ "${failures}" -eq 0 ]]; then
  echo "ALL CPU SUITES PASSED"
else
  echo "${failures} SUITE(S) FAILED"
fi
exit "${failures}"
