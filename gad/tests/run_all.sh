#!/usr/bin/env bash
# All CPU suites use freshly patched, isolated worktrees. No GPU model/training.
# Usage: bash LMOps/gad/tests/run_all.sh [PYTHON_BIN]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LMOPS_ROOT="${REPO_ROOT}/LMOps"
PYTHON_BIN="${1:-${GAD_PYTHON:-python3}}"
TEST_ROOT="$(mktemp -d /tmp/gad-cpu-tests.XXXXXX)"
cleanup() {
  "${PYTHON_BIN}" "${LMOPS_ROOT}/gad/tests/backend_fixture.py" "${TEST_ROOT}" cleanup
  rm -rf "${TEST_ROOT}"
}
trap cleanup EXIT
export GAD_TEST_BACKEND_ROOT="${TEST_ROOT}/original"
export GAD_TEST_EXTENSION_ROOT="${TEST_ROOT}/extensions"
export OUTLINES_CACHE_DIR="${TEST_ROOT}/outlines"
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export PYTHONPATH="${LMOPS_ROOT}/gad/tests:${LMOPS_ROOT}/gad/tools:${LMOPS_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TOKENIZERS_PARALLELISM=false
"${PYTHON_BIN}" "${LMOPS_ROOT}/gad/tests/backend_fixture.py" "${TEST_ROOT}"
failures=0
run_suite() {
  local name="$1"; shift
  echo "=== ${name}"
  if "$@"; then
    echo "=== ${name}: PASSED"
  else
    echo "=== ${name}: FAILED"
    failures=$((failures + 1))
  fi
}
for test in "${LMOPS_ROOT}/sample/tests/"test_*.py "${LMOPS_ROOT}/gad/tests/"test_*.py "${LMOPS_ROOT}/scoring/tests/"test_*.py; do
  run_suite "${test#${LMOPS_ROOT}/}" "${PYTHON_BIN}" "${test}"
done
run_suite "unified scorer self-test" "${PYTHON_BIN}" "${LMOPS_ROOT}/gad/tools/math_metrics.py" --self-test
run_suite "legacy scorer self-test" "${PYTHON_BIN}" "${LMOPS_ROOT}/gad/tools/math_metrics.py" --self-test --scorer-path "${LMOPS_ROOT}/sample/math_scorer.py"
if [[ "${failures}" -eq 0 ]]; then
  echo "ALL CPU SUITES PASSED"
else
  echo "${failures} SUITE(S) FAILED"
fi
exit "${failures}"
