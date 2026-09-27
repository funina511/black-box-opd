#!/usr/bin/env python3
"""Verify that the backend import guard also applies *inside Ray workers*.

`tools/pipeline.py` puts a generated ``sitecustomize.py`` in front of every stage
PYTHONPATH, so a foreign ``verl`` is refused at interpreter start-up. That is easy
to demonstrate in a plain subprocess, but the plan also requires the check to hold
for the Ray workers that actually run the model code - and Ray workers inherit
the driver's environment, which is a property of the deployment, not of our code.

This script closes that gap with one command to run on the training host::

    python tools/check_ray_guard.py --backend .backends/gad
    python tools/check_ray_guard.py --backend .backends/gad --keep-temp

It runs two cases in separate processes:

* **positive** - the worker must import ``verl`` from the given backend;
* **negative** - with a decoy ``verl`` in front, the worker must refuse it.

Ray cannot be started in every sandbox (it needs shared memory and a working
raylet); when it fails to start the script reports that instead of pretending the
check passed, and exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import PipelineError, ensure_dir, git_rev_parse, log, write_json  # noqa: E402

CHILD = r'''
import json, os, sys

payload = {}
try:
    import ray
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR")
except Exception as exc:  # noqa: BLE001
    print("@@RESULT@@" + json.dumps({"ray_error": f"{type(exc).__name__}: {exc}"}))
    raise SystemExit(0)


@ray.remote
def check():
    import pathlib
    import verl
    return str(pathlib.Path(verl.__file__).resolve())


try:
    payload["worker_verl"] = ray.get(check.remote())
except Exception as exc:  # noqa: BLE001
    # The *full* message is sent: Ray wraps the worker exception in a long
    # traceback, so truncating here would drop the guard marker and turn a
    # correctly refused import into a false failure. The parent summarizes.
    message = f"{type(exc).__name__}: {exc}"
    payload["worker_error"] = message
    payload["worker_error_len"] = len(message)
finally:
    try:
        ray.shutdown()
    except Exception:  # noqa: BLE001
        pass

print("@@RESULT@@" + json.dumps(payload))
'''

GUARD_MARKER = "gad-guard"
EXCERPT_LIMIT = 400


def summarize_worker_error(message: str, *, marker: str = GUARD_MARKER, limit: int = EXCERPT_LIMIT) -> dict:
    """Decide *on the full text* whether the guard fired, and keep a bounded excerpt.

    The decision must never depend on the excerpt: Ray appends the worker
    traceback to the exception message, so the ``gad-guard`` line can sit far
    beyond any fixed prefix.
    """
    message = str(message or "")
    has_marker = marker in message
    if has_marker:
        position = message.index(marker)
        start = max(0, position - limit // 2)
        excerpt = message[start:start + limit]
        if start:
            excerpt = "..." + excerpt
        if start + limit < len(message):
            excerpt = excerpt + "..."
    else:
        excerpt = message[:limit] + ("..." if len(message) > limit else "")
    return {
        "worker_error": excerpt,
        "worker_error_len": len(message),
        "worker_error_has_guard": has_marker,
    }


def run_case(*, python_bin: str, guard_dir: Path, target: Path, expected: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(guard_dir), str(target)])
    env["GAD_EXPECT_VERL_PATH"] = str(expected)
    env["GAD_BACKEND_STAGE"] = "guard-check"
    result = subprocess.run([python_bin, "-c", CHILD], env=env, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            payload = json.loads(line[len("@@RESULT@@"):])
            if payload.get("worker_error") is not None:
                payload.update(summarize_worker_error(payload.pop("worker_error")))
            return payload
    return {"error": f"no result marker (exit {result.returncode})", "stderr": result.stderr[-400:]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, help="Stage backend worktree, e.g. .backends/gad")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    backend = Path(args.backend).expanduser().resolve()
    if not (backend / "verl").is_dir():
        raise PipelineError(f"{backend} is not a verl worktree")
    commit = git_rev_parse(backend, "HEAD")

    temp_root = Path(tempfile.mkdtemp(prefix="gad-ray-guard-"))
    guard_dir = ensure_dir(temp_root / "guard")
    decoy = ensure_dir(temp_root / "decoy" / "verl")
    (decoy / "__init__.py").write_text("", encoding="utf-8")

    # Reuse the exact guard the pipeline installs, so this checks the real thing.
    from pipeline import write_guard

    write_guard(None, temp_root / "run", "guard-check", backend)
    shutil.copy2(temp_root / "run" / "_guard" / "guard-check" / "sitecustomize.py", guard_dir / "sitecustomize.py")

    log(f"backend : {backend} @ {commit[:10]}")
    log(f"guard   : {guard_dir}")

    positive = run_case(python_bin=args.python_bin, guard_dir=guard_dir, target=backend, expected=backend)
    negative = run_case(
        python_bin=args.python_bin, guard_dir=guard_dir, target=temp_root / "decoy", expected=backend
    )

    report = {
        "backend": str(backend),
        "backend_commit": commit,
        "python_bin": args.python_bin,
        "positive_case": positive,
        "negative_case": negative,
        "ray_available": "ray_error" not in positive and "ray_error" not in negative,
        "positive_ok": str(positive.get("worker_verl", "")).startswith(str(backend)),
        "negative_ok": bool(negative.get("worker_error_has_guard"))
        or GUARD_MARKER in str(negative.get("worker_error", "")),
    }
    report["ok"] = bool(report["ray_available"] and report["positive_ok"] and report["negative_ok"])

    if args.report:
        write_json(args.report, report)
    if not args.keep_temp:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["ray_available"]:
        log("Ray could not start in this environment; the worker-side check is INCONCLUSIVE (not passed).")
        return 2
    if not report["ok"]:
        log("worker-side guard check FAILED")
        return 1
    log("worker-side guard verified: workers import the stage backend and refuse a foreign verl")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PipelineError as exc:
        log(f"FATAL: {exc}")
        raise SystemExit(2)
