#!/usr/bin/env python3
"""Create one detached git worktree per GAD stage and apply the engineering patches.

Why: SeqKD, Warmup and GAD live on three different branches of the same verl
fork. Running them from a single checkout means `git checkout` between stages,
which makes the "which code produced this run" question unanswerable and risks
importing a stray ``verl`` from another project (for example an installed copy
under a different workspace). Instead every stage gets its own immutable
worktree pinned to a commit, plus a recorded patch hash.

Usage::

    python tools/prepare_backends.py --config configs/pipeline.example.yaml
    python tools/prepare_backends.py --repo /path/to/gad/verl --backend-root /path/to/gad/.backends
    python tools/prepare_backends.py --repo ... --print-pythonpath gad

Outputs under ``--backend-root``::

    seqkd/ warmup/ gad/            detached worktrees at the pinned commits
    <stage>.patch                  the exact engineering diff applied to that stage
    backend_manifest.json          commits, patch hashes, verification results

This script never commits, never pushes and never deletes a worktree unless
``--recreate`` is passed explicitly.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend_patches import ALL_PATCHES, apply_patches, verify_patches  # noqa: E402
from common import (  # noqa: E402
    PipelineError,
    capture_command,
    ensure_dir,
    git_is_dirty,
    git_rev_parse,
    log,
    run_command,
    sha256_file,
    utc_timestamp,
    write_json,
)

# Pinned by LMOps/GAD修改计划.md section 1.
PINNED_BACKENDS = {
    "seqkd": {"branch": "seqkd", "commit": "c8b3a62f5edd4994e13a15a461cf9527973695bd"},
    "warmup": {"branch": "warmup", "commit": "b6e88286c3e16d6f364e698f60c3be608f1a36cc"},
    "gad": {"branch": "gad", "commit": "707757b9202c466d986e529a4658434d0ac6a7ee"},
}


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    import yaml

    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise PipelineError(f"config not found: {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def commit_exists(repo: Path, commit: str) -> bool:
    code, _, _ = capture_command(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"])
    return code == 0


def ensure_commit(repo: Path, branch: str, commit: str, *, allow_fetch: bool) -> bool:
    if commit_exists(repo, commit):
        return False
    if not allow_fetch:
        raise PipelineError(
            f"commit {commit} ({branch}) is not present in {repo}. "
            "Run with --fetch (needs network) or fetch the branch manually."
        )
    log(f"fetching origin/{branch} in {repo}")
    run_command(["git", "-C", str(repo), "fetch", "origin", branch])
    if not commit_exists(repo, commit):
        resolved = git_rev_parse(repo, f"origin/{branch}")
        raise PipelineError(
            f"after fetching origin/{branch}, commit {commit} is still missing (origin/{branch} is {resolved}). "
            "The branch moved: update the pin in configs/pipeline.example.yaml and re-run."
        )
    return True


def worktree_state(path: Path) -> dict:
    if not (path / ".git").exists():
        return {"exists": path.exists(), "is_worktree": False}
    return {
        "exists": True,
        "is_worktree": True,
        "commit": git_rev_parse(path, "HEAD"),
        "dirty": git_is_dirty(path),
    }


def create_worktree(repo: Path, path: Path, commit: str, *, recreate: bool) -> None:
    state = worktree_state(path)
    if state.get("is_worktree"):
        if state.get("commit") == commit and not recreate:
            log(f"worktree already at {commit[:10]}: {path}")
            return
        if not recreate:
            raise PipelineError(
                f"{path} is a worktree at {str(state.get('commit'))[:10]} but {commit[:10]} is pinned. "
                "Re-run with --recreate to rebuild it (this deletes the directory)."
            )
        log(f"removing worktree {path} (--recreate)")
        run_command(["git", "-C", str(repo), "worktree", "remove", "--force", str(path)], check=False)
        if path.exists():
            shutil.rmtree(path)
    elif path.exists():
        raise PipelineError(f"{path} exists but is not a git worktree; move it aside first")

    ensure_dir(path.parent)
    run_command(["git", "-C", str(repo), "worktree", "add", "--detach", str(path), commit])


def write_patch_file(worktree: Path, target: Path) -> tuple[str, int]:
    code, diff, err = capture_command(["git", "-C", str(worktree), "diff"])
    if code != 0:
        raise PipelineError(f"git diff failed in {worktree}: {err}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not diff.strip():
        target.write_text("# no engineering patches applied to this backend\n", encoding="utf-8")
    else:
        target.write_text(diff + "\n", encoding="utf-8")
    return sha256_file(target), len(diff.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Pipeline config (uses its `backends` section).")
    parser.add_argument("--repo", default=None, help="Source verl checkout (the pinned clone).")
    parser.add_argument("--backend-root", default=None, help="Where the per-stage worktrees are created.")
    parser.add_argument("--stages", default="seqkd,warmup,gad")
    parser.add_argument("--fetch", action="store_true", help="Fetch the branch when the pinned commit is missing.")
    parser.add_argument("--recreate", action="store_true", help="Rebuild an existing worktree at the wrong commit.")
    parser.add_argument("--no-patches", action="store_true", help="Only create worktrees, apply nothing.")
    parser.add_argument("--print-pythonpath", default=None, metavar="STAGE",
                        help="Print the PYTHONPATH for one stage and exit.")
    args = parser.parse_args()

    config = load_config(args.config)
    backends_cfg = config.get("backends", {}) if isinstance(config, dict) else {}
    repo = Path(
        args.repo
        or backends_cfg.get("source_repo")
        or (Path(__file__).resolve().parents[1] / "verl")
    ).expanduser().resolve()
    backend_root = Path(
        args.backend_root
        or backends_cfg.get("root")
        or (Path(__file__).resolve().parents[1] / ".backends")
    ).expanduser().resolve()

    pins = dict(PINNED_BACKENDS)
    for stage, override in (backends_cfg.get("pins") or {}).items():
        pins.setdefault(stage, {}).update(override or {})

    if args.print_pythonpath:
        stage = args.print_pythonpath
        if stage not in pins:
            raise PipelineError(f"unknown stage {stage!r}; known: {sorted(pins)}")
        print(backend_root / stage)
        return 0

    if not repo.exists():
        raise PipelineError(f"source repo not found: {repo}")
    if not (repo / "verl").is_dir():
        raise PipelineError(f"{repo} does not look like a verl checkout (no verl/ directory)")

    repo_commit = git_rev_parse(repo, "HEAD")
    repo_dirty = git_is_dirty(repo)
    log(f"source repo : {repo} @ {repo_commit[:10]}{' (dirty)' if repo_dirty else ''}")
    log(f"backend root: {backend_root}")
    if repo_dirty:
        log("WARNING: the source checkout has local changes; the worktrees use committed objects only")

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    manifest = {
        "stage": "prepare_backends",
        "created_at": utc_timestamp(),
        "source_repo": str(repo),
        "source_repo_commit": repo_commit,
        "source_repo_dirty": repo_dirty,
        "backend_root": str(backend_root),
        "patches_available": [{"name": p.name, "description": p.description} for p in ALL_PATCHES],
        "backends": {},
    }

    failures: list[str] = []
    for stage in stages:
        if stage not in pins:
            raise PipelineError(f"unknown stage {stage!r}; known: {sorted(pins)}")
        pin = pins[stage]
        branch, commit = pin["branch"], pin["commit"]
        path = backend_root / stage
        log(f"--- {stage}: branch={branch} commit={commit[:10]} -> {path}")
        fetched = ensure_commit(repo, branch, commit, allow_fetch=args.fetch)
        create_worktree(repo, path, commit, recreate=args.recreate)

        applied: list[dict] = []
        if not args.no_patches:
            for result in apply_patches(path):
                applied.append({"name": result.name, "changed": result.changed, "files": result.files})
            ok, problems = verify_patches(path)
            if not ok:
                failures.extend(f"{stage}: {problem}" for problem in problems)

        patch_file = backend_root / f"{stage}.patch"
        patch_sha, patch_lines = write_patch_file(path, patch_file)

        manifest["backends"][stage] = {
            "branch": branch,
            "requested_commit": commit,
            "worktree": str(path),
            "head_commit": git_rev_parse(path, "HEAD"),
            "fetch_used": fetched,
            "patches_applied": applied,
            "patch_file": str(patch_file),
            "patch_sha256": patch_sha,
            "patch_lines": patch_lines,
            "pythonpath": str(path),
            "verified": not failures or not any(f.startswith(f"{stage}:") for f in failures),
        }
        log(f"{stage}: HEAD={manifest['backends'][stage]['head_commit'][:10]} patch={patch_sha[:12]} ({patch_lines} lines)")

    manifest["ok"] = not failures
    manifest["problems"] = failures
    write_json(backend_root / "backend_manifest.json", manifest)

    if failures:
        for problem in failures:
            log(f"ERROR: {problem}")
        log("backend preparation FAILED")
        return 1
    log(f"backend manifest -> {backend_root / 'backend_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
