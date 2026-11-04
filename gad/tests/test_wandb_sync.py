"""CPU regression tests; W&B/network uploads are mocked."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import wandb_sync


class WandbSyncTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run_dir = Path(self.temp.name) / "experiment"
        (self.run_dir / "logs").mkdir(parents=True)
        self.created = []
        self.wandb = SimpleNamespace(init=Mock(side_effect=self.new_run), Settings=Mock())
        self.patch = patch.dict(sys.modules, {"wandb": self.wandb})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def new_run(self, **kwargs):
        directory = self.run_dir / "wandb" / f"offline-run-test-{len(self.created)}" / "files"
        directory.mkdir(parents=True)
        run = SimpleNamespace(dir=str(directory), log=Mock(), define_metric=Mock(), finish=Mock())
        self.created.append(run)
        return run

    def write_log(self, text, stage="warmup"):
        (self.run_dir / "logs" / f"{stage}.log").write_text(text)

    def backfill(self, stage="warmup", force=False):
        return wandb_sync.backfill_stage(self.run_dir, stage, "test-project", force=force)

    def test_growing_log_updates_without_completion_marker_and_skips_unchanged(self):
        self.write_log("step:1 - critic/d_loss:0.5\n")
        self.backfill()
        self.backfill()
        self.assertEqual(len(self.created), 1)
        self.write_log("step:1 - critic/d_loss:0.5\nstep:2 - critic/d_loss:0.4\n")
        self.backfill()
        self.assertEqual(len(self.created), 2)
        self.assertEqual([c.kwargs["step"] for c in self.created[-1].log.call_args_list], [1, 2])
        # Keep old bytes for provenance, but do not upload two versions by default.
        self.assertTrue(Path(self.created[0].dir).parent.is_dir())
        self.assertEqual(wandb_sync.offline_runs(self.run_dir), [Path(self.created[-1].dir).parent])
        self.backfill()
        self.assertEqual(len(self.created), 2)

    def test_validation_changes_refresh_even_if_console_is_unchanged(self):
        self.write_log("step:1 - critic/d_loss:0.5\n")
        self.backfill()
        path = self.run_dir / "warmup" / "validation_math_metrics.jsonl"
        path.parent.mkdir()
        path.write_text(json.dumps({"global_step": 1, "acc": 0.5}) + "\n")
        self.backfill()
        self.assertEqual(len(self.created), 2)
        self.assertEqual(self.created[-1].log.call_args.args[0]["val_math/val/acc"], 0.5)

    def test_stage_steps_use_original_handoff_even_with_resumed_partial_log(self):
        (self.run_dir / "handoff_manifest.json").write_text(json.dumps({"step": 212}))
        self.write_log("step:213 - critic/d_loss:0.5\nstep:300 - critic/d_loss:0.4\n", "gad")
        self.backfill("gad")
        calls = self.created[-1].log.call_args_list
        self.assertEqual([c.args[0]["training/stage_step"] for c in calls], [1, 88])
        self.assertEqual([c.kwargs["step"] for c in calls], [213, 300])
        # W&B defaults to this axis; the original global steps are still present.
        self.created[-1].define_metric.assert_any_call("*", step_metric="training/stage_step")
        self.write_log("step:300 - critic/d_loss:0.4\n", "gad")
        self.backfill("gad")
        self.assertEqual(self.created[-1].log.call_args.args[0]["training/stage_step"], 88)

    def test_zero_start_for_seqkd_and_warmup_including_initial_validation(self):
        for stage in ("seqkd", "warmup"):
            self.write_log("step:0 - val/acc:0.1\nstep:12 - loss:0.5\n", stage)
            self.backfill(stage)
            self.assertEqual([c.args[0]["training/stage_step"] for c in self.created[-1].log.call_args_list], [0, 12])

    def test_legacy_backfill_is_refreshed_once_and_old_versions_are_excluded(self):
        self.write_log("step:1 - loss:0.5\n")
        old = self.run_dir / "wandb" / "offline-run-legacy"
        old.mkdir(parents=True)
        (old.parent / "backfill_warmup.json").write_text(json.dumps({
            "offline_run": str(old), "source_sha256": wandb_sync.sha256_file(self.run_dir / "logs/warmup.log")}))
        self.backfill()
        self.backfill(force=True)
        self.assertEqual(wandb_sync.offline_runs(self.run_dir), [Path(self.created[-1].dir).parent])
        self.assertTrue(old.is_dir())

    def test_upload_and_listing_filter_native_and_backfill_runs_by_stage(self):
        warmup = self.run_dir / "wandb" / "offline-run-warmup"
        gad = warmup.with_name("offline-run-gad")
        unknown = warmup.with_name("offline-run-unknown")
        for directory in (warmup, gad, unknown):
            directory.mkdir(parents=True)
        argv = ["wandb_sync.py", str(self.run_dir), "--stages", "gad"]
        with patch.object(wandb_sync, "offline_stage", side_effect=lambda p: {
                warmup: "warmup", gad: "gad"}.get(p)), \
                patch.object(wandb_sync, "resolve_run", return_value=self.run_dir), \
                patch.object(wandb_sync.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as upload:
            for action in ([], ["--upload"]):
                output = io.StringIO()
                with patch.object(sys, "argv", argv + action), contextlib.redirect_stdout(output):
                    self.assertEqual(wandb_sync.main(), 0)
                self.assertNotIn(warmup.name, output.getvalue())
                self.assertIn(gad.name, output.getvalue())
                if not action:
                    upload.assert_not_called()
            self.assertEqual(upload.call_count, 1)
            command = upload.call_args.args[0]
            self.assertIn(str(gad), command)
            self.assertNotIn(str(warmup), command)
            self.assertNotIn(str(unknown), command)


if __name__ == "__main__":
    unittest.main()
