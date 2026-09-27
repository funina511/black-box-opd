#!/usr/bin/env python3
"""End-to-end Warmup -> GAD export test with realistic sharded checkpoints.

The handoff tool drives the *original* ``tools/merge_model2hf.py``, so a real
export needs FSDP-style ``DTensor`` shards. This test builds them from a tiny
Qwen3 model on CPU (gloo, world size 1, replicated placement), stages a complete
Warmup checkpoint and then runs the handoff **without** ``--skip-merge``.

It verifies what plan section 3.3 asks for and what is easy to get wrong:

* the merged actor really reloads as a causal LM with unchanged weights;
* the merged discriminator still has its single-scalar value head, and the head
  weight survives the merge;
* the merged discriminator's ``config.json`` is corrected to a
  ``...ForTokenClassification`` architecture with ``num_labels: 1`` - because
  ``merge_model2hf.py`` rebuilds from the *student's* architecture string and
  would otherwise leave a value head that downstream loaders mis-route;
* the GAD directory gets the checkpoint, the latest-step marker and the
  dataloader state, while the Warmup directory stays untouched.

Run::

    python LMOps/gad/tests/test_handoff_export.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

GAD_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = GAD_DIR / "tools"
MERGE_TOOL = GAD_DIR / "tools" / "merge_model2hf.py"
PYTHON = sys.executable

SCORE_WEIGHT_VALUE = 0.125
SCORE_BIAS_VALUE = -0.5
TRAINED_SENTINEL = 0.75

TRAINED_STEP_BUILDER = r'''
import json, os, sys
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29518")
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate
from transformers import AutoModelForCausalLM

gad_dir = Path(sys.argv[1])
handoff_step = int(sys.argv[2])
trained_step = int(sys.argv[3])

dist.init_process_group("gloo", rank=0, world_size=1)
mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))

# the trained actor: same architecture as the init, different weights
model = AutoModelForCausalLM.from_pretrained(gad_dir / f"global_step_{handoff_step}" / "actor" / "huggingface")
with torch.no_grad():
    for parameter in model.parameters():
        parameter.add_(0.01)
    # set last so the sentinel survives exactly in bf16
    model.model.embed_tokens.weight.flatten()[0] = float(sys.argv[4]) if len(sys.argv) > 4 else 0.75

ckpt = gad_dir / f"global_step_{trained_step}"
role_dir = ckpt / "actor"
role_dir.mkdir(parents=True, exist_ok=True)
sharded = {k: distribute_tensor(v.detach().clone(), mesh, [Replicate()]) for k, v in model.state_dict().items()}
torch.save(sharded, role_dir / "model_world_size_1_rank_0.pt")
torch.save({"state": {}}, role_dir / "optim_world_size_1_rank_0.pt")
torch.save({"lr_scheduler": None, "rng": {}}, role_dir / "extra_state_world_size_1_rank_0.pt")
source = gad_dir / f"global_step_{handoff_step}" / "actor"
for item in source.iterdir():
    if item.is_file() and item.suffix != ".pt":
        (role_dir / item.name).write_bytes(item.read_bytes())
torch.save({"_num_yielded": trained_step * 10}, ckpt / "data.pt")
(gad_dir / "latest_checkpointed_iteration.txt").write_text(str(trained_step), encoding="utf-8")
dist.destroy_process_group()
'''

# Qwen3-1.7B ships `tie_word_embeddings: true`, and for tied models
# `save_pretrained` drops lm_head.weight while the FSDP shard keeps it. The
# exporter must accept that (and prove the tie survives) instead of reporting the
# LM head as lost data - the first real run would otherwise abort at the handoff.
TIED_SHARD_BUILDER = r'''
import json, os, sys
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29519")
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification

root = Path(sys.argv[1])
step = int(sys.argv[2])

dist.init_process_group("gloo", rank=0, world_size=1)
mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))

config = AutoConfig.for_model(
    "qwen3", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
)
config.tie_word_embeddings = True
actor = AutoModelForCausalLM.from_config(config)
actor_dir = root / "seed" / "actor"
actor.save_pretrained(actor_dir)

critic_config = AutoConfig.from_pretrained(actor_dir)
critic_config.num_labels = 1
critic = AutoModelForTokenClassification.from_pretrained(actor_dir, config=critic_config)
with torch.no_grad():
    critic.score.weight.fill_(0.25)
    if critic.score.bias is not None:
        critic.score.bias.fill_(0.0)
critic_dir = root / "seed" / "critic"
critic.save_pretrained(critic_dir)

ckpt = root / "warmup" / f"global_step_{step}"
for role, module in (("actor", actor), ("critic", critic)):
    role_dir = ckpt / role
    role_dir.mkdir(parents=True, exist_ok=True)
    sharded = {k: distribute_tensor(v.detach().clone(), mesh, [Replicate()]) for k, v in module.state_dict().items()}
    torch.save(sharded, role_dir / "model_world_size_1_rank_0.pt")
    torch.save({"state": {}}, role_dir / "optim_world_size_1_rank_0.pt")
    torch.save({"lr_scheduler": None, "rng": {}}, role_dir / "extra_state_world_size_1_rank_0.pt")
    (role_dir / "config.json").write_text((actor_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (role_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (role_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

torch.save({"_num_yielded": step * 10}, ckpt / "data.pt")
(root / "warmup" / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
hit = torch.load(ckpt / "actor" / "model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
summary = {
    "tie_word_embeddings": bool(config.tie_word_embeddings),
    "shard_has_lm_head": "lm_head.weight" in hit,
    "shard_lm_head_equals_embed": bool(torch.equal(hit["lm_head.weight"]._local_tensor, hit["model.embed_tokens.weight"]._local_tensor)),
    "vocab_size": int(config.vocab_size),
}
(root / "seed" / "tied_summary.json").write_text(json.dumps(summary), encoding="utf-8")
dist.destroy_process_group()
'''

SHARD_BUILDER = r'''
import json, os, sys
from pathlib import Path

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29517")
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification

root = Path(sys.argv[1])
step = int(sys.argv[2])

dist.init_process_group("gloo", rank=0, world_size=1)
# real FSDP checkpoints carry a named mesh; merge_model2hf asserts on it
mesh = DeviceMesh("cpu", [0], mesh_dim_names=("fsdp",))

config = AutoConfig.for_model(
    "qwen3", hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, vocab_size=64,
)
actor = AutoModelForCausalLM.from_config(config)
actor_dir = root / "seed" / "actor"
actor.save_pretrained(actor_dir)

critic_config = AutoConfig.from_pretrained(actor_dir)
critic_config.num_labels = 1
critic = AutoModelForTokenClassification.from_pretrained(actor_dir, config=critic_config)
# Give the value head an identifiable value so we can prove it survived the merge.
with torch.no_grad():
    critic.score.weight.fill_(float(sys.argv[3]))
    if critic.score.bias is not None:
        critic.score.bias.fill_(float(sys.argv[4]))
critic_dir = root / "seed" / "critic"
critic.save_pretrained(critic_dir)

ckpt = root / "warmup" / f"global_step_{step}"
for role, module in (("actor", actor), ("critic", critic)):
    role_dir = ckpt / role
    role_dir.mkdir(parents=True, exist_ok=True)
    sharded = {k: distribute_tensor(v.detach().clone(), mesh, [Replicate()]) for k, v in module.state_dict().items()}
    torch.save(sharded, role_dir / "model_world_size_1_rank_0.pt")
    torch.save({"state": {}}, role_dir / "optim_world_size_1_rank_0.pt")
    torch.save({"lr_scheduler": None, "rng": {}}, role_dir / "extra_state_world_size_1_rank_0.pt")
    # what verl writes next to the shards: the *student's* config for both roles
    (role_dir / "config.json").write_text((actor_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (role_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (role_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

torch.save({"_num_yielded": step * 10, "_sampler_iter_yielded": step * 10}, ckpt / "data.pt")
(root / "warmup" / "latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")

summary = {
    "actor_keys": len(actor.state_dict()),
    "critic_keys": len(critic.state_dict()),
    "hidden_size": config.hidden_size,
}
(root / "seed" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
dist.destroy_process_group()
'''


class HandoffExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.step = 6
        if not MERGE_TOOL.exists():
            raise unittest.SkipTest("merge_model2hf.py missing")
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gad_handoff_hf"))
        result = subprocess.run(
            [PYTHON, "-c", SHARD_BUILDER, str(cls.root), str(cls.step), str(SCORE_WEIGHT_VALUE), str(SCORE_BIAS_VALUE)],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(f"could not build DTensor shards in this environment:\n{result.stderr[-800:]}")

        cls.gad_dir = cls.root / "gad"
        cls.handoff = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--warmup-dir", str(cls.root / "warmup"),
                "--gad-dir", str(cls.gad_dir),
                "--merge-tool", str(MERGE_TOOL),
                "--python-bin", PYTHON,
                "--gad-steps", "4",
                "--manifest", str(cls.root / "handoff.json"),
            ],
            capture_output=True, text=True,
        )
        cls.manifest = (
            json.loads((cls.root / "handoff.json").read_text(encoding="utf-8"))
            if (cls.root / "handoff.json").exists()
            else {}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_handoff_reported_success(self) -> None:
        self.assertEqual(self.handoff.returncode, 0, self.handoff.stdout + self.handoff.stderr)
        self.assertTrue(self.manifest.get("ok"), self.manifest.get("problems"))

    def test_merged_actor_reloads_with_unchanged_weights(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        actor_hf = Path(self.manifest["actor_hf"])
        merged = AutoModelForCausalLM.from_pretrained(actor_hf)
        seed = torch.load(
            self.root / "warmup" / f"global_step_{self.step}" / "actor" / "model_world_size_1_rank_0.pt",
            map_location="cpu", weights_only=False,
        )
        for key, value in merged.state_dict().items():
            self.assertTrue(torch.allclose(value.float(), seed[key]._local_tensor.float(), atol=1e-2),
                            f"actor weight changed during merge: {key}")

    def test_merged_discriminator_keeps_its_single_scalar_value_head(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification

        critic_hf = Path(self.manifest["critic_hf"])
        config = json.loads((critic_hf / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["num_labels"], 1)
        self.assertTrue(config["architectures"][0].endswith("ForTokenClassification"), config["architectures"])

        auto_config = AutoConfig.from_pretrained(critic_hf)
        model = AutoModelForTokenClassification.from_pretrained(critic_hf, config=auto_config)
        self.assertTrue(hasattr(model, "score"))
        self.assertEqual(model.score.out_features, 1)
        self.assertTrue(torch.allclose(model.score.weight.float(),
                                       torch.full_like(model.score.weight.float(), SCORE_WEIGHT_VALUE), atol=1e-2))
        self.assertTrue(torch.allclose(model.score.bias.float(),
                                       torch.full_like(model.score.bias.float(), SCORE_BIAS_VALUE), atol=1e-2))

    def test_manifest_records_the_value_head_and_step_target(self) -> None:
        value_head = self.manifest["critic_value_head"]
        self.assertEqual(value_head["score_keys"], ["score.bias", "score.weight"])
        self.assertEqual(value_head["score_weight_shape"], [1, 16])
        self.assertEqual(self.manifest["step"], self.step)
        self.assertEqual(self.manifest["target_total_steps"], self.step + 4)
        self.assertEqual(self.manifest["dataloader_state"]["_num_yielded"], self.step * 10)

    def test_gad_directory_is_staged_and_warmup_is_untouched(self) -> None:
        self.assertEqual((self.gad_dir / "latest_checkpointed_iteration.txt").read_text().strip(), str(self.step))
        staged = self.gad_dir / f"global_step_{self.step}"
        for role in ("actor", "critic"):
            self.assertTrue((staged / role / "model_world_size_1_rank_0.pt").exists())
            self.assertTrue((staged / role / "huggingface" / "config.json").exists())
        self.assertTrue((staged / "data.pt").exists())
        # the warmup checkpoint must not gain merge outputs
        self.assertFalse((self.root / "warmup" / f"global_step_{self.step}" / "actor" / "huggingface").exists())

    def test_second_handoff_refuses_to_overwrite(self) -> None:
        result = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--warmup-dir", str(self.root / "warmup"),
                "--gad-dir", str(self.gad_dir),
                "--skip-merge",
            ],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to overwrite", result.stdout + result.stderr)


class FinalModelExportTests(unittest.TestCase):
    """The final evaluation must score the *trained* actor, not the Warmup init.

    ``handoff_manifest.actor_hf`` is the merged Warmup checkpoint that GAD starts
    from; reporting it as "the GAD model" silently publishes before-training
    numbers. So the export is exercised against a GAD directory that holds both
    the handoff step and a later, distinct step.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.handoff_step = 6
        if not MERGE_TOOL.exists():
            raise unittest.SkipTest("merge_model2hf.py missing")
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gad_handoff_hf"))
        result = subprocess.run(
            [PYTHON, "-c", SHARD_BUILDER, str(cls.root), str(cls.handoff_step), str(SCORE_WEIGHT_VALUE),
             str(SCORE_BIAS_VALUE)],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(f"could not build DTensor shards in this environment:\n{result.stderr[-800:]}")

        # start from the staged handoff, then add a *later* trained step whose
        # actor weights differ from the init (so the two are distinguishable)
        cls.gad_dir = cls.root / "gad"
        handoff = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--warmup-dir", str(cls.root / "warmup"),
                "--gad-dir", str(cls.gad_dir),
                "--merge-tool", str(MERGE_TOOL),
                "--python-bin", PYTHON,
                "--manifest", str(cls.root / "handoff.json"),
            ],
            capture_output=True, text=True,
        )
        if handoff.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(f"handoff failed:\n{handoff.stdout[-800:]}{handoff.stderr[-800:]}")

        cls.trained_step = cls.handoff_step + 4
        builder = subprocess.run(
            [PYTHON, "-c", TRAINED_STEP_BUILDER, str(cls.gad_dir), str(cls.handoff_step), str(cls.trained_step),
             str(TRAINED_SENTINEL)],
            capture_output=True, text=True, env=env,
        )
        if builder.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(f"could not build the trained step:\n{builder.stderr[-800:]}")

        cls.export = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--mode", "export-final",
                "--gad-dir", str(cls.gad_dir),
                "--merge-tool", str(MERGE_TOOL),
                "--python-bin", PYTHON,
                "--manifest", str(cls.root / "final_model.json"),
            ],
            capture_output=True, text=True,
        )
        cls.manifest = (
            json.loads((cls.root / "final_model.json").read_text(encoding="utf-8"))
            if (cls.root / "final_model.json").exists()
            else {}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_export_selects_the_last_complete_gad_step(self) -> None:
        self.assertEqual(self.export.returncode, 0, self.export.stdout + self.export.stderr)
        self.assertTrue(self.manifest.get("ok"), self.manifest.get("problems"))
        self.assertEqual(self.manifest["step"], self.trained_step)
        self.assertNotEqual(self.manifest["step"], self.handoff_step)
        self.assertIn(f"global_step_{self.trained_step}", self.manifest["actor_hf"])

    def test_exported_model_carries_the_trained_weights(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        actor_hf = Path(self.manifest["actor_hf"])
        merged = AutoModelForCausalLM.from_pretrained(actor_hf)
        source = torch.load(
            self.gad_dir / f"global_step_{self.trained_step}" / "actor" / "model_world_size_1_rank_0.pt",
            map_location="cpu", weights_only=False,
        )
        compared = 0
        for key, value in merged.state_dict().items():
            expert = source.get(key)
            if expert is None or value.shape != expert._local_tensor.shape:
                continue
            self.assertTrue(
                torch.allclose(value.float(), expert._local_tensor.float(), atol=1e-2),
                f"exported weight does not match the trained shard: {key}",
            )
            compared += 1
        self.assertGreater(compared, 0, "no weight could be compared")
        # the trained sentinel proves this is not the Warmup init export
        embed = merged.state_dict()["model.embed_tokens.weight"].float().flatten()[0]
        self.assertAlmostEqual(float(embed), TRAINED_SENTINEL, places=3)

    def test_exported_actor_is_not_a_discriminator(self) -> None:
        self.assertEqual(self.manifest["weight_keys"]["lost"], [])
        self.assertEqual(self.manifest["weight_keys"]["added"], [])
        self.assertFalse([key for key in self.manifest["config"]["architectures"] if "TokenClassification" in key])
        actor_hf = Path(self.manifest["actor_hf"])
        self.assertFalse((actor_hf / "score.weight").exists())

    def test_incomplete_checkpoint_is_never_exported(self) -> None:
        # a newer but incomplete step must be skipped, not exported
        incomplete = self.gad_dir / f"global_step_{self.trained_step + 5}" / "actor"
        incomplete.mkdir(parents=True)
        result = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--mode", "export-final",
                "--gad-dir", str(self.gad_dir),
                "--python-bin", PYTHON,
                "--check-only",
                "--manifest", str(self.root / "final_model_check.json"),
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        checked = json.loads((self.root / "final_model_check.json").read_text(encoding="utf-8"))
        self.assertEqual(checked["step"], self.trained_step)
        inspected = {item["step"]: item["problems"] for item in checked["inspected_checkpoints"]}
        self.assertIn(self.trained_step + 5, inspected)
        self.assertTrue(inspected[self.trained_step + 5])


class TiedEmbeddingExportTests(unittest.TestCase):
    """The real student (Qwen3-1.7B) is `tie_word_embeddings: true`.

    For tied models a plain ``save_pretrained`` drops ``lm_head.weight`` (it is
    an alias of the embedding) while verl's FSDP shard keeps it. The exporter
    must never report a *correct* export as data loss, must still refuse a
    genuine loss, and must prove the merged model really re-ties the head.
    (``merge_model2hf.py`` happens to keep the head because it materialises the
    model with ``to_empty``, which breaks the alias - the tolerance exists for
    the paths where that is not the case, e.g. a plain ``save_pretrained``.)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.step = 5
        if not MERGE_TOOL.exists():
            raise unittest.SkipTest("merge_model2hf.py missing")
        env = dict(os.environ)
        env.setdefault("HF_HOME", str(Path(os.environ.get("TMPDIR", "/tmp")) / "gad_handoff_hf"))
        built = subprocess.run(
            [PYTHON, "-c", TIED_SHARD_BUILDER, str(cls.root), str(cls.step)],
            capture_output=True, text=True, env=env,
        )
        if built.returncode != 0:
            cls._tmp.cleanup()
            raise unittest.SkipTest(f"could not build tied DTensor shards:\n{built.stderr[-800:]}")
        cls.tied = json.loads((cls.root / "seed" / "tied_summary.json").read_text(encoding="utf-8"))

        cls.gad_dir = cls.root / "gad"
        cls.handoff_proc = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--warmup-dir", str(cls.root / "warmup"),
                "--gad-dir", str(cls.gad_dir),
                "--merge-tool", str(MERGE_TOOL),
                "--python-bin", PYTHON,
                "--manifest", str(cls.root / "handoff.json"),
            ],
            capture_output=True, text=True,
        )
        cls.manifest = (
            json.loads((cls.root / "handoff.json").read_text(encoding="utf-8"))
            if (cls.root / "handoff.json").exists()
            else {}
        )
        cls.export_proc = subprocess.run(
            [
                PYTHON, str(TOOLS_DIR / "prepare_gad_checkpoint.py"),
                "--mode", "export-final",
                "--gad-dir", str(cls.gad_dir),
                "--merge-tool", str(MERGE_TOOL),
                "--python-bin", PYTHON,
                "--manifest", str(cls.root / "final_model.json"),
            ],
            capture_output=True, text=True,
        )
        cls.final = (
            json.loads((cls.root / "final_model.json").read_text(encoding="utf-8"))
            if (cls.root / "final_model.json").exists()
            else {}
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_fixture_really_is_tied(self) -> None:
        self.assertTrue(self.tied["tie_word_embeddings"])
        self.assertTrue(self.tied["shard_has_lm_head"], "the FSDP shard should carry the tied LM head")
        self.assertTrue(self.tied["shard_lm_head_equals_embed"])

    def test_handoff_accepts_the_tied_model_without_data_loss(self) -> None:
        self.assertEqual(self.handoff_proc.returncode, 0, self.handoff_proc.stdout + self.handoff_proc.stderr)
        self.assertTrue(self.manifest.get("ok"), self.manifest.get("problems"))
        keys = self.manifest["weight_keys"]
        self.assertEqual(keys["actor_lost"], [])
        self.assertEqual(keys["actor_added"], [])
        self.assertEqual(keys["critic_lost"], [])
        self.assertTrue(keys["actor_tie_word_embeddings"])

    def test_final_export_accepts_the_tied_model_without_data_loss(self) -> None:
        self.assertEqual(self.export_proc.returncode, 0, self.export_proc.stdout + self.export_proc.stderr)
        self.assertTrue(self.final.get("ok"), self.final.get("problems"))
        self.assertEqual(self.final["weight_keys"]["lost"], [])
        self.assertEqual(self.final["weight_keys"]["added"], [])
        self.assertTrue(self.final["config"]["tie_word_embeddings"])
        # the head must survive: dropping it silently would be data loss
        from safetensors import safe_open

        merged_keys: set[str] = set()
        for path in sorted(Path(self.final["actor_hf"]).glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                merged_keys.update(handle.keys())
        self.assertIn("lm_head.weight", merged_keys)
        self.assertIn("model.embed_tokens.weight", merged_keys)

    def test_merged_actor_keeps_the_trained_lm_head(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM

        actor_hf = Path(self.final["actor_hf"])
        merged = AutoModelForCausalLM.from_pretrained(actor_hf)
        state = merged.state_dict()
        self.assertTrue(
            torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]),
            "the merged export must have lm_head tied to embed_tokens",
        )
        source = torch.load(
            self.gad_dir / f"global_step_{self.step}" / "actor" / "model_world_size_1_rank_0.pt",
            map_location="cpu", weights_only=False,
        )
        self.assertTrue(
            torch.allclose(state["lm_head.weight"].float(), source["lm_head.weight"]._local_tensor.float(), atol=1e-2),
            "the (re-)tied LM head must match the trained shard",
        )

    def test_tied_head_absence_is_excused_but_real_loss_is_not(self) -> None:
        """Unit-level check of the comparison helper used by both export paths."""
        sys.path.insert(0, str(TOOLS_DIR))
        import prepare_gad_checkpoint as pgc

        source = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
        merged = {"model.embed_tokens.weight", "model.norm.weight"}
        tied = pgc.compare_weight_keys(source, merged, config={"tie_word_embeddings": True}, role="actor")
        self.assertEqual(tied["lost"], [])
        self.assertEqual(tied["tied_absent"], ["lm_head.weight"])
        self.assertEqual(tied["problems"], [])

        untied = pgc.compare_weight_keys(source, merged, config={"tie_word_embeddings": False}, role="actor")
        self.assertEqual(untied["lost"], ["lm_head.weight"])

        # tie declared but the embedding the head ties to is gone -> real loss
        broken = pgc.compare_weight_keys(
            source, {"model.norm.weight"}, config={"tie_word_embeddings": True}, role="actor"
        )
        self.assertFalse(broken["tied_absent"])
        self.assertIn("lm_head.weight", broken["lost"])
        self.assertTrue(broken["problems"])

        # re-materialised head is also fine for a tied export
        readded = pgc.compare_weight_keys(
            {"model.embed_tokens.weight"}, {"model.embed_tokens.weight", "lm_head.weight"},
            config={"tie_word_embeddings": True}, role="actor",
        )
        self.assertEqual(readded["tied_readded"], ["lm_head.weight"])
        self.assertEqual(readded["added"], [])

    def test_a_genuine_missing_weight_is_still_refused(self) -> None:
        """The tied-head excuse must not become a blanket exemption.

        Uses the *real* key sets produced above (source shard vs merged export)
        and removes one ordinary weight from the merged side: that must be
        reported as lost data, not excused by the tie.
        """
        sys.path.insert(0, str(TOOLS_DIR))
        import prepare_gad_checkpoint as pgc

        source = pgc.source_weight_keys(self.gad_dir / f"global_step_{self.step}" / "actor")
        merged = set()
        from safetensors import safe_open

        for path in sorted(Path(self.final["actor_hf"]).glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                merged.update(handle.keys())
        self.assertTrue(source and merged)
        self.assertEqual(source, merged, "the fixture merge should be lossless")

        victim = next(iter(sorted(key for key in merged if key.endswith("norm.weight"))), None)
        self.assertIsNotNone(victim, f"no norm weight in the fixture: {sorted(merged)[:5]}")
        comparison = pgc.compare_weight_keys(
            source, merged - {victim}, config={"tie_word_embeddings": True}, role="actor"
        )
        self.assertEqual(comparison["lost"], [victim])
        self.assertEqual(comparison["tied_absent"], [])
        self.assertFalse(comparison["problems"])  # reported by the caller as "lost"


if __name__ == "__main__":
    unittest.main(verbosity=2)
