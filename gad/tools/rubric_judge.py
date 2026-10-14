#!/usr/bin/env python3
"""Own and identify one independent judge process; never kill unrelated PIDs."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import compose_config, load_run_config
from extension_config import service_identity, ROOT
from rubric.core.contracts import digest, file_hash


def process_birth(pid):
    return Path(f"/proc/{pid}/stat").read_text().split(") ")[1].split()[19]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "train.yaml"))
    p.add_argument("--run")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--action", choices=("start", "health", "stop", "logs"), default="health")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = load_run_config(args.run) if args.run else compose_config(args.config, settings=args.set)
    options = {key[13:]: value for key, value in cfg.data.items() if key.startswith("rubric.judge.")}
    url = urlparse(options["base_url"])
    port = url.port or 8010
    directory = ROOT / "runs/_rubric_judge" / str(port)
    owner_file = directory / "owner.json"
    fingerprint_file = directory / "fingerprint.json"
    log = directory / "server.log"
    if args.action == "logs":
        print("\n".join(log.read_text().splitlines()[-80:]) if log.exists() else "No judge log")
        return
    owner = json.loads(owner_file.read_text()) if owner_file.is_file() else None
    if args.action == "stop":
        if not owner:
            print("No script-owned judge process")
            return
        pid = owner["pid"]
        if not Path(f"/proc/{pid}").exists():
            owner_file.unlink()
            print("Owned judge already exited")
            return
        if process_birth(pid) != owner["birth"] or os.getpgid(pid) != pid or "rubric_judge_server.py" not in Path(f"/proc/{pid}/cmdline").read_text():
            raise RuntimeError("PID ownership changed; refusing to signal")
        if args.dry_run:
            print(f"Would stop owned judge process group {pid}")
            return
        os.killpg(pid, signal.SIGTERM)
        owner_file.unlink()
        print(f"Stopped owned judge group {pid}")
        return
    identity = service_identity(cfg)
    expected = digest(identity)
    if args.action == "health":
        from gad.extensions.judge_client import JudgeClient
        # No tokenizer/GPU load is needed for identity-only health checks.
        result = JudgeClient(options, expected, tokenizer=object()).health()
        print(json.dumps({"healthy": True, "identity_hash": result["identity_hash"], "model": options["model"],
                          "gpu_ids": result.get("gpu_ids"), "fingerprint_file": str(fingerprint_file)}, indent=2))
        return
    if url.hostname not in ("127.0.0.1", "localhost"):
        raise RuntimeError("start owns only a loopback local endpoint; use health for explicitly configured external services")
    gpu_ids = str(cfg.require("resources.judge_gpus"))
    if set(gpu_ids.split(",")) & set(str(cfg.get("resources.gpus")).split(",")):
        raise RuntimeError("judge GPU IDs overlap training")
    argv = [str(cfg.require("run.python_bin")), str(Path(__file__).with_name("rubric_judge_server.py")),
            "--model", options["tokenizer_path"], "--served-model-name", options["model"],
            "--host", "127.0.0.1", "--port", str(port), "--dtype", options["dtype"],
            "--tensor-parallel-size", str(options["tensor_parallel_size"]),
            "--max-model-len", str(options["max_context_tokens"]),
            "--gpu-memory-utilization", str(options["gpu_memory_utilization"]),
            "--cpu-offload-gb", str(options["cpu_offload_gb"]),
            "--max-num-seqs", "16", "--enforce-eager", "--disable-log-requests",
            "--guided-decoding-backend", "xgrammar"]
    if args.dry_run:
        print(json.dumps({"argv": argv, "gpu_ids": gpu_ids, "identity_hash": expected, "log": str(log)}, indent=2))
        return
    if owner and Path(f"/proc/{owner['pid']}").exists():
        if process_birth(owner["pid"]) == owner["birth"]:
            print(f"Judge process already owned: {owner['pid']}; use --action health")
            return
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    busy = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in query.stdout.splitlines()}
    if any(busy.get(gpu.strip(), 100000) > 512 for gpu in gpu_ids.split(",")):
        raise RuntimeError("judge GPUs unavailable; inspect nvidia-smi before starting")
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {"identity": identity, "identity_hash": expected,
                   "model_sha256": service_identity(cfg, full=True)["model_sha256"], "gpu_ids": gpu_ids}
    fingerprint_file.write_text(json.dumps(fingerprint, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids, GAD_JUDGE_FINGERPRINT_FILE=str(fingerprint_file),
               VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false")
    if options.get("api_key_env"):
        if not os.environ.get(options["api_key_env"]):
            raise RuntimeError("judge API key environment variable is unset")
        env["VLLM_API_KEY"] = os.environ[options["api_key_env"]]
    with log.open("a") as stream:
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=ROOT)
    owner_file.write_text(json.dumps({"pid": child.pid, "birth": process_birth(child.pid), "identity_hash": expected}))
    print(f"Started judge pid={child.pid}, GPUs={gpu_ids}; startup log: {log}")
    print("Run scripts/rubric_judge.sh --action health after model loading completes.")


if __name__ == "__main__":
    main()
