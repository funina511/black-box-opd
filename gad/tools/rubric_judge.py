#!/usr/bin/env python3
"""Own and identify one independent judge process; never kill unrelated PIDs."""
import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import add_config_arguments,select_config
from extension_config import judge_engine_settings, service_identity, ROOT
from rubric.core.contracts import digest


def process_birth(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[19]


def live_group(pgid):
    members = set()
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(fields[2]) == pgid and fields[0] not in ("Z", "X"):
            members.add(int(path.parent.name))
    return members


def gpu_pids(gpu_ids):
    query = subprocess.run(["nvidia-smi", "-i", gpu_ids, "--query-compute-apps=pid",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return {int(line) for line in query.stdout.splitlines() if line.strip()}


def stop_owned(owner, owner_file, *, dry_run=False, restart_gpus=None):
    if not owner:
        print("No script-owned judge process")
        return
    pid = owner["pid"]
    members = live_group(pid)
    if members:
        if (pid not in members or process_birth(pid) != owner["birth"] or os.getpgid(pid) != pid
                or "rubric_judge_server.py" not in Path(f"/proc/{pid}/cmdline").read_text()):
            raise RuntimeError("PID ownership changed; refusing to signal")
        devices = gpu_pids(restart_gpus) if restart_gpus else set()
        if devices - members:
            raise RuntimeError(f"Other GPU processes {sorted(devices - members)} are active; leave the Judge running")
        if dry_run:
            print(f"Would stop owned judge process group {pid}")
            return
        os.killpg(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while live_group(pid) or (devices and devices & gpu_pids(restart_gpus)):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Judge group {pid} has not released its workers; inspect the server log")
            time.sleep(0.5)
        print(f"Stopped owned judge group {pid}; released GPU workers {sorted(devices)}", flush=True)
    if not dry_run:
        owner_file.unlink()


def wait_ready(child, options, expected, gpu_ids, log):
    from gad.extensions.judge_client import JudgeClient, ServiceUnavailable
    client = JudgeClient(options, expected, tokenizer=object())
    deadline = time.monotonic() + 300
    while child.poll() is None:
        try:
            result = client.health(deadline=min(deadline, time.monotonic() + 5))
        except ServiceUnavailable as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Timed out waiting for judge pid={child.pid}; inspect {log}") from exc
            time.sleep(1)
            continue
        if str(result.get("gpu_ids")) != gpu_ids:
            raise RuntimeError(f"Judge GPU allocation differs from configured GPUs {gpu_ids}")
        print(f"Ready: model={options['model']} pid={child.pid} identity={expected} GPUs={gpu_ids}", flush=True)
        return
    raise RuntimeError(f"Judge pid={child.pid} exited with status {child.returncode}; inspect {log}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_config_arguments(p)
    p.add_argument("--action", choices=("start", "restart", "health", "stop", "logs"), default="health")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    cfg = select_config(args)
    url = urlparse(cfg.require("rubric.judge.base_url"))
    directory = ROOT / "runs/_rubric_judge" / str(url.port or 8010)
    if args.action in ("start", "restart", "stop") and url.hostname not in ("127.0.0.1", "localhost"):
        raise RuntimeError("Judge process control requires a loopback endpoint")
    if args.action in ("start", "restart", "stop") and not args.dry_run:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "control.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            manage(cfg, args, directory)
    else:
        manage(cfg, args, directory)


def manage(cfg, args, directory):
    options = {key[13:]: value for key, value in cfg.data.items() if key.startswith("rubric.judge.")}
    thinking = options['enable_thinking']
    if type(thinking) is not bool or (thinking and options['reasoning_parser'] != 'deepseek_r1') or (not thinking and options['reasoning_parser'] is not None):
        raise ValueError('judge thinking/parser combination is unsupported on vLLM 0.8.5')
    url = urlparse(options["base_url"])
    port = url.port or 8010
    owner_file = directory / "owner.json"
    fingerprint_file = directory / "fingerprint.json"
    log = directory / "server.log"
    if args.action == "logs":
        print("\n".join(log.read_text().splitlines()[-80:]) if log.exists() else "No judge log")
        return
    owner = json.loads(owner_file.read_text()) if owner_file.is_file() else None
    if args.action == "stop":
        stop_owned(owner, owner_file, dry_run=args.dry_run)
        return
    identity = service_identity(cfg)
    expected = digest(identity)
    gpu_ids = str(cfg.require("resources.judge_gpus"))
    scheduling = judge_engine_settings(cfg)
    launch_settings = {**scheduling, "gpu_memory_utilization": options["gpu_memory_utilization"]}
    if args.action == "health":
        from gad.extensions.judge_client import JudgeClient
        # No tokenizer/GPU load is needed for identity-only health checks.
        result = JudgeClient(options, expected, tokenizer=object()).health()
        if str(result.get("gpu_ids")) != gpu_ids:
            raise RuntimeError(f"Running Judge uses GPUs {result.get('gpu_ids')}; expected {gpu_ids}")
        if "launch_settings" in result and result["launch_settings"] != launch_settings:
            raise RuntimeError("running Judge launch settings differ; stop it before changing the server configuration")
        print(json.dumps({"healthy": True, "identity_hash": result["identity_hash"], "model": options["model"],
                          "gpu_ids": gpu_ids, "launch_settings": result.get("launch_settings"),
                          "fingerprint_file": str(fingerprint_file)}, indent=2))
        return
    if (set(gpu_ids.split(",")) & set(str(cfg.get("resources.gpus")).split(","))
            and cfg.get("resources.allow_judge_gpu_sharing") is not True):
        raise RuntimeError("judge GPU IDs overlap training")
    argv = [str(cfg.require("run.python_bin")), str(Path(__file__).with_name("rubric_judge_server.py")),
            "--model", options["tokenizer_path"], "--served-model-name", options["model"],
            "--host", "127.0.0.1", "--port", str(port), "--dtype", options["dtype"],
            "--tensor-parallel-size", str(options["tensor_parallel_size"]),
            "--max-model-len", str(options["max_context_tokens"]),
            "--gpu-memory-utilization", str(options["gpu_memory_utilization"]),
            "--cpu-offload-gb", str(options["cpu_offload_gb"]),
            "--max-num-seqs", str(scheduling["max_num_seqs"]), "--disable-log-requests",
            "--guided-decoding-backend", "xgrammar"]
    if scheduling["enforce_eager"]:
        argv.append("--enforce-eager")
    if thinking:
        argv += ['--enable-reasoning', '--reasoning-parser', options['reasoning_parser']]
    if args.dry_run:
        print(json.dumps({"action": args.action, "argv": argv, "gpu_ids": gpu_ids, "identity_hash": expected, "log": str(log),
                          "environment": {'VLLM_USE_V1': '0'}, 'enable_thinking': thinking}, indent=2))
        return
    if options.get("api_key_env") and not os.environ.get(options["api_key_env"]):
        raise RuntimeError("judge API key environment variable is unset")
    if args.action == "restart":
        if owner and owner.get("identity_hash") != expected:
            raise RuntimeError("Owned Judge identity differs from requested configuration; refusing to restart")
        stop_owned(owner, owner_file, restart_gpus=gpu_ids)
        with socket.socket() as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Loopback port {port} is already owned by another process")
    elif owner and Path(f"/proc/{owner['pid']}").exists():
        if process_birth(owner["pid"]) == owner["birth"]:
            current = json.loads(fingerprint_file.read_text())
            if owner["identity_hash"] != expected or current.get("gpu_ids") != gpu_ids:
                raise RuntimeError("running Judge identity or GPU allocation differs; stop it before starting another configuration")
            if "launch_settings" in current:
                if current["launch_settings"] != launch_settings:
                    raise RuntimeError("running Judge launch settings differ; stop it before starting another configuration")
            else:
                # Legacy launches record the scheduling defaults in their
                # known identity and the GPU memory fraction in their argv.
                process_args = Path(f"/proc/{owner['pid']}/cmdline").read_bytes().decode().split("\0")
                fraction = "--gpu-memory-utilization"
                if fraction not in process_args or float(process_args[process_args.index(fraction) + 1]) != options["gpu_memory_utilization"]:
                    raise RuntimeError("legacy Judge GPU memory fraction differs or cannot be verified; stop it before starting another configuration")
            print(f"Judge process already owned: {owner['pid']}; use --action health")
            return
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True)
    busy = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in query.stdout.splitlines()}
    if any(busy.get(gpu.strip(), 100000) > 512 for gpu in gpu_ids.split(",")):
        raise RuntimeError("judge GPUs unavailable; inspect nvidia-smi before starting")
    fingerprint = {"identity": identity, "identity_hash": expected, "gpu_ids": gpu_ids, "launch_settings": launch_settings}
    fingerprint_file.write_text(json.dumps(fingerprint, indent=2))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_ids, GAD_JUDGE_FINGERPRINT_FILE=str(fingerprint_file),
               VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false", VLLM_USE_V1="0")
    if options.get("api_key_env"):
        env["VLLM_API_KEY"] = os.environ[options["api_key_env"]]
    with log.open("a") as stream:
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True, cwd=ROOT)
    owner_file.write_text(json.dumps({"pid": child.pid, "birth": process_birth(child.pid), "identity_hash": expected}))
    print(f"Started judge pid={child.pid}, GPUs={gpu_ids}; startup log: {log}", flush=True)
    if args.action == "restart":
        wait_ready(child, options, expected, gpu_ids, log)
    else:
        print("Run scripts/rubric_judge.sh --action health after model loading completes.")


if __name__ == "__main__":
    main()
