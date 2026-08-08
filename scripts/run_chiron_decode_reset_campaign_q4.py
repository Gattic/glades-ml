#!/usr/bin/env python3
"""Run the frozen isolated CHIRON decode-reset diagnostic campaign."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/tmp/chiron_decode_reset_instrumented_q4_20260808")
BINARY = REPO / "unit-tests/build/glades-unit-tests"
RUNS, GPU_HOUR_BAR = 20, 0.05

class CampaignError(RuntimeError): pass

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()

def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n"); temp.replace(path)

def gpu_processes() -> list[str]:
    completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode: raise CampaignError("nvidia-smi process query failed: " + completed.stderr.strip())
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]

def manifest(root: Path) -> dict[str, Any]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json")
    return {"schema": "chiron_decode_reset_q4_manifest_v1", "self_excluded": "artifact_manifest.json",
            "artifacts": [{"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)} for path in files]}

def run() -> int:
    if ROOT.exists(): raise CampaignError(f"refusing existing campaign root: {ROOT}")
    if not BINARY.is_file(): raise CampaignError(f"missing unit-test binary: {BINARY}")
    status = subprocess.check_output(["git", "-C", str(REPO), "status", "--short"], text=True).strip()
    if status: raise CampaignError("repository must be clean")
    if gpu_processes(): raise CampaignError("unrelated GPU process before campaign")
    ROOT.mkdir(); (ROOT / "logs").mkdir()
    summary: dict[str, Any] = {"schema": "chiron_decode_reset_q4_v1", "status": "RUNNING",
        "head": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        "binary": str(BINARY), "binary_sha256": sha256(BINARY), "planned_runs": RUNS,
        "gpu_hour_bar": GPU_HOUR_BAR, "runs": [], "adaptive_reruns": 0}
    atomic_json(ROOT / "summary.json", summary); total = 0.0
    for index in range(1, RUNS + 1):
        active = gpu_processes()
        if active: raise CampaignError(f"unrelated GPU process before run {index}: {'; '.join(active)}")
        log = ROOT / "logs" / f"chiron-model-{index:02d}.log"
        env = os.environ.copy(); env["CHIRON_DECODE_CAMPAIGN_RUN"] = str(index)
        command = [str(BINARY), "chiron-model"]; started_unix = time.time(); started = time.monotonic()
        with log.open("w") as stream:
            stream.write("command=" + " ".join(command) + f"\ncampaign_run={index}\n"); stream.flush()
            process = subprocess.Popen(command, cwd=REPO / "unit-tests", env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
            try: returncode = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(); returncode = -999
        elapsed = time.monotonic() - started; total += elapsed
        text = log.read_text(errors="replace"); diagnostic = "CHIRON_DECODE_MISMATCH" in text
        row = {"index": index, "pid": process.pid, "command": command, "started_unix": started_unix,
               "elapsed_seconds": elapsed, "returncode": returncode, "timed_out": returncode == -999,
               "diagnostic_mismatch": diagnostic, "log": str(log), "log_sha256": sha256(log)}
        summary["runs"].append(row); atomic_json(ROOT / "summary.json", summary)
        if returncode != 0 or diagnostic:
            summary["status"] = "FAIL-CONFIRMED" if diagnostic else "FAIL-INFRA"
            summary["stopped_at_run"] = index; break
    else: summary["status"] = "PASS"
    summary["completed_runs"] = len(summary["runs"]); summary["elapsed_seconds"] = total
    summary["conservative_gpu_hours"] = total / 3600.0
    if summary["conservative_gpu_hours"] > GPU_HOUR_BAR and summary["status"] == "PASS": summary["status"] = "FAIL-INFRA"
    atomic_json(ROOT / "summary.json", summary); atomic_json(ROOT / "artifact_manifest.json", manifest(ROOT))
    print(f"CHIRON_DECODE_RESET_Q4_{summary['status']} " + json.dumps({"summary": str(ROOT / "summary.json"),
          "completed_runs": summary["completed_runs"], "gpu_hours": summary["conservative_gpu_hours"]}, sort_keys=True))
    return 0 if summary["status"] == "PASS" else 2

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true"); args = parser.parse_args()
    if not args.run:
        print(json.dumps({"schema": "chiron_decode_reset_q4_description_v1", "default_starts_process": False,
            "root": str(ROOT), "runs": RUNS, "gpu_hour_bar": GPU_HOUR_BAR,
            "command": [str(BINARY), "chiron-model"]}, indent=2, sort_keys=True)); return 0
    try: return run()
    except CampaignError as exc:
        print(f"CHIRON_DECODE_RESET_Q4_ERROR: {exc}", file=os.sys.stderr); return 3

if __name__ == "__main__": raise SystemExit(main())
