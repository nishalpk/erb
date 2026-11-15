"""Run the remaining local work in sequence: full model runs, then the probe, then the analysis.

Keeps Windows from idle-sleeping while it runs (SetThreadExecutionState; released when the process exits;
a closed lid or manual Sleep still sleeps). Each step resumes, so rerunning after an interruption is safe.

    python scripts/run_all_local.py [--skip-probe]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "results" / "run_all_local.log"


def keep_awake():
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def step(name, args):
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as log:
        log.write(f"\n=== {name} start {time.strftime('%Y-%m-%d %H:%M')}\n")
        log.flush()
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
        r = subprocess.run([PY, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        log.write(f"=== {name} end rc={r.returncode} after {(time.time() - t0) / 60:.1f} min\n")
    print(f"[{time.strftime('%H:%M')}] {name}: rc={r.returncode}, {(time.time() - t0) / 60:.1f} min", flush=True)
    return r.returncode


def main():
    keep_awake()
    steps = [("smolvlm2-2.2b", ["src/04_run_models.py", "--model", "smolvlm2-2.2b"]),
             ("llava-ov-0.5b", ["src/04_run_models.py", "--model", "llava-ov-0.5b"]),
             ("qwen2.5-vl-3b", ["src/04_run_models.py", "--model", "qwen2.5-vl-3b"])]
    if "--skip-probe" not in sys.argv:
        steps.append(("probe+patching (qwen2.5-vl-3b)", ["src/06_probe_patch.py", "--model", "qwen2.5-vl-3b", "--n", "200"]))
    steps.append(("analysis", ["src/05_analysis.py"]))
    for name, args in steps:
        step(name, args)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
