from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path("/tmp/hack_paris")
LOG = ROOT / "logs" / "vae_jepa_train.log"
PID = ROOT / "logs" / "vae_jepa_train.pid"
OUT = ROOT / "data" / "vae_jepa_submission.csv"
VAE = ROOT / "artifacts" / "vae_jepa" / "vae.pt"
JEPA = ROOT / "artifacts" / "vae_jepa" / "jepa.pt"


def running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


print("pid_file:", PID, PID.exists())
if PID.exists():
    pid = int(PID.read_text().strip())
    print("pid:", pid, "running:", running(pid))
    try:
        print(subprocess.check_output(["ps", "-p", str(pid), "-o", "pid,stat,etime,pcpu,pmem,cmd"], text=True))
    except Exception as exc:
        print("ps failed:", repr(exc))
print("log:", LOG, LOG.exists(), LOG.stat().st_size if LOG.exists() else None)
if LOG.exists():
    lines = LOG.read_text(errors="replace").splitlines()
    print("last log lines:")
    for line in lines[-30:]:
        print(line)
print("out:", OUT, OUT.exists(), OUT.stat().st_size if OUT.exists() else None)
print("vae:", VAE, VAE.exists(), VAE.stat().st_size if VAE.exists() else None)
print("jepa:", JEPA, JEPA.exists(), JEPA.stat().st_size if JEPA.exists() else None)
