from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def show(path: str, max_items: int = 30) -> None:
    root = Path(path)
    print(f"\n## {root} exists={root.exists()}")
    if not root.exists():
        return
    try:
        for item in list(root.iterdir())[:max_items]:
            print(" -", item)
    except Exception as exc:
        print("cannot list:", repr(exc))


for path in ["/kaggle", "/kaggle/input", "/workspace", "/root", "/home", "/tmp/hack_paris", "/mnt", "/data"]:
    show(path)

print("\n## searching train_pairs.csv")
for base in [Path("/kaggle"), Path("/workspace"), Path("/root"), Path("/home"), Path("/tmp"), Path("/mnt"), Path("/data")]:
    if not base.exists():
        continue
    try:
        matches = list(base.glob("**/dataset1/train_pairs.csv"))[:20]
    except Exception as exc:
        print(base, "search failed:", repr(exc))
        continue
    for match in matches:
        print(" -", match)

print("\n## kaggle tooling")
print("kaggle command:", shutil.which("kaggle"))
print("KAGGLE_USERNAME set:", bool(os.environ.get("KAGGLE_USERNAME")))
print("KAGGLE_KEY set:", bool(os.environ.get("KAGGLE_KEY")))
for cfg in [Path.home() / ".kaggle" / "kaggle.json", Path("/root/.kaggle/kaggle.json")]:
    print(cfg, "exists:", cfg.exists())
if shutil.which("kaggle"):
    try:
        print(subprocess.check_output(["kaggle", "--version"], text=True, stderr=subprocess.STDOUT, timeout=30))
    except Exception as exc:
        print("kaggle --version failed:", repr(exc))

print("\n## disk")
try:
    print(subprocess.check_output(["df", "-h"], text=True, timeout=30))
except Exception as exc:
    print("df failed:", repr(exc))
