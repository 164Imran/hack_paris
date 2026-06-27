from __future__ import annotations

import shutil
from pathlib import Path


source = Path("/tmp/hack_paris/notebooks/vae_outputs_visualization.ipynb")
target = Path.cwd() / "vae_outputs_visualization.ipynb"
shutil.copy2(source, target)
print("cwd:", Path.cwd())
print("copied:", source, "->", target)
