# -*- coding: utf-8 -*-
"""
Reconstruction de MRIPairPreprocessor (les modules src/ d'origine ont ete supprimes).

Fournit l'API utilisee par le script de continue-training MaskedSliceJEPA :
  - MRIPairPreprocessor(add_vae_reconstructions=False, slices_per_axis=6)
  - MRIPairPreprocessor.resolve_path(data_root, image_path)        [staticmethod]
  - preprocessor.load_volume(path) -> (torch.Tensor [X,Y,Z], affine)
  - preprocessor.select_slice_indices(n) -> list[int]
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import torch


class MRIPairPreprocessor:
    def __init__(self, add_vae_reconstructions: bool = False, slices_per_axis: int = 6) -> None:
        self.add_vae_reconstructions = add_vae_reconstructions
        self.slices_per_axis = slices_per_axis

    # ------------------------------------------------------------------ paths
    @staticmethod
    def resolve_path(data_root: Path, image_path: str) -> Path:
        """Resout un chemin du CSV (souvent en .nii.gz) vers le fichier reel sur disque."""
        data_root = Path(data_root)
        # le chemin du CSV est relatif a la racine du dataset, separateurs '/'
        full = data_root / Path(*str(image_path).split("/"))
        if full.exists():
            return full
        # .nii.gz -> .nii  (les fichiers extraits sont en .nii)
        if str(full).endswith(".gz"):
            alt = Path(str(full)[:-3])
            if alt.exists():
                return alt
        # .nii -> .nii.gz
        alt_gz = Path(str(full) + ".gz")
        if alt_gz.exists():
            return alt_gz
        return full

    # ------------------------------------------------------------------ volume
    def load_volume(self, path: Path):
        """Charge un volume .nii, normalise en [0, 1] (percentiles). Retourne (tensor[X,Y,Z], affine)."""
        img = nib.load(str(path))
        vol = np.asarray(img.get_fdata(), dtype=np.float32)
        pos = vol[vol > 0]
        if pos.size:
            lo, hi = np.percentile(pos, [1.0, 99.0])
        else:
            lo, hi = float(vol.min()), float(vol.max())
        vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        tensor = torch.from_numpy(np.ascontiguousarray(vol))
        affine = torch.from_numpy(np.asarray(img.affine, dtype=np.float32))
        return tensor, affine

    # ------------------------------------------------------------------ slices
    def select_slice_indices(self, n: int) -> list[int]:
        """Indices de `slices_per_axis` coupes regulierement espacees dans la zone centrale."""
        k = min(self.slices_per_axis, n)
        if k <= 0:
            return [n // 2]
        lo, hi = 0.2 * n, 0.8 * n
        idx = np.linspace(lo, hi, k)
        idx = np.clip(np.round(idx).astype(int), 0, n - 1)
        # garder l'ordre, retirer les doublons eventuels
        seen, out = set(), []
        for i in idx.tolist():
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out
