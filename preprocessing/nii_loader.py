"""Chargement d'un fichier NIfTI (.nii / .nii.gz) en tensor torch.

La classe `NiiVolume` lit un volume 3D et le convertit en un seul tensor
empile le long de l'axe de coupe, de forme `(D, H, W)`. Chaque layer (tranche)
est alors un tenseur 2D accessible par indexation : `volume[i]`.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch


class NiiVolume:
    """Charge un .nii et expose le volume comme un tensor de layers.

    Le tensor renvoye a la forme `(D, H, W)` ou `D` est le nombre de tranches
    le long de `axis`. Indexer le tensor donne un layer 2D : `tensor[i]`.

    Parameters
    ----------
    axis:
        Axe du volume le long duquel les tranches sont empilees (defaut 2).
    normalize:
        Si vrai, met les intensites a l'echelle [0, 1] (min-max).
    dtype:
        Type du tensor de sortie (defaut torch.float32).
    """

    def __init__(self, axis: int = 2, normalize: bool = True, dtype: torch.dtype = torch.float32) -> None:
        """Memorise les options de chargement."""
        self.axis = axis
        self.normalize = normalize
        self.dtype = dtype

        self.path: Path | None = None
        self.affine: np.ndarray | None = None
        self.tensor: torch.Tensor | None = None

    def load(self, path: str | Path) -> torch.Tensor:
        """Charge le fichier .nii et renvoie le tensor `(D, H, W)`.

        Chaque element de la premiere dimension est un layer 2D.
        """
        path = Path(path)
        nii = nib.load(str(path))
        volume = torch.from_numpy(nii.get_fdata().astype(np.float32))
        volume = torch.nan_to_num(volume)

        if self.normalize:
            vmin, vmax = volume.min(), volume.max()
            if vmax > vmin:
                volume = (volume - vmin) / (vmax - vmin)

        # Place l'axe de coupe en premier -> (D, H, W), chaque layer = tensor[i]
        volume = volume.movedim(self.axis, 0).contiguous().to(self.dtype)

        self.path = path
        self.affine = nii.affine
        self.tensor = volume
        return volume

    def layers(self) -> list[torch.Tensor]:
        """Renvoie la liste des layers 2D (un tenseur par tranche)."""
        self._require_loaded()
        return list(self.tensor)

    def __len__(self) -> int:
        """Nombre de layers du volume charge."""
        self._require_loaded()
        return self.tensor.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        """Renvoie le layer 2D a la position `index`."""
        self._require_loaded()
        return self.tensor[index]

    def _require_loaded(self) -> None:
        """Verifie qu'un volume a bien ete charge avant tout acces."""
        if self.tensor is None:
            raise RuntimeError("Aucun volume charge. Appelle d'abord .load(path).")


if __name__ == "__main__":
    # Petit test manuel sur le premier .nii du dossier data.
    import glob

    sample = sorted(glob.glob("data/*.nii"))[0]
    vol = NiiVolume()
    tensor = vol.load(sample)
    print(f"{Path(sample).name} -> tensor {tuple(tensor.shape)} dtype={tensor.dtype}")
    print(f"{len(vol)} layers ; layer[0] shape = {tuple(vol[0].shape)}")
