"""Visualisation d'un volume MRI a partir du tensor produit par `NiiVolume`.

`VolumeVisualizer` prend un tensor `(D, H, W)` (chaque layer = une tranche 2D),
tel que renvoye par `NiiVolume.load`, et affiche tout le volume sous forme de
grille de tranches.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .nii_loader import NiiVolume


class VolumeVisualizer:
    """Affiche l'ensemble des layers d'un volume sous forme de grille.

    Parameters
    ----------
    tensor:
        Volume de forme `(D, H, W)` ou `D` est le nombre de layers.
    name:
        Nom affiche dans le titre (ex. l'identifiant du volume).
    """

    def __init__(self, tensor: torch.Tensor, name: str | None = None) -> None:
        """Memorise le volume a visualiser apres validation de sa forme."""
        if tensor.ndim != 3:
            raise ValueError(f"Volume attendu (D, H, W), recu {tuple(tensor.shape)}")
        self.tensor = tensor
        self.name = name or "volume"

    @classmethod
    def from_nii(cls, path: str | Path, axis: int = 2, normalize: bool = True) -> "VolumeVisualizer":
        """Charge un .nii via `NiiVolume` puis prepare le visualiseur."""
        loader = NiiVolume(axis=axis, normalize=normalize)
        tensor = loader.load(path)
        return cls(tensor, name=Path(path).name.replace(".nii", ""))

    def show(self, n_cols: int = 10, scale: float = 1.4, cmap: str = "gray",
             show_index: bool = True):
        """Affiche tous les layers, `n_cols` par ligne, et renvoie la figure."""
        n_slices = self.tensor.shape[0]
        n_rows = int(np.ceil(n_slices / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
        axes = np.atleast_1d(axes).ravel()
        fig.suptitle(f"{self.name}  —  {n_slices} layers", fontsize=13)

        for z in range(n_slices):
            ax = axes[z]
            ax.imshow(self.tensor[z], cmap=cmap, vmin=0.0, vmax=1.0)
            if show_index:
                ax.set_title(str(z), fontsize=6, pad=1)
            ax.set_xticks([]); ax.set_yticks([])

        for ax in axes[n_slices:]:  # masque les cases vides
            ax.axis("off")

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def show_layer(self, index: int, scale: float = 4.0, cmap: str = "gray"):
        """Affiche une seule tranche (image par image) et renvoie la figure."""
        n_slices = self.tensor.shape[0]
        if index < 0:
            index += n_slices
        if not 0 <= index < n_slices:
            raise IndexError(f"layer {index} hors limites (0..{n_slices - 1})")
        fig, ax = plt.subplots(figsize=(scale, scale))
        ax.imshow(self.tensor[index], cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(f"{self.name}  —  layer {index}/{n_slices - 1}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        return fig

    def animate(self, interval: int = 120, scale: float = 4.0, cmap: str = "gray"):
        """Renvoie une animation qui defile le volume image par image.

        Dans un notebook : `from IPython.display import HTML; HTML(viz.animate().to_jshtml())`.
        """
        from matplotlib import animation

        n_slices = self.tensor.shape[0]
        fig, ax = plt.subplots(figsize=(scale, scale))
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(self.tensor[0], cmap=cmap, vmin=0.0, vmax=1.0)
        title = ax.set_title(f"{self.name}  —  layer 0/{n_slices - 1}", fontsize=11)

        def update(z):
            im.set_data(self.tensor[z])
            title.set_text(f"{self.name}  —  layer {z}/{n_slices - 1}")
            return im, title

        return animation.FuncAnimation(fig, update, frames=n_slices, interval=interval, blit=False)

    def save(self, out_path: str | Path, **kwargs) -> Path:
        """Genere la grille et l'enregistre en image."""
        fig = self.show(**kwargs)
        out_path = Path(out_path)
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return out_path


if __name__ == "__main__":
    import glob

    sample = sorted(glob.glob("data/*.nii"))[0]
    viz = VolumeVisualizer.from_nii(sample)
    out = viz.save("volume_layers.png")
    print(f"{viz.name} : {viz.tensor.shape[0]} layers -> {out}")
