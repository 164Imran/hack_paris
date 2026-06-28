# -*- coding: utf-8 -*-
"""
Plot 3D d'un volume IRM (.nii / .nii.gz).

Usage:
    python plot_3d.py                         # 1er volume trouvé dans data/
    python plot_3d.py chemin/vers/volume.nii  # volume précis
    python plot_3d.py volume.nii --mode iso    # mode: volume | iso | slices

Produit un fichier HTML interactif (ouvrable dans le navigateur) et tente de l'ouvrir.
"""
import os
import sys
import glob
import argparse

import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure
from skimage.transform import resize

DATA_ROOT = r"C:\Users\User\PycharmProjects\hack_paris\data\ehl-paris-medical-image-retrieval"


def find_default_volume():
    """Premier volume .nii trouvé sous DATA_ROOT."""
    hits = glob.glob(os.path.join(DATA_ROOT, "**", "*.nii*"), recursive=True)
    if not hits:
        sys.exit(f"Aucun volume .nii trouvé sous {DATA_ROOT}")
    return hits[0]


def load_volume(path, max_dim=96):
    """Charge le volume, normalise en [0,1] et sous-échantillonne si trop gros."""
    vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    # normalisation robuste (percentiles)
    pos = vol[vol > 0]
    lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (vol.min(), vol.max()))
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    # sous-échantillonnage pour rester fluide en 3D
    factor = max(1, int(np.ceil(max(vol.shape) / max_dim)))
    if factor > 1:
        new_shape = tuple(s // factor for s in vol.shape)
        vol = resize(vol, new_shape, order=1, anti_aliasing=True, preserve_range=True)
    return vol.astype(np.float32)


def plot_volume(vol, title):
    """Rendu volumétrique semi-transparent."""
    X, Y, Z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]
    fig = go.Figure(data=go.Volume(
        x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=vol.ravel(),
        isomin=0.15, isomax=1.0, opacity=0.08, surface_count=18,
        colorscale="gray",
        caps=dict(x_show=False, y_show=False, z_show=False),
    ))
    fig.update_layout(title=f"Rendu volumique — {title}",
                      scene=dict(aspectmode="data"), width=800, height=750)
    return fig


def plot_iso(vol, title, level=0.25):
    """Isosurface (marching cubes)."""
    verts, faces, _, _ = measure.marching_cubes(vol, level=level)
    fig = go.Figure(data=[go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        intensity=verts[:, 2], colorscale="gray",
        opacity=1.0, showscale=True, colorbar=dict(title="Z"),
    )])
    fig.update_layout(title=f"Isosurface (seuil={level}) — {title}",
                      scene=dict(aspectmode="data"), width=800, height=750)
    return fig


def plot_slices(vol, title, n=8):
    """Coupes axiales empilées en 3D à leur vraie hauteur Z."""
    zs = np.linspace(vol.shape[2] * 0.2, vol.shape[2] * 0.8, n).astype(int)
    X, Y = np.mgrid[0:vol.shape[0], 0:vol.shape[1]]
    fig = go.Figure()
    for z in zs:
        sl = vol[:, :, z]
        fig.add_surface(
            x=X, y=Y, z=np.full_like(sl, z, dtype=float),
            surfacecolor=sl, colorscale="gray", showscale=False, opacity=0.9,
        )
    fig.update_layout(title=f"Coupes empilées en 3D — {title}",
                      scene=dict(xaxis_title="X", yaxis_title="Y",
                                 zaxis_title="Z (profondeur)", aspectmode="data"),
                      width=800, height=750)
    return fig


def main():
    ap = argparse.ArgumentParser(description="Plot 3D d'un volume IRM .nii")
    ap.add_argument("path", nargs="?", default=None, help="chemin du volume .nii(.gz)")
    ap.add_argument("--mode", choices=["volume", "iso", "slices"], default="volume")
    ap.add_argument("--max-dim", type=int, default=96, help="taille max d'un axe (sous-échantillonnage)")
    ap.add_argument("--out", default=None, help="fichier HTML de sortie")
    ap.add_argument("--no-open", action="store_true", help="ne pas ouvrir le navigateur")
    args = ap.parse_args()

    path = args.path or find_default_volume()
    if not os.path.exists(path) and path.endswith(".gz"):
        path = path[:-3]
    if not os.path.exists(path):
        sys.exit(f"Introuvable: {path}")

    title = os.path.basename(path)
    print(f"Chargement : {path}")
    vol = load_volume(path, max_dim=args.max_dim)
    print(f"  shape utilisée : {vol.shape}  (intensités [{vol.min():.2f}, {vol.max():.2f}])")

    fig = {"volume": plot_volume, "iso": plot_iso, "slices": plot_slices}[args.mode](vol, title)

    out = args.out or f"plot3d_{args.mode}_{os.path.splitext(title)[0]}.html"
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"[OK] ecrit : {os.path.abspath(out)}")

    if not args.no_open:
        try:
            fig.show()
        except Exception as e:
            print(f"(ouverture auto échouée: {e} — ouvre {out} manuellement)")


if __name__ == "__main__":
    main()
