# -*- coding: utf-8 -*-
"""
GIF animé (rotation 360 degres) d'un rendu 3D noir & blanc d'un volume IRM .nii.

Usage:
    python plot_3d_gif.py                          # 1er volume de data/
    python plot_3d_gif.py chemin/vol.nii           # volume precis
    python plot_3d_gif.py vol.nii --frames 48 --fps 20 --out rot.gif

Rend une isosurface (marching cubes) ombree en niveaux de gris, capture une rotation
complete et l'enregistre en GIF.
"""
import os
import sys
import glob
import argparse

import numpy as np
import nibabel as nib
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
from skimage.transform import resize

DATA_ROOT = r"C:\Users\User\PycharmProjects\hack_paris\data\ehl-paris-medical-image-retrieval"


def find_default_volume():
    hits = glob.glob(os.path.join(DATA_ROOT, "**", "*.nii*"), recursive=True)
    if not hits:
        sys.exit(f"Aucun volume .nii trouve sous {DATA_ROOT}")
    return hits[0]


def load_volume(path, max_dim=80):
    vol = np.asarray(nib.load(path).get_fdata(), dtype=np.float32)
    pos = vol[vol > 0]
    lo, hi = (np.percentile(pos, [1, 99]) if pos.size else (vol.min(), vol.max()))
    vol = np.clip((vol - lo) / max(hi - lo, 1e-6), 0, 1)
    factor = max(1, int(np.ceil(max(vol.shape) / max_dim)))
    if factor > 1:
        vol = resize(vol, tuple(s // factor for s in vol.shape),
                     order=1, anti_aliasing=True, preserve_range=True)
    return vol.astype(np.float32)


def build_mesh(vol, level=0.25):
    """Isosurface + couleur de face en niveaux de gris (ombrage de Lambert)."""
    verts, faces, normals, _ = measure.marching_cubes(vol, level=level, step_size=1)
    light = np.array([0.4, 0.3, 0.85])
    light = light / np.linalg.norm(light)
    fnorm = normals[faces].mean(axis=1)                 # normale moyenne par face
    fnorm /= (np.linalg.norm(fnorm, axis=1, keepdims=True) + 1e-9)
    shade = np.clip(np.abs(fnorm @ light), 0, 1)        # intensite [0,1]
    gray = 0.15 + 0.85 * shade                          # eviter le noir total
    facecolors = np.stack([gray, gray, gray, np.ones_like(gray)], axis=1)
    return verts, faces, facecolors


def render_gif(vol, out, frames=48, fps=18, level=0.25, elev=18):
    verts, faces, facecolors = build_mesh(vol, level=level)
    tris = verts[faces]
    print(f"  isosurface : {len(verts)} sommets, {len(faces)} faces")

    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    coll = Poly3DCollection(tris, facecolors=facecolors, edgecolors="none")
    ax.add_collection3d(coll)
    ax.set_xlim(0, vol.shape[0]); ax.set_ylim(0, vol.shape[1]); ax.set_zlim(0, vol.shape[2])
    ax.set_box_aspect(vol.shape)
    ax.set_axis_off()

    imgs = []
    for k in range(frames):
        ax.view_init(elev=elev, azim=k * 360.0 / frames)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        imgs.append(buf)
        if (k + 1) % 10 == 0 or k == frames - 1:
            print(f"  frame {k+1}/{frames}")
    plt.close(fig)

    imageio.mimsave(out, imgs, fps=fps, loop=0)
    print(f"[OK] GIF ecrit : {os.path.abspath(out)}  ({frames} frames, {fps} fps)")


def main():
    ap = argparse.ArgumentParser(description="GIF anime d'un rendu 3D N&B d'un volume IRM")
    ap.add_argument("path", nargs="?", default=None)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=18)
    ap.add_argument("--level", type=float, default=0.25, help="seuil de l'isosurface")
    ap.add_argument("--max-dim", type=int, default=80)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = args.path or find_default_volume()
    if not os.path.exists(path) and path.endswith(".gz"):
        path = path[:-3]
    if not os.path.exists(path):
        sys.exit(f"Introuvable: {path}")

    title = os.path.splitext(os.path.basename(path))[0]
    print(f"Chargement : {path}")
    vol = load_volume(path, max_dim=args.max_dim)
    print(f"  shape utilisee : {vol.shape}")

    out = args.out or f"rot3d_{title}.gif"
    render_gif(vol, out, frames=args.frames, fps=args.fps, level=args.level)


if __name__ == "__main__":
    main()
