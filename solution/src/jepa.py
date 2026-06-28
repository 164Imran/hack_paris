# -*- coding: utf-8 -*-
"""
Reconstruction de MaskedSliceJEPA + utilitaires (les modules src/ d'origine ont ete supprimes).

API utilisee par le script de continue-training :
  - MaskedSliceJEPA(image_size, patch_size, token_dim, encoder_depth, predictor_depth,
                    heads, mask_ratio, ema_decay)
        .image_size, .mask_ratio, .ema_decay
        .patch_embed.proj (Conv2d), .pos_embed, .predictor_pos, .mask_token
        .context_encoder.layers, .predictor.layers
        .encode_context(x), .encode_target(x), forward(t1, t2), .update_ema()
  - build_axial_slice_cache(csv_path, data_root, preprocessor, axes) -> list[dict]
  - AugmentedJepaPairDataset(cache, image_size, preprocessor, augment=True)
  - jepa_loss(output, variance_weight, retrieval_weight) -> dict[str, Tensor]

C'est une implementation JEPA cross-modale minimale mais coherente (entrainable de zero).
L'architecture exacte d'origine et son checkpoint n'existant plus, ce code reproduit l'interface
et un comportement raisonnable, pas les poids initiaux.
"""
from __future__ import annotations

import copy
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


# ============================================================ modele

class PatchEmbed(nn.Module):
    def __init__(self, token_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, token_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # x: (B,1,H,W)
        z = self.proj(x)                                  # (B,D,h,w)
        return z.flatten(2).transpose(1, 2)               # (B,N,D)


def _encoder(token_dim: int, depth: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=token_dim, nhead=heads, dim_feedforward=token_dim * 4,
        dropout=0.0, batch_first=True, activation="gelu",
    )
    return nn.TransformerEncoder(layer, num_layers=depth)


class MaskedSliceJEPA(nn.Module):
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        token_dim: int = 128,
        encoder_depth: int = 4,
        predictor_depth: int = 2,
        heads: int = 4,
        mask_ratio: float = 0.5,
        ema_decay: float = 0.996,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.num_patches = (image_size // patch_size) ** 2

        # --- voie contexte (T1) ---
        self.patch_embed = PatchEmbed(token_dim, patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, token_dim) * 0.02)
        self.context_encoder = _encoder(token_dim, encoder_depth, heads)

        # --- predicteur ---
        self.predictor = _encoder(token_dim, predictor_depth, heads)
        self.predictor_pos = nn.Parameter(torch.randn(1, self.num_patches, token_dim) * 0.02)
        self.mask_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # --- voie cible (T2), copie EMA, sans gradient ---
        self.target_patch_embed = copy.deepcopy(self.patch_embed)
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for p in self.target_patch_embed.parameters():
            p.requires_grad_(False)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    # ---------------------------------------------------- encodeurs
    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.patch_embed(x) + self.pos_embed
        return self.context_encoder(z)

    @torch.no_grad()
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.target_patch_embed(x) + self.pos_embed
        return self.target_encoder(z)

    # ---------------------------------------------------- EMA
    @torch.no_grad()
    def update_ema(self) -> None:
        d = self.ema_decay
        for ema, src in zip(self.target_patch_embed.parameters(), self.patch_embed.parameters()):
            ema.mul_(d).add_(src.detach(), alpha=1.0 - d)
        for ema, src in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            ema.mul_(d).add_(src.detach(), alpha=1.0 - d)

    # ---------------------------------------------------- forward
    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> dict[str, torch.Tensor]:
        ctx = self.encode_context(t1)                  # (B,N,D)
        tgt = self.encode_target(t2)                   # (B,N,D) sans grad
        B, N, _ = ctx.shape

        z1 = ctx.mean(dim=1)                            # embedding T1 (B,D)
        z2 = tgt.mean(dim=1)                            # embedding T2 (B,D)

        n_mask = max(1, int(round(N * self.mask_ratio)))
        perm = torch.randperm(N, device=ctx.device)
        mask_idx = perm[:n_mask]

        pred_in = ctx.clone()
        pred_in[:, mask_idx, :] = self.mask_token.to(ctx.dtype)
        pred_in = pred_in + self.predictor_pos
        pred = self.predictor(pred_in)[:, mask_idx, :]  # (B,n_mask,D)
        target = tgt[:, mask_idx, :]                    # (B,n_mask,D)

        return {"prediction": pred, "target": target, "z1": z1, "z2": z2}


# ============================================================ pertes

def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def _variance(z: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.relu(1.0 - std).mean()


def jepa_loss(output: dict[str, torch.Tensor], variance_weight: float = 0.05,
              retrieval_weight: float = 1.0) -> dict[str, torch.Tensor]:
    prediction_loss = F.mse_loss(output["prediction"], output["target"])
    z1, z2 = output["z1"], output["z2"]
    if z1.size(0) > 1:
        retrieval_loss = _info_nce(z1, z2)
        variance_loss = 0.5 * (_variance(z1) + _variance(z2))
    else:
        retrieval_loss = prediction_loss.new_zeros(())
        variance_loss = prediction_loss.new_zeros(())
    loss = prediction_loss + variance_weight * variance_loss + retrieval_weight * retrieval_loss
    return {
        "loss": loss,
        "prediction_loss": prediction_loss,
        "variance_loss": variance_loss,
        "retrieval_loss": retrieval_loss,
    }


# ============================================================ donnees

def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def _extract_axis_slices(volume: torch.Tensor, axis: int, preprocessor) -> torch.Tensor:
    moved = torch.movedim(volume, axis - 1, 0)
    indices = preprocessor.select_slice_indices(moved.shape[0])
    return moved[indices].float().contiguous()          # (S,H,W)


def build_axial_slice_cache(csv_path: Path, data_root: Path, preprocessor,
                            axes: tuple[int, ...] = (1, 2, 3)) -> list[dict]:
    """Pour chaque paire (train_pairs.csv), pre-extrait les coupes T1/T2 sur chaque axe."""
    from preprocessing import MRIPairPreprocessor
    rows = _read_csv(csv_path)
    cache: list[dict] = []
    for row in rows:
        qpath = MRIPairPreprocessor.resolve_path(data_root, row["query_image"])
        tpath = MRIPairPreprocessor.resolve_path(data_root, row["target_image"])
        qvol, _ = preprocessor.load_volume(qpath)
        tvol, _ = preprocessor.load_volume(tpath)
        cache.append({
            "t1": {axis: _extract_axis_slices(qvol, axis, preprocessor) for axis in axes},
            "t2": {axis: _extract_axis_slices(tvol, axis, preprocessor) for axis in axes},
        })
    return cache


class AugmentedJepaPairDataset(Dataset):
    """Aplatit le cache en paires de coupes (t1, t2) et applique une augmentation legere."""

    def __init__(self, cache: list[dict], image_size: int, preprocessor, augment: bool = True) -> None:
        self.cache = cache
        self.image_size = image_size
        self.preprocessor = preprocessor
        self.augment = augment
        self.index: list[tuple[int, int, int]] = []
        for ci, entry in enumerate(cache):
            for axis, slices in entry["t1"].items():
                for si in range(slices.shape[0]):
                    self.index.append((ci, axis, si))

    def __len__(self) -> int:
        return len(self.index)

    def _prep(self, sl: torch.Tensor, flip_h: bool, flip_v: bool, noise: float) -> torch.Tensor:
        x = sl.unsqueeze(0).unsqueeze(0)                                  # (1,1,H,W)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = x.squeeze(0)                                                  # (1,H,W)
        if flip_h:
            x = torch.flip(x, dims=[-1])
        if flip_v:
            x = torch.flip(x, dims=[-2])
        if noise > 0:
            x = (x + noise * torch.randn_like(x)).clamp_(0.0, 1.0)
        return x

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ci, axis, si = self.index[i]
        entry = self.cache[ci]
        t1 = entry["t1"][axis][si]
        t2 = entry["t2"][axis][si]
        if self.augment:
            flip_h = bool(torch.rand(()) < 0.5)
            flip_v = bool(torch.rand(()) < 0.5)
            noise = 0.02
        else:
            flip_h = flip_v = False
            noise = 0.0
        return {"t1": self._prep(t1, flip_h, flip_v, noise),
                "t2": self._prep(t2, flip_h, flip_v, noise)}
