"""Practical VAE + ViT volumetric JEPA pipeline for MRI retrieval.

The implementation follows the project PDF at an executable scale:
- train a slice VAE on dataset3 T2 volumes to create post-operative-like T2
  reconstructions;
- train a JEPA model on all labelled dataset1 T1/T2 pairs by default;
- encode every fifth axial slice with a ViT, aggregate slice tokens with an
  axial Transformer, and rank gallery images by latent JEPA prediction error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    resolved = path if path.is_absolute() else data_root / path
    if resolved.exists():
        return resolved
    if resolved.name.endswith(".nii.gz"):
        fallback = resolved.with_name(resolved.name[:-3])
        if fallback.exists():
            return fallback
    return resolved


def load_volume(path: Path) -> np.ndarray:
    volume = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = volume[volume > 0]
    if nonzero.size:
        low, high = np.percentile(nonzero, [1, 99])
        volume = np.clip(volume, low, high)
        volume = (volume - low) / max(float(high - low), 1e-6)
    return volume.astype(np.float32)


def every_n_slice_indices(depth: int, slice_step: int, max_slices: int) -> list[int]:
    indices = list(range(0, depth, slice_step))
    if not indices:
        indices = [depth // 2]
    if len(indices) > max_slices:
        indices = indices[:max_slices]
    return indices


def volume_to_slice_stack(path: Path, image_size: int, slice_step: int, max_slices: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a 3D MRI and return fixed [K, 1, H, W] slices plus [K] valid mask.

    Slices are sampled every `slice_step` planes on the axial axis. Short volumes
    are padded with zeros so the ViT/axial transformer can batch them.
    """
    volume = load_volume(path)
    indices = every_n_slice_indices(volume.shape[2], slice_step=slice_step, max_slices=max_slices)
    slices = torch.from_numpy(np.stack([volume[:, :, z] for z in indices])).unsqueeze(1)
    slices = F.interpolate(slices, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mask = torch.zeros(max_slices, dtype=torch.bool)
    mask[: len(indices)] = True
    if len(indices) < max_slices:
        padding = torch.zeros(max_slices - len(indices), 1, image_size, image_size, dtype=slices.dtype)
        slices = torch.cat([slices, padding], dim=0)
    return slices.float(), mask


class T2SliceDataset(Dataset):
    """Dataset3 T2 volumes sampled every N planes for VAE training."""

    def __init__(
        self,
        paths: list[Path],
        image_size: int,
        slice_step: int,
        max_slices: int,
        max_volumes: int | None = None,
    ) -> None:
        self.paths = paths[:max_volumes] if max_volumes else paths
        self.image_size = image_size
        self.slice_step = slice_step
        self.max_slices = max_slices

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        slices, mask = volume_to_slice_stack(
            self.paths[index],
            image_size=self.image_size,
            slice_step=self.slice_step,
            max_slices=self.max_slices,
        )
        return {"slices": slices, "mask": mask}


class PairVolumeDataset(Dataset):
    """Dataset1 labelled ceT1/T2 pairs with every-N-slice sampling."""

    def __init__(
        self,
        data_root: Path,
        pair_csv: Path,
        image_size: int,
        slice_step: int,
        max_slices: int,
        max_pairs: int | None = None,
    ) -> None:
        rows = read_csv(pair_csv)
        self.rows = rows[:max_pairs] if max_pairs else rows
        self.data_root = data_root
        self.image_size = image_size
        self.slice_step = slice_step
        self.max_slices = max_slices

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        query, query_mask = volume_to_slice_stack(
            resolve_path(self.data_root, row["query_image"]),
            image_size=self.image_size,
            slice_step=self.slice_step,
            max_slices=self.max_slices,
        )
        target, target_mask = volume_to_slice_stack(
            resolve_path(self.data_root, row["target_image"]),
            image_size=self.image_size,
            slice_step=self.slice_step,
            max_slices=self.max_slices,
        )
        return {
            "query": query,
            "query_mask": query_mask,
            "target": target,
            "target_mask": target_mask,
            "pair_id": row["pair_id"],
        }


def cached_tensor(payload: dict[str, object]) -> torch.Tensor:
    """Return cached [K, 1, H, W] data as float in [0, 1]."""
    tensor = torch.as_tensor(payload["tensor"])
    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0
    return tensor.float()


class CachedT2VolumeDataset(Dataset):
    """T2 volumes already converted to fixed slice stacks in a .pt cache."""

    def __init__(self, cache: dict[str, object], max_volumes: int | None = None) -> None:
        images = cache["images"]
        self.payloads = [
            payload
            for payload in images.values()
            if payload.get("dataset") == "dataset3" and "T2" in str(payload.get("modality", ""))
        ]
        self.payloads = sorted(self.payloads, key=lambda item: str(item["id"]))
        if max_volumes is not None:
            self.payloads = self.payloads[:max_volumes]
        if not self.payloads:
            raise ValueError("No dataset3 T2 volumes found in precomputed cache")

    def __len__(self) -> int:
        return len(self.payloads)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        payload = self.payloads[index]
        return {"slices": cached_tensor(payload), "mask": torch.as_tensor(payload["mask"]).bool()}


class CachedPairVolumeDataset(Dataset):
    """Dataset1 labelled pairs from a precomputed .pt cache."""

    def __init__(self, cache: dict[str, object], max_pairs: int | None = None) -> None:
        self.images = cache["images"]
        rows = cache["train_pairs"]
        self.rows = rows[:max_pairs] if max_pairs else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        query_payload = self.images[row["query_id"]]
        target_payload = self.images[row["target_id"]]
        return {
            "query": cached_tensor(query_payload),
            "query_mask": torch.as_tensor(query_payload["mask"]).bool(),
            "target": cached_tensor(target_payload),
            "target_mask": torch.as_tensor(target_payload["mask"]).bool(),
            "pair_id": row["pair_id"],
        }


class ConvVAE(nn.Module):
    """Small 2D VAE trained on dataset3 T2 slices for pseudo target generation."""

    def __init__(self, latent_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.logvar = nn.Linear(128, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden).clamp(-8, 8)

    def decode(self, z: torch.Tensor, image_size: int) -> torch.Tensor:
        x = self.decoder_fc(z).view(-1, 128, 8, 8)
        x = self.decoder(x)
        if x.shape[-1] != image_size:
            x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decode(z, x.shape[-1]), mu, logvar


class ViTSliceEncoder(nn.Module):
    """A compact ViT encoder for one 2D MRI slice."""

    def __init__(self, image_size: int, patch_size: int, token_dim: int, depth: int, heads: int) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.patch = nn.Conv2d(1, token_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls.expand(len(x), -1, -1)
        tokens = torch.cat([cls, patches], dim=1) + self.pos[:, : patches.shape[1] + 1]
        tokens = self.transformer(tokens)
        return self.norm(tokens[:, 0])


class VolumetricJEPA(nn.Module):
    """ViT per slice + axial Transformer JEPA model."""

    def __init__(
        self,
        image_size: int,
        max_slices: int,
        patch_size: int = 16,
        token_dim: int = 128,
        vit_depth: int = 2,
        axial_depth: int = 2,
        heads: int = 4,
        ema_decay: float = 0.99,
    ) -> None:
        super().__init__()
        self.ema_decay = ema_decay
        self.online_slice = ViTSliceEncoder(image_size, patch_size, token_dim, vit_depth, heads)
        self.target_slice = ViTSliceEncoder(image_size, patch_size, token_dim, vit_depth, heads)
        self.slice_pos = nn.Parameter(torch.randn(1, max_slices, token_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.online_axial = nn.TransformerEncoder(layer, num_layers=axial_depth)
        target_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=token_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.target_axial = nn.TransformerEncoder(target_layer, num_layers=axial_depth)
        self.predictor = nn.Sequential(nn.LayerNorm(token_dim), nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, token_dim))
        self.reset_target_encoder()

    def reset_target_encoder(self) -> None:
        self.target_slice.load_state_dict(self.online_slice.state_dict())
        self.target_axial.load_state_dict(self.online_axial.state_dict())
        for parameter in self.target_slice.parameters():
            parameter.requires_grad = False
        for parameter in self.target_axial.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_ema(self) -> None:
        for online, target in zip(self.online_slice.parameters(), self.target_slice.parameters()):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)
        for online, target in zip(self.online_axial.parameters(), self.target_axial.parameters()):
            target.data.mul_(self.ema_decay).add_(online.data, alpha=1.0 - self.ema_decay)

    def encode_online_tokens(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slices, channels, height, width = volume.shape
        tokens = self.online_slice(volume.reshape(batch * slices, channels, height, width)).view(batch, slices, -1)
        tokens = tokens + self.slice_pos[:, :slices]
        return self.online_axial(tokens, src_key_padding_mask=~mask)

    @torch.no_grad()
    def encode_target_tokens(self, volume: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, slices, channels, height, width = volume.shape
        tokens = self.target_slice(volume.reshape(batch * slices, channels, height, width)).view(batch, slices, -1)
        tokens = tokens + self.slice_pos[:, :slices]
        return self.target_axial(tokens, src_key_padding_mask=~mask)

    def predict_target_tokens(self, query_volume: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.encode_online_tokens(query_volume, query_mask))

    @staticmethod
    def masked_global(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.float().unsqueeze(-1)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    denominator = weights.sum().clamp_min(1.0) * pred.shape[-1]
    return ((pred - target).pow(2) * weights).sum() / denominator


def vicreg_loss(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    invariance = F.mse_loss(x, y)
    if x.shape[0] < 2:
        return 25.0 * invariance
    std_x = torch.sqrt(x.var(dim=0) + 1e-4)
    std_y = torch.sqrt(y.var(dim=0) + 1e-4)
    variance = torch.mean(F.relu(gamma - std_x)) + torch.mean(F.relu(gamma - std_y))
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.shape[0] - 1)
    cov_y = (y.T @ y) / (y.shape[0] - 1)
    off_x = cov_x.flatten()[:-1].view(cov_x.shape[0] - 1, cov_x.shape[0] + 1)[:, 1:].flatten()
    off_y = cov_y.flatten()[:-1].view(cov_y.shape[0] - 1, cov_y.shape[0] + 1)[:, 1:].flatten()
    return 25.0 * invariance + 25.0 * variance + off_x.pow(2).mean() + off_y.pow(2).mean()


def train_vae(vae: ConvVAE, dataset: Dataset, device: torch.device, epochs: int, batch_size: int, lr: float) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    vae.train()
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for batch in loader:
            if isinstance(batch, dict):
                batch = batch["slices"][batch["mask"]]
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 1e-4 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(batch)
            count += len(batch)
        print(f"vae epoch {epoch + 1}/{epochs} loss={total / max(count, 1):.5f}")


def train_jepa(model: VolumetricJEPA, vae: ConvVAE, dataset: PairVolumeDataset, device: torch.device, epochs: int, batch_size: int, lr: float) -> None:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    vae.eval()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for batch in loader:
            query = batch["query"].to(device)
            query_mask = batch["query_mask"].to(device)
            target = batch["target"].to(device)
            target_mask = batch["target_mask"].to(device)
            valid_mask = query_mask & target_mask
            b, k, c, h, w = target.shape
            with torch.no_grad():
                pseudo_target, _, _ = vae(target.reshape(b * k, c, h, w))
                pseudo_target = pseudo_target.view(b, k, c, h, w)

            pred = model.predict_target_tokens(query, query_mask)
            target_tokens = model.encode_target_tokens(target, target_mask)
            pseudo_tokens = model.encode_target_tokens(pseudo_target, target_mask)
            jepa = masked_mse(pred, target_tokens, valid_mask) + 0.5 * masked_mse(pred, pseudo_tokens, valid_mask)
            query_global = model.masked_global(model.encode_online_tokens(query, query_mask), query_mask)
            target_global = model.masked_global(target_tokens, target_mask)
            retrieval = torch.tensor(0.0, device=device)
            if len(query_global) > 1:
                logits = query_global @ target_global.T / 0.1
                labels = torch.arange(len(query_global), device=device)
                retrieval = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            vic = vicreg_loss(query_global, target_global)
            loss = jepa + 0.1 * retrieval + 0.01 * vic
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.update_ema()
            total += float(loss.item()) * len(query)
            count += len(query)
        print(f"jepa epoch {epoch + 1}/{epochs} loss={total / max(count, 1):.5f}")


@torch.no_grad()
def feature_query(model: VolumetricJEPA, image_path: Path, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    volume, mask = volume_to_slice_stack(image_path, args.image_size, args.slice_step, args.max_slices)
    volume = volume.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    return model.predict_target_tokens(volume, mask).squeeze(0).cpu(), mask.squeeze(0).cpu()


@torch.no_grad()
def feature_target(model: VolumetricJEPA, image_path: Path, args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    volume, mask = volume_to_slice_stack(image_path, args.image_size, args.slice_step, args.max_slices)
    volume = volume.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    return model.encode_target_tokens(volume, mask).squeeze(0).cpu(), mask.squeeze(0).cpu()


@torch.no_grad()
def cached_feature(
    model: VolumetricJEPA,
    payload: dict[str, object],
    encoder: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    volume = cached_tensor(payload).unsqueeze(0).to(device)
    mask = torch.as_tensor(payload["mask"]).bool().unsqueeze(0).to(device)
    if encoder == "query":
        tokens = model.predict_target_tokens(volume, mask)
    elif encoder == "target":
        tokens = model.encode_target_tokens(volume, mask)
    else:
        raise ValueError(f"Unknown encoder: {encoder}")
    return tokens.squeeze(0).cpu(), mask.squeeze(0).cpu()


def energy(query: tuple[torch.Tensor, torch.Tensor], target: tuple[torch.Tensor, torch.Tensor], window: int = 1, trim: float = 1.0) -> float:
    query_tokens, query_mask = query
    target_tokens, target_mask = target
    q_valid = torch.nonzero(query_mask, as_tuple=False).flatten()
    t_valid = torch.nonzero(target_mask, as_tuple=False).flatten()
    if len(q_valid) == 0 or len(t_valid) == 0:
        return float("inf")
    errors = []
    for q_order, q_index in enumerate(q_valid):
        if len(q_valid) == 1:
            target_center = 0
        else:
            target_center = round(q_order * (len(t_valid) - 1) / (len(q_valid) - 1))
        start = max(0, target_center - window)
        stop = min(len(t_valid), target_center + window + 1)
        candidate_indices = t_valid[start:stop]
        dist = (target_tokens[candidate_indices] - query_tokens[q_index].unsqueeze(0)).pow(2).mean(dim=1)
        errors.append(dist.min())
    error_tensor = torch.stack(errors)
    if trim < 1.0:
        keep = max(1, int(math.ceil(len(error_tensor) * trim)))
        error_tensor = torch.topk(error_tensor, k=keep, largest=False).values
    return float(error_tensor.mean().item())


def make_submission(model: VolumetricJEPA, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    rows = []
    for query_csv, gallery_csv in zip(args.query_csv, args.gallery_csv):
        query_df = pd.read_csv(query_csv)
        gallery_df = pd.read_csv(gallery_csv)
        target_features = {
            row.target_id: feature_target(model, resolve_path(args.data_root, row.target_image), args, device)
            for row in gallery_df.itertuples(index=False)
        }
        for query_row in query_df.itertuples(index=False):
            q_features = feature_query(model, resolve_path(args.data_root, query_row.query_image), args, device)
            scores = [
                (target_id, energy(q_features, t_features, window=args.score_window, trim=args.trim_fraction))
                for target_id, t_features in target_features.items()
            ]
            rows.append({"query_id": query_row.query_id, "target_id_ranking": " ".join([target_id for target_id, _ in sorted(scores, key=lambda item: item[1])])})
    submission = pd.DataFrame(rows)
    if args.sample_submission and args.sample_submission.exists():
        sample = pd.read_csv(args.sample_submission)
        submission = sample[["query_id"]].merge(submission, on="query_id", how="left")
        if submission["target_id_ranking"].isna().any():
            raise ValueError("Submission missing rows from sample template")
    submission.to_csv(args.out, index=False)
    return submission


def make_submission_from_cache(
    model: VolumetricJEPA,
    cache: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    images = cache["images"]
    for prediction_set in cache["prediction_sets"]:
        target_features = {
            target_id: cached_feature(model, images[target_id], encoder="target", device=device)
            for target_id in prediction_set["target_ids"]
        }
        for query_id in prediction_set["query_ids"]:
            q_features = cached_feature(model, images[query_id], encoder="query", device=device)
            scores = [
                (target_id, energy(q_features, t_features, window=args.score_window, trim=args.trim_fraction))
                for target_id, t_features in target_features.items()
            ]
            rows.append(
                {
                    "query_id": query_id,
                    "target_id_ranking": " ".join(
                        [target_id for target_id, _ in sorted(scores, key=lambda item: item[1])]
                    ),
                }
            )
    submission = pd.DataFrame(rows)
    sample_query_order = cache.get("sample_query_order")
    if sample_query_order:
        sample = pd.DataFrame({"query_id": sample_query_order})
        submission = sample.merge(submission, on="query_id", how="left")
        if submission["target_id_ranking"].isna().any():
            raise ValueError("Cached submission missing rows from sample template")
    elif args.sample_submission and args.sample_submission.exists():
        sample = pd.read_csv(args.sample_submission)
        submission = sample[["query_id"]].merge(submission, on="query_id", how="left")
        if submission["target_id_ranking"].isna().any():
            raise ValueError("Submission missing rows from sample template")
    submission.to_csv(args.out, index=False)
    return submission


def dataset3_t2_paths(data_root: Path, include_test: bool) -> list[Path]:
    csvs = [data_root / "dataset3" / "val_gallery.csv"]
    if include_test:
        csvs.append(data_root / "dataset3" / "test_gallery.csv")
    return [resolve_path(data_root, row["target_image"]) for csv_path in csvs for row in read_csv(csv_path)]


def save_artifacts(save_dir: Path, vae: ConvVAE, model: VolumetricJEPA, args: argparse.Namespace) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": vae.state_dict(), "latent_dim": args.vae_latent_dim, "image_size": args.image_size}, save_dir / "vae.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": args.image_size,
            "max_slices": args.max_slices,
            "patch_size": args.patch_size,
            "token_dim": args.token_dim,
            "vit_depth": args.vit_depth,
            "axial_depth": args.axial_depth,
            "heads": args.heads,
        },
        save_dir / "jepa.pt",
    )
    with (save_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2, default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VAE + ViT volumetric JEPA submission pipeline.")
    parser.add_argument("--data-root", type=Path, default=Path("data/ehl-paris-medical-image-retrieval"))
    parser.add_argument("--precomputed-pt", type=Path, default=None, help="Compressed tensor cache built by build_volumetric_cache_pt.py.")
    parser.add_argument("--out", type=Path, default=Path("vae_jepa_submission.csv"))
    parser.add_argument("--sample-submission", type=Path, default=None)
    parser.add_argument("--query-csv", type=Path, action="append", default=[])
    parser.add_argument("--gallery-csv", type=Path, action="append", default=[])
    parser.add_argument("--train-pairs-csv", type=Path, default=None)
    parser.add_argument("--include-dataset3-test-for-vae", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--slice-step", type=int, default=5)
    parser.add_argument("--max-slices", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--token-dim", type=int, default=128)
    parser.add_argument("--vae-latent-dim", type=int, default=128)
    parser.add_argument("--vit-depth", type=int, default=2)
    parser.add_argument("--axial-depth", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vae-epochs", type=int, default=5)
    parser.add_argument("--jepa-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-vae-volumes", type=int, default=None)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--score-window", type=int, default=1)
    parser.add_argument("--trim-fraction", type=float, default=0.85)
    parser.add_argument("--save-dir", type=Path, default=Path("artifacts/vae_jepa"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"))
    print("device", device)

    cache = None
    if args.precomputed_pt is not None:
        print(f"loading precomputed cache: {args.precomputed_pt}")
        cache = torch.load(args.precomputed_pt, map_location="cpu", weights_only=False)
        cache_config = cache.get("config", {})
        args.image_size = int(cache_config.get("image_size", args.image_size))
        args.slice_step = int(cache_config.get("slice_step", args.slice_step))
        args.max_slices = int(cache_config.get("max_slices", args.max_slices))
        print("cache config", cache_config)
    elif args.train_pairs_csv is None:
        args.train_pairs_csv = args.data_root / "dataset1" / "train_pairs.csv"
    if cache is None and (not args.query_csv or not args.gallery_csv):
        raise ValueError("Pass --query-csv/--gallery-csv or use --precomputed-pt with cached prediction sets")
    if len(args.query_csv) != len(args.gallery_csv):
        raise ValueError("--query-csv and --gallery-csv must have the same count")

    vae = ConvVAE(latent_dim=args.vae_latent_dim).to(device)
    if cache is None:
        vae_paths = dataset3_t2_paths(args.data_root, include_test=args.include_dataset3_test_for_vae)
        vae_dataset: Dataset = T2SliceDataset(
            vae_paths,
            image_size=args.image_size,
            slice_step=args.slice_step,
            max_slices=args.max_slices,
            max_volumes=args.max_vae_volumes,
        )
    else:
        vae_dataset = CachedT2VolumeDataset(cache, max_volumes=args.max_vae_volumes)
    train_vae(
        vae,
        vae_dataset,
        device=device,
        epochs=args.vae_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    model = VolumetricJEPA(
        image_size=args.image_size,
        max_slices=args.max_slices,
        patch_size=args.patch_size,
        token_dim=args.token_dim,
        vit_depth=args.vit_depth,
        axial_depth=args.axial_depth,
        heads=args.heads,
    ).to(device)
    if cache is None:
        pair_dataset: Dataset = PairVolumeDataset(
            args.data_root,
            args.train_pairs_csv,
            args.image_size,
            args.slice_step,
            args.max_slices,
            args.max_train_pairs,
        )
    else:
        pair_dataset = CachedPairVolumeDataset(cache, max_pairs=args.max_train_pairs)
    train_jepa(
        model,
        vae,
        pair_dataset,
        device=device,
        epochs=args.jepa_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    save_artifacts(args.save_dir, vae, model, args)
    submission = make_submission_from_cache(model, cache, args, device) if cache is not None else make_submission(model, args, device)
    print(f"wrote {len(submission)} rows to {args.out}")
    print(f"artifacts saved to {args.save_dir}")


if __name__ == "__main__":
    main()
