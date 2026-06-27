"""Robust IoU scoring for MRI volumetric retrieval.

Turns cached [K, 1, H, W] axial slice stacks into per-slice anatomical masks and
scores a query/gallery pair with a windowed (handles geometric drift in
dataset2), trimmed (ignores operated/missing regions in dataset3), optionally
soft (continuous instead of hard-thresholded), center-weighted IoU. Used as an
auxiliary anatomical signal alongside a latent JEPA distance, not as the sole
retrieval score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def slice_masks(slices: torch.Tensor, valid: torch.Tensor, threshold_quantile: float = 0.60) -> torch.Tensor:
    """[K,1,H,W] float in [0,1] -> [K,H,W] bool anatomical mask, per-slice quantile threshold."""
    masks = torch.zeros(slices.shape[0], slices.shape[-2], slices.shape[-1], dtype=torch.bool)
    for k in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        flat = slices[k, 0]
        foreground = flat[flat > 0]
        if foreground.numel() == 0:
            continue
        threshold = torch.quantile(foreground, threshold_quantile)
        masks[k] = flat > threshold
    return masks


def soft_slice_masks(
    slices: torch.Tensor, valid: torch.Tensor, threshold_quantile: float = 0.60, temperature: float = 0.05
) -> torch.Tensor:
    """Soft probabilistic counterpart of slice_masks: sigmoid around the threshold."""
    probs = torch.zeros(slices.shape[0], slices.shape[-2], slices.shape[-1])
    for k in torch.nonzero(valid, as_tuple=False).flatten().tolist():
        flat = slices[k, 0]
        foreground = flat[flat > 0]
        if foreground.numel() == 0:
            continue
        threshold = torch.quantile(foreground, threshold_quantile)
        probs[k] = torch.sigmoid((flat - threshold) / max(temperature, 1e-6))
    return probs


def center_weights(valid: torch.Tensor) -> torch.Tensor:
    """Gaussian weights peaking at the central valid slice; zero on invalid slices.

    Central slices carry more anatomical signal than the near-empty extremes,
    so they should dominate the average rather than contribute equally.
    """
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    weights = torch.zeros(valid.shape[0])
    if len(indices) == 0:
        return weights
    center = indices.float().mean()
    spread = max(float(len(indices)) / 2.0, 1.0)
    weights[indices] = torch.exp(-0.5 * ((indices.float() - center) / spread) ** 2)
    return weights


def _hard_iou(a: torch.Tensor, b: torch.Tensor) -> float:
    intersection = (a & b).sum().float()
    union = (a | b).sum().float().clamp_min(1.0)
    return float((intersection / union).item())


def _soft_iou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    intersection = (a * b).sum()
    union = a.sum() + b.sum() - intersection
    return float((intersection / union.clamp_min(eps)).item())


@dataclass
class RobustIoUConfig:
    threshold_quantile: float = 0.60
    window: int = 0  # +/- slice window for matching (handles geometric drift, e.g. dataset2)
    trim_fraction: float = 1.0  # keep this fraction of best-matching slices (handles dataset3 operated regions)
    soft: bool = False
    center_weighted: bool = True


DATASET_PRESETS: dict[str, RobustIoUConfig] = {
    "dataset1": RobustIoUConfig(window=0, trim_fraction=1.0),
    "dataset2": RobustIoUConfig(window=2, trim_fraction=1.0),
    "dataset3": RobustIoUConfig(window=1, trim_fraction=0.70, soft=True),
}


def config_for_dataset(dataset_name: str) -> RobustIoUConfig:
    return DATASET_PRESETS.get(dataset_name, RobustIoUConfig())


def masks_for_config(slices: torch.Tensor, valid: torch.Tensor, config: RobustIoUConfig) -> torch.Tensor:
    """Compute the per-slice mask representation (hard or soft) used by `config`.

    Split out from `robust_iou_score` so callers scoring many query/gallery
    pairs against the same volume can compute each volume's masks once and
    reuse them, instead of re-thresholding on every pair.
    """
    if config.soft:
        return soft_slice_masks(slices, valid, config.threshold_quantile)
    return slice_masks(slices, valid, config.threshold_quantile)


def robust_iou_from_masks(
    query_masks: torch.Tensor,
    query_valid: torch.Tensor,
    gallery_masks: torch.Tensor,
    gallery_valid: torch.Tensor,
    config: RobustIoUConfig,
) -> float:
    """Windowed + trimmed + center-weighted IoU given precomputed per-slice masks.

    Slice k of the query is matched against a small window around the
    proportionally-corresponding slice of the gallery (rather than the same
    index), since the two volumes can have different valid-slice counts and
    geometric offsets. The best-matching window IoU per query slice is then
    center-weighted and optionally trimmed to the best-matching fraction.
    """
    q_valid_idx = torch.nonzero(query_valid, as_tuple=False).flatten()
    g_valid_idx = torch.nonzero(gallery_valid, as_tuple=False).flatten()
    if len(q_valid_idx) == 0 or len(g_valid_idx) == 0:
        return 0.0

    iou_fn = _soft_iou if config.soft else _hard_iou
    q_masks, g_masks = query_masks, gallery_masks
    weights = center_weights(query_valid) if config.center_weighted else torch.ones(query_valid.shape[0])

    per_slice_scores = []
    per_slice_weights = []
    for order, q_index in enumerate(q_valid_idx.tolist()):
        if len(q_valid_idx) == 1:
            center = 0
        else:
            center = round(order * (len(g_valid_idx) - 1) / (len(q_valid_idx) - 1))
        start = max(0, center - config.window)
        stop = min(len(g_valid_idx), center + config.window + 1)
        candidates = g_valid_idx[start:stop].tolist()
        best = max(iou_fn(q_masks[q_index], g_masks[g_index]) for g_index in candidates)
        per_slice_scores.append(best)
        per_slice_weights.append(float(weights[q_index].item()))

    scores = torch.tensor(per_slice_scores)
    slice_weights = torch.tensor(per_slice_weights)

    if config.trim_fraction < 1.0:
        keep = max(1, math.ceil(len(scores) * config.trim_fraction))
        top_indices = torch.topk(scores, k=keep, largest=True).indices
        scores = scores[top_indices]
        slice_weights = slice_weights[top_indices]

    weight_sum = slice_weights.sum().clamp_min(1e-6)
    return float((scores * slice_weights).sum().item() / weight_sum.item())


def robust_iou_score(
    query_slices: torch.Tensor,
    query_valid: torch.Tensor,
    gallery_slices: torch.Tensor,
    gallery_valid: torch.Tensor,
    config: RobustIoUConfig,
) -> float:
    """Convenience one-shot scorer: compute masks then score a single pair.

    For ranking one query against many gallery items, call `masks_for_config`
    once per image and reuse the result via `robust_iou_from_masks` instead.
    """
    query_masks = masks_for_config(query_slices, query_valid, config)
    gallery_masks = masks_for_config(gallery_slices, gallery_valid, config)
    return robust_iou_from_masks(query_masks, query_valid, gallery_masks, gallery_valid, config)
