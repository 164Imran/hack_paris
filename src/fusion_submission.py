"""Rank-fusion submission: JEPA latent energy + global cosine + robust IoU.

Score(q, g) = alpha * E_JEPA(q, g) + beta * (1 - cos(h_q, h_g)) + gamma * (1 - IoU_robust(q, g))

Lower score wins. IoU is never used alone: it is z-normalized per query and
combined with the trained JEPA latent distance, matching the principle that
anatomical mask overlap is an auxiliary ranking feature, not the primary
retrieval signal (raw intensities differ across T1/T2, and IoU alone can't
distinguish two anatomically similar patients).

Requires a precomputed cache (build_volumetric_cache_pt.py) and a trained
checkpoint produced by volumetric_jepa_vae_submission.py's --save-dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from robust_iou import RobustIoUConfig, config_for_dataset, masks_for_config, robust_iou_from_masks
from volumetric_jepa_vae_submission import VolumetricJEPA, cached_tensor, energy


def load_model(checkpoint_path: Path, device: torch.device) -> VolumetricJEPA:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VolumetricJEPA(
        image_size=checkpoint["image_size"],
        max_slices=checkpoint["max_slices"],
        patch_size=checkpoint["patch_size"],
        token_dim=checkpoint["token_dim"],
        vit_depth=checkpoint["vit_depth"],
        axial_depth=checkpoint["axial_depth"],
        heads=checkpoint["heads"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def zscore(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def dataset_name_from_csv(csv_path: str) -> str:
    return Path(csv_path).parent.name


@torch.no_grad()
def image_iou_masks(images: dict[str, dict[str, object]], image_id: str, config: RobustIoUConfig) -> torch.Tensor:
    valid = torch.as_tensor(images[image_id]["mask"]).bool()
    return masks_for_config(cached_tensor(images[image_id]), valid, config)


@torch.no_grad()
def image_jepa_features(
    model: VolumetricJEPA, images: dict[str, dict[str, object]], image_id: str, role: str, device: torch.device
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return (tokens, valid_mask) for `energy()` plus the masked-global pooled embedding."""
    slices = cached_tensor(images[image_id]).unsqueeze(0).to(device)
    valid = torch.as_tensor(images[image_id]["mask"]).bool().unsqueeze(0).to(device)
    if role == "query":
        tokens = model.predict_target_tokens(slices, valid)
        global_embed = model.masked_global(model.encode_online_tokens(slices, valid), valid)
    else:
        tokens = model.encode_target_tokens(slices, valid)
        global_embed = model.masked_global(tokens, valid)
    return (tokens.squeeze(0).cpu(), valid.squeeze(0).cpu()), global_embed.squeeze(0).cpu()


@torch.no_grad()
def fuse_prediction_set(
    model: VolumetricJEPA,
    images: dict[str, dict[str, object]],
    query_ids: list[str],
    target_ids: list[str],
    dataset_name: str,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, str]]:
    iou_config = config_for_dataset(dataset_name)

    target_jepa = {target_id: image_jepa_features(model, images, target_id, "target", device) for target_id in target_ids}
    target_iou_masks = {target_id: image_iou_masks(images, target_id, iou_config) for target_id in target_ids}
    target_valid = {target_id: torch.as_tensor(images[target_id]["mask"]).bool() for target_id in target_ids}

    rows = []
    for query_index, query_id in enumerate(query_ids, start=1):
        (query_tokens, query_valid_tokens), query_global = image_jepa_features(model, images, query_id, "query", device)
        query_iou_masks = image_iou_masks(images, query_id, iou_config)
        query_valid = torch.as_tensor(images[query_id]["mask"]).bool()

        jepa_scores, cos_scores, iou_scores = [], [], []
        for target_id in target_ids:
            target_tokens, target_valid_tokens = target_jepa[target_id][0]
            jepa_scores.append(
                energy((query_tokens, query_valid_tokens), (target_tokens, target_valid_tokens), window=args.score_window, trim=args.trim_fraction)
            )
            cos_scores.append(float(torch.dot(query_global, target_jepa[target_id][1]).item()))
            iou_scores.append(
                robust_iou_from_masks(query_iou_masks, query_valid, target_iou_masks[target_id], target_valid[target_id], iou_config)
            )

        jepa_z = zscore(torch.tensor(jepa_scores))
        cos_dist_z = zscore(1.0 - torch.tensor(cos_scores))
        iou_dist_z = zscore(1.0 - torch.tensor(iou_scores))
        fused = args.alpha * jepa_z + args.beta * cos_dist_z + args.gamma * iou_dist_z

        ranking = [target_ids[i] for i in torch.argsort(fused).tolist()]
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(ranking)})
        if query_index == 1 or query_index % 10 == 0 or query_index == len(query_ids):
            print(f"  [{dataset_name}] query {query_index}/{len(query_ids)}")
    return rows


def make_fusion_submission(args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    cache = torch.load(args.precomputed_pt, map_location="cpu", weights_only=False)
    model = load_model(args.jepa_checkpoint, device)
    images = cache["images"]

    rows: list[dict[str, str]] = []
    for prediction_set in cache["prediction_sets"]:
        dataset_name = dataset_name_from_csv(prediction_set["gallery_csv"])
        print(f"[{dataset_name}] {len(prediction_set['query_ids'])} queries x {len(prediction_set['target_ids'])} gallery")
        rows.extend(
            fuse_prediction_set(
                model,
                images,
                prediction_set["query_ids"],
                prediction_set["target_ids"],
                dataset_name,
                args,
                device,
            )
        )

    submission = pd.DataFrame(rows)
    sample_query_order = cache.get("sample_query_order")
    if sample_query_order:
        sample = pd.DataFrame({"query_id": sample_query_order})
        submission = sample.merge(submission, on="query_id", how="left")
        if submission["target_id_ranking"].isna().any():
            raise ValueError("Fusion submission missing rows from sample template")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank-fusion submission: JEPA energy + cosine + robust IoU.")
    parser.add_argument("--precomputed-pt", type=Path, required=True)
    parser.add_argument("--jepa-checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/fusion_submission.csv"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--score-window", type=int, default=1, help="JEPA energy slice-matching window.")
    parser.add_argument("--trim-fraction", type=float, default=0.85, help="JEPA energy trim fraction.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight on JEPA latent energy.")
    parser.add_argument("--beta", type=float, default=0.2, help="Weight on global cosine distance.")
    parser.add_argument("--gamma", type=float, default=0.3, help="Weight on robust IoU distance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if has_mps else "cpu"))
    submission = make_fusion_submission(args, device)
    print(f"wrote {len(submission)} rows to {args.out}")
    print(f"device: {device}")


if __name__ == "__main__":
    main()
