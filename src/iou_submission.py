from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


class MRIIoUSubmission:
    """IoU-style MRI retrieval baseline.

    The method turns each MRI volume into fixed-size binary projection masks and
    ranks gallery images by average IoU with each query. It uses torch tensors,
    so on AMD ROCm builds it can run on the GPU through torch's cuda API.
    """

    def __init__(
        self,
        data_root: str | Path,
        output_csv: str | Path = "iou_submission.csv",
        device: str | None = None,
        projection_size: int = 128,
        threshold_quantile: float = 0.60,
        batch_size: int = 64,
    ) -> None:
        self.data_root = Path(data_root)
        self.output_csv = Path(output_csv)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.projection_size = projection_size
        self.threshold_quantile = threshold_quantile
        self.batch_size = batch_size
        self.feature_cache: dict[str, torch.Tensor] = {}

    def build_submission(
        self,
        query_csvs: list[str | Path],
        gallery_csvs: list[str | Path],
    ) -> pd.DataFrame:
        """Rank every query against the matching same-split gallery."""
        if len(query_csvs) != len(gallery_csvs):
            raise ValueError("query_csvs and gallery_csvs must have the same length")

        rows: list[dict[str, str]] = []
        for query_csv, gallery_csv in zip(query_csvs, gallery_csvs):
            query_df = pd.read_csv(query_csv)
            gallery_df = pd.read_csv(gallery_csv)
            gallery_ids = gallery_df["target_id"].astype(str).tolist()
            gallery_features = torch.stack(
                [self.feature_for_row(row, id_col="target_id", image_col="target_image") for _, row in gallery_df.iterrows()]
            ).to(self.device)

            for _, query_row in query_df.iterrows():
                query_id = str(query_row["query_id"])
                query_feature = self.feature_for_row(query_row, id_col="query_id", image_col="query_image").to(self.device)
                scores = self.batched_iou_scores(query_feature, gallery_features)
                ranking = [gallery_ids[index] for index in torch.argsort(scores, descending=True).cpu().tolist()]
                rows.append({"query_id": query_id, "target_id_ranking": " ".join(ranking)})

        submission = pd.DataFrame(rows)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(self.output_csv, index=False)
        return submission

    def feature_for_row(self, row: pd.Series, id_col: str, image_col: str) -> torch.Tensor:
        """Load/cache one image feature."""
        image_id = str(row[id_col])
        if image_id not in self.feature_cache:
            image_path = self.resolve_path(str(row[image_col]))
            self.feature_cache[image_id] = self.volume_feature(image_path).cpu()
        return self.feature_cache[image_id]

    def volume_feature(self, image_path: Path) -> torch.Tensor:
        """Convert a 3D MRI into [3, H, W] binary projection masks."""
        volume = np.asarray(nib.load(str(image_path)).get_fdata(dtype=np.float32))
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        foreground_values = volume[volume > 0]
        if foreground_values.size == 0:
            mask = np.zeros_like(volume, dtype=np.float32)
        else:
            threshold = np.quantile(foreground_values, self.threshold_quantile)
            mask = (volume > threshold).astype(np.float32)

        projection_tensors = []
        for projection in (mask.max(axis=0), mask.max(axis=1), mask.max(axis=2)):
            tensor = torch.from_numpy(projection).unsqueeze(0).unsqueeze(0)
            tensor = F.interpolate(
                tensor,
                size=(self.projection_size, self.projection_size),
                mode="nearest",
            ).squeeze(0).squeeze(0)
            projection_tensors.append(tensor)
        return torch.stack(projection_tensors, dim=0).bool()

    def batched_iou_scores(self, query_feature: torch.Tensor, gallery_features: torch.Tensor) -> torch.Tensor:
        """Compute mean projection IoU from one query to all gallery features."""
        query_feature = query_feature.unsqueeze(0)
        scores = []
        for start in range(0, gallery_features.shape[0], self.batch_size):
            batch = gallery_features[start : start + self.batch_size]
            intersection = (query_feature & batch).sum(dim=(1, 2, 3)).float()
            union = (query_feature | batch).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
            scores.append(intersection / union)
        return torch.cat(scores, dim=0)

    def resolve_path(self, image_path: str) -> Path:
        path = Path(image_path)
        resolved = path if path.is_absolute() else self.data_root / path
        if resolved.exists():
            return resolved
        if resolved.name.endswith(".nii.gz"):
            nii_path = resolved.with_name(resolved.name[:-3])
            if nii_path.exists():
                return nii_path
        return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Kaggle MRI retrieval submission with an IoU baseline.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("iou_submission.csv"))
    parser.add_argument("--query-csv", type=Path, action="append", required=True)
    parser.add_argument("--gallery-csv", type=Path, action="append", required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--projection-size", type=int, default=128)
    parser.add_argument("--threshold-quantile", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = MRIIoUSubmission(
        data_root=args.data_root,
        output_csv=args.out,
        device=args.device,
        projection_size=args.projection_size,
        threshold_quantile=args.threshold_quantile,
        batch_size=args.batch_size,
    )
    submission = runner.build_submission(args.query_csv, args.gallery_csv)
    print(f"Wrote {len(submission)} rows to {args.out}")
    print(f"Device: {runner.device}")


if __name__ == "__main__":
    main()
