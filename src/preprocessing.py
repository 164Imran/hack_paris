from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
)


AXES = (0, 1, 2)


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV manifest as dictionaries."""
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_path(data_root: Path, image_path: str) -> Path:
    """Resolve dataset CSV image paths relative to data_root."""
    path = Path(image_path)
    return path if path.is_absolute() else data_root / path


def select_slice_indices(num_slices: int, slice_skip: int, include_last: bool) -> list[int]:
    """Keep one slice every `slice_skip` slices."""
    if slice_skip < 1:
        raise ValueError("--slice-skip must be >= 1")
    indices = list(range(0, num_slices, slice_skip))
    last_index = num_slices - 1
    if include_last and indices[-1] != last_index:
        indices.append(last_index)
    return indices


def monai_loader(normalize: bool) -> Compose:
    """Build the MONAI loading/preprocessing pipeline."""
    transforms = [
        LoadImaged(keys="image", image_only=False),
        EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
        Orientationd(keys="image", axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
    ]
    if normalize:
        transforms.append(
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                channel_wise=True,
            )
        )
    return Compose(transforms)


def load_volume(path: Path, normalize: bool) -> tuple[torch.Tensor, tuple[float, ...]]:
    """Load one 3D MRI as [X, Y, Z] with MONAI."""
    row = monai_loader(normalize=normalize)({"image": str(path)})
    volume = torch.as_tensor(row["image"]).float()
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(f"{path} must be a single-channel 3D MRI volume, got shape {tuple(volume.shape)}")

    volume = torch.nan_to_num(volume[0], nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    metadata = row.get("image_meta_dict", {})
    voxel_size = tuple(float(v) for v in metadata.get("pixdim", [1.0, 1.0, 1.0, 1.0])[1:4])
    return volume, voxel_size


def extract_axis_slices(
    volume: torch.Tensor,
    slice_skip: int,
    include_last: bool,
) -> dict[str, dict[str, object]]:
    """Return three tensors, one per slicing axis.

    Output for each axis:
      axis_0 -> [S0, Y, Z]
      axis_1 -> [S1, X, Z]
      axis_2 -> [S2, X, Y]
    """
    axis_data: dict[str, dict[str, object]] = {}
    for axis in AXES:
        moved = torch.movedim(volume, axis, 0)
        indices = select_slice_indices(moved.shape[0], slice_skip=slice_skip, include_last=include_last)
        slices = moved[indices].contiguous()
        axis_data[f"axis_{axis}"] = {
            "tensor": slices,
            "slice_indices": indices,
            "shape": tuple(slices.shape),
        }
    return axis_data


def build_pair_example(
    row: dict[str, str],
    data_root: Path,
    slice_skip: int,
    normalize: bool,
    include_last: bool,
) -> dict[str, object]:
    """Build one patient/pair example with T1/T2 tensors on all three axes."""
    query_path = resolve_path(data_root, row["query_image"])
    target_path = resolve_path(data_root, row["target_image"])
    query_volume, query_voxel_size = load_volume(query_path, normalize=normalize)
    target_volume, target_voxel_size = load_volume(target_path, normalize=normalize)

    return {
        "pair_id": row.get("pair_id", f"{row['query_id']}__{row['target_id']}"),
        "query_id": row["query_id"],
        "target_id": row["target_id"],
        "query_modality": row.get("query_modality", "query"),
        "target_modality": row.get("target_modality", "target"),
        "dataset": row.get("dataset"),
        "query_path": str(query_path),
        "target_path": str(target_path),
        "query_volume_shape": tuple(query_volume.shape),
        "target_volume_shape": tuple(target_volume.shape),
        "query_voxel_size": query_voxel_size,
        "target_voxel_size": target_voxel_size,
        "query_axes": extract_axis_slices(query_volume, slice_skip=slice_skip, include_last=include_last),
        "target_axes": extract_axis_slices(target_volume, slice_skip=slice_skip, include_last=include_last),
    }


def build_paired_tensor_dataset(
    data_root: Path,
    pair_csv: Path,
    output_path: Path,
    slice_skip: int,
    normalize: bool = True,
    include_last: bool = False,
) -> dict[str, object]:
    """Build the full paired MRI tensor dataset from a train_pairs-style CSV."""
    rows = read_csv(pair_csv)
    if not rows:
        raise ValueError(f"No rows found in {pair_csv}")

    examples = [
        build_pair_example(
            row=row,
            data_root=data_root,
            slice_skip=slice_skip,
            normalize=normalize,
            include_last=include_last,
        )
        for row in rows
    ]

    dataset = {
        "examples": examples,
        "config": {
            "data_root": str(data_root),
            "pair_csv": str(pair_csv),
            "slice_skip": slice_skip,
            "axes": AXES,
            "normalize": normalize,
            "include_last": include_last,
            "loader": "MONAI LoadImaged + EnsureChannelFirstd + Orientationd(RAS)",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a paired T1/T2 MRI tensor dataset with slices extracted on all 3 axes."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder used to resolve CSV image paths.")
    parser.add_argument("--pair-csv", type=Path, required=True, help="CSV with query_image and target_image columns.")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt dataset file.")
    parser.add_argument("--slice-skip", type=int, default=1, help="Keep one slice every N slices, e.g. 5 keeps 0,5,10...")
    parser.add_argument("--include-last", action="store_true", help="Also keep the final slice of each axis.")
    parser.add_argument("--no-normalize", action="store_true", help="Keep raw intensities instead of MONAI percentile scaling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_paired_tensor_dataset(
        data_root=args.data_root.resolve(),
        pair_csv=args.pair_csv,
        output_path=args.output,
        slice_skip=args.slice_skip,
        normalize=not args.no_normalize,
        include_last=args.include_last,
    )
    first = dataset["examples"][0]
    print(f"Saved: {args.output}")
    print(f"Pairs: {len(dataset['examples'])}")
    print(f"First pair: {first['pair_id']}")
    print(f"Query shape: {first['query_volume_shape']} Target shape: {first['target_volume_shape']}")
    for axis in ("axis_0", "axis_1", "axis_2"):
        print(
            f"{axis}: query {first['query_axes'][axis]['shape']} "
            f"target {first['target_axes'][axis]['shape']}"
        )


if __name__ == "__main__":
    main()
