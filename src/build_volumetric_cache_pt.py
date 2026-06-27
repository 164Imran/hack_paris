from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from volumetric_jepa_vae_submission import resolve_path, volume_to_slice_stack


DEFAULT_SPLITS = ("val", "test")
DEFAULT_DATASETS = ("dataset1", "dataset2", "dataset3")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def default_prediction_csvs(data_root: Path) -> tuple[list[Path], list[Path]]:
    query_csvs = []
    gallery_csvs = []
    for dataset_name in DEFAULT_DATASETS:
        for split in DEFAULT_SPLITS:
            query_csvs.append(data_root / dataset_name / f"{split}_queries.csv")
            gallery_csvs.append(data_root / dataset_name / f"{split}_gallery.csv")
    return query_csvs, gallery_csvs


def tensor_to_storage(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "uint8":
        return (tensor.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    if dtype == "float16":
        return tensor.to(torch.float16)
    if dtype == "bfloat16":
        return tensor.to(torch.bfloat16)
    if dtype == "float32":
        return tensor.to(torch.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def add_image_record(
    images: dict[str, dict[str, str]],
    image_id: str,
    image_path: str,
    modality: str,
    dataset: str,
    role: str,
) -> None:
    if image_id in images:
        images[image_id]["roles"] = " ".join(sorted(set(images[image_id]["roles"].split()) | {role}))
        return
    images[image_id] = {
        "id": image_id,
        "image_path": image_path,
        "modality": modality,
        "dataset": dataset,
        "roles": role,
    }


def collect_manifest(
    data_root: Path,
    train_pairs_csv: Path,
    query_csvs: list[Path],
    gallery_csvs: list[Path],
) -> tuple[dict[str, dict[str, str]], list[dict[str, str]], list[dict[str, object]]]:
    images: dict[str, dict[str, str]] = {}
    train_pairs = read_csv(train_pairs_csv)
    for row in train_pairs:
        add_image_record(
            images,
            image_id=row["query_id"],
            image_path=row["query_image"],
            modality=row.get("query_modality", "query"),
            dataset=row.get("dataset", "dataset1"),
            role="train_query",
        )
        add_image_record(
            images,
            image_id=row["target_id"],
            image_path=row["target_image"],
            modality=row.get("target_modality", "target"),
            dataset=row.get("dataset", "dataset1"),
            role="train_target",
        )

    prediction_sets: list[dict[str, object]] = []
    for query_csv, gallery_csv in zip(query_csvs, gallery_csvs):
        queries = read_csv(query_csv)
        targets = read_csv(gallery_csv)
        query_ids = []
        target_ids = []
        for row in queries:
            query_ids.append(row["query_id"])
            add_image_record(
                images,
                image_id=row["query_id"],
                image_path=row["query_image"],
                modality=row.get("query_modality", "query"),
                dataset=row.get("dataset", query_csv.parent.name),
                role=f"{query_csv.parent.name}_{query_csv.stem}",
            )
        for row in targets:
            target_ids.append(row["target_id"])
            add_image_record(
                images,
                image_id=row["target_id"],
                image_path=row["target_image"],
                modality=row.get("target_modality", "target"),
                dataset=row.get("dataset", gallery_csv.parent.name),
                role=f"{gallery_csv.parent.name}_{gallery_csv.stem}",
            )
        prediction_sets.append(
            {
                "query_csv": str(query_csv),
                "gallery_csv": str(gallery_csv),
                "query_ids": query_ids,
                "target_ids": target_ids,
            }
        )
    return images, train_pairs, prediction_sets


def build_cache(args: argparse.Namespace) -> dict[str, object]:
    data_root = args.data_root.resolve()
    query_csvs, gallery_csvs = default_prediction_csvs(data_root)
    if args.query_csv or args.gallery_csv:
        query_csvs = args.query_csv
        gallery_csvs = args.gallery_csv
    if len(query_csvs) != len(gallery_csvs):
        raise ValueError("--query-csv and --gallery-csv must have the same count")

    image_manifest, train_pairs, prediction_sets = collect_manifest(
        data_root=data_root,
        train_pairs_csv=args.train_pairs_csv,
        query_csvs=query_csvs,
        gallery_csvs=gallery_csvs,
    )
    sample_query_order = None
    if args.sample_submission and args.sample_submission.exists():
        sample_query_order = [row["query_id"] for row in read_csv(args.sample_submission)]

    cache_images: dict[str, dict[str, object]] = {}
    records = sorted(image_manifest.values(), key=lambda item: item["id"])
    print(f"images to cache: {len(records)}")
    for index, record in enumerate(records, start=1):
        path = resolve_path(data_root, record["image_path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing image for {record['id']}: {path}")
        tensor, mask = volume_to_slice_stack(
            path,
            image_size=args.image_size,
            slice_step=args.slice_step,
            max_slices=args.max_slices,
        )
        cache_images[record["id"]] = {
            **record,
            "resolved_path": str(path),
            "tensor": tensor_to_storage(tensor, args.dtype).cpu(),
            "mask": mask.cpu(),
            "shape": tuple(tensor.shape),
        }
        if index == 1 or index % args.log_every == 0 or index == len(records):
            print(f"[{index}/{len(records)}] cached {record['id']} {tuple(tensor.shape)}")

    cache = {
        "images": cache_images,
        "train_pairs": train_pairs,
        "prediction_sets": prediction_sets,
        "sample_query_order": sample_query_order,
        "config": {
            "data_root": str(data_root),
            "train_pairs_csv": str(args.train_pairs_csv),
            "image_size": args.image_size,
            "slice_step": args.slice_step,
            "max_slices": args.max_slices,
            "dtype": args.dtype,
            "num_images": len(cache_images),
            "num_train_pairs": len(train_pairs),
            "prediction_sets": len(prediction_sets),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, args.out)
    metadata_path = args.out.with_suffix(args.out.suffix + ".json")
    metadata_path.write_text(json.dumps(cache["config"], indent=2), encoding="utf-8")
    print(f"saved cache: {args.out}")
    print(f"saved metadata: {metadata_path}")
    print(f"cache config: {cache['config']}")
    return cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compressed .pt cache from local MRI NIfTI files.")
    parser.add_argument("--data-root", type=Path, default=Path("data/ehl-paris-medical-image-retrieval"))
    parser.add_argument("--train-pairs-csv", type=Path, default=None)
    parser.add_argument("--sample-submission", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--query-csv", type=Path, action="append", default=[])
    parser.add_argument("--gallery-csv", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=Path("data/mri_volumetric_cache_step5_96_uint8.pt"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--slice-step", type=int, default=5)
    parser.add_argument("--max-slices", type=int, default=32)
    parser.add_argument("--dtype", choices=("uint8", "float16", "bfloat16", "float32"), default="uint8")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.train_pairs_csv is None:
        args.train_pairs_csv = args.data_root / "dataset1" / "train_pairs.csv"
    return args


def main() -> None:
    build_cache(parse_args())


if __name__ == "__main__":
    main()
