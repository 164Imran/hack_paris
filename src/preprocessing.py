from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
)


class MRIPairPreprocessor:
    """Create a paired T1/T2 MRI tensor dataset from train_pairs.csv.

    One-cell Colab/Kaggle usage:

        from preprocessing import MRIPairPreprocessor

        preprocessor = MRIPairPreprocessor(slice_skip=5, skip_first=10, skip_last=10)
        dataset = preprocessor.create_dataset(
            train_pairs_csv="/content/dataset1/train_pairs.csv",
            data_root="/content",
            output_path="/content/preprocessed_pairs.pt",
        )

    `data_root` must be the folder that contains `dataset1` when CSV paths look
    like `dataset1/images/train/queries/q_xxx.nii.gz`.
    """

    axes = (0, 1, 2)
    augmentation_count = 3

    def __init__(
        self,
        slice_skip: int = 1,
        normalize: bool = True,
        include_last: bool = False,
        skip_first: int = 0,
        skip_last: int = 0,
        slice_indices: list[int] | None = None,
        seed: int = 20260627,
        max_translation: float = 0.12,
        elastic_alpha: float = 0.06,
        elastic_sigma: float = 16.0,
        output_dtype: str | torch.dtype = "float32",
    ) -> None:
        self.slice_skip = slice_skip
        self.normalize = normalize
        self.include_last = include_last
        self.skip_first = skip_first
        self.skip_last = skip_last
        self.slice_indices = slice_indices
        self.seed = seed
        self.max_translation = max_translation
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.output_dtype = self.resolve_dtype(output_dtype)

    def create_dataset(
        self,
        train_pairs_csv: str | Path,
        data_root: str | Path | None = None,
        output_path: str | Path | None = None,
        max_pairs: int | None = None,
        output_format: str | None = None,
    ) -> dict[str, object]:
        """Read train_pairs.csv, build all tensors, and optionally save a .pt file."""
        train_pairs_csv = Path(train_pairs_csv)
        rows = self.read_csv(train_pairs_csv)
        if max_pairs is not None:
            rows = rows[:max_pairs]
        if not rows:
            raise ValueError(f"No rows found in {train_pairs_csv}")
        data_root = self.infer_data_root(train_pairs_csv, data_root, rows[0])

        rng = random.Random(self.seed)
        examples = [self.build_pair_example(row=row, data_root=data_root, rng=rng) for row in rows]
        dataset = {
            "examples": examples,
            "config": self.config(data_root=data_root, train_pairs_csv=train_pairs_csv, max_pairs=max_pairs),
        }
        if output_path is not None:
            self.save_dataset(dataset, output_path, output_format=output_format)
        return dataset

    def create_from_dataset1(
        self,
        dataset_root: str | Path,
        output_path: str | Path | None = None,
        max_pairs: int | None = None,
        output_format: str | None = None,
    ) -> dict[str, object]:
        """Shortcut when you pass the `dataset1` folder itself."""
        dataset_root = Path(dataset_root)
        return self.create_dataset(
            train_pairs_csv=dataset_root / "train_pairs.csv",
            data_root=dataset_root.parent,
            output_path=output_path,
            max_pairs=max_pairs,
            output_format=output_format,
        )

    def save_dataset(
        self,
        dataset: dict[str, object],
        output_path: str | Path,
        output_format: str | None = None,
    ) -> None:
        """Save dataset as .pt, .npz, or .npy/.npi."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_format = self.resolve_output_format(output_path, output_format)

        if save_format == "pt":
            torch.save(dataset, output_path)
        elif save_format == "npz":
            np.savez_compressed(output_path, dataset=np.array(dataset, dtype=object))
        elif save_format == "npy":
            np.save(output_path, np.array(dataset, dtype=object), allow_pickle=True)
        else:
            raise ValueError(f"Unsupported output format: {save_format}")

    @staticmethod
    def load_numpy_dataset(path: str | Path) -> dict[str, object]:
        """Load a dataset saved as .npz or .npy/.npi."""
        path = Path(path)
        if path.suffix.lower() == ".npz":
            return np.load(path, allow_pickle=True)["dataset"].item()
        return np.load(path, allow_pickle=True).item()

    def build_pair_example(self, row: dict[str, str], data_root: Path, rng: random.Random) -> dict[str, object]:
        """Build one paired T1/T2 example with all named views."""
        query_path = self.resolve_path(data_root, row["query_image"])
        target_path = self.resolve_path(data_root, row["target_image"])
        query_volume, query_voxel_size = self.load_volume(query_path)
        target_volume, target_voxel_size = self.load_volume(target_path)

        query_base_axes = self.extract_axis_slices(query_volume)
        target_base_axes = self.extract_axis_slices(target_volume)
        query_rigid_params = self.sample_rigid_params(rng)
        target_rigid_params = self.sample_rigid_params(rng)
        query_flip_params = self.sample_flip_params(rng)
        target_flip_params = self.sample_flip_params(rng, avoid=query_flip_params)
        query_nonlinear_seeds = [rng.randrange(2**31) for _ in range(self.augmentation_count)]
        target_nonlinear_seeds = [rng.randrange(2**31) for _ in range(self.augmentation_count)]
        query_transform_params = self.transform_params(query_rigid_params, query_flip_params, query_nonlinear_seeds)
        target_transform_params = self.transform_params(target_rigid_params, target_flip_params, target_nonlinear_seeds)

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
            "query_transform_params": query_transform_params,
            "target_transform_params": target_transform_params,
            "query_views": self.augment_axis_slices(
                base_axes=query_base_axes,
                rigid_params=query_rigid_params,
                flip_params=query_flip_params,
                nonlinear_seeds=query_nonlinear_seeds,
                image_id=row["query_id"],
                modality=row.get("query_modality", "query"),
            ),
            "target_views": self.augment_axis_slices(
                base_axes=target_base_axes,
                rigid_params=target_rigid_params,
                flip_params=target_flip_params,
                nonlinear_seeds=target_nonlinear_seeds,
                image_id=row["target_id"],
                modality=row.get("target_modality", "target"),
            ),
        }

    def load_volume(self, path: Path) -> tuple[torch.Tensor, tuple[float, ...]]:
        """Load one NIfTI volume as [X, Y, Z] using MONAI."""
        row = self.monai_loader()({"image": str(path)})
        volume = torch.as_tensor(row["image"]).float()
        if volume.ndim != 4 or volume.shape[0] != 1:
            raise ValueError(f"{path} must be a single-channel 3D MRI volume, got shape {tuple(volume.shape)}")

        volume = torch.nan_to_num(volume[0], nan=0.0, posinf=0.0, neginf=0.0).contiguous()
        metadata = row.get("image_meta_dict", {})
        voxel_size = tuple(float(v) for v in metadata.get("pixdim", [1.0, 1.0, 1.0, 1.0])[1:4])
        return volume, voxel_size

    def monai_loader(self) -> Compose:
        """Return the MONAI loading pipeline."""
        transforms = [
            LoadImaged(keys="image", image_only=False),
            EnsureChannelFirstd(keys="image", channel_dim="no_channel"),
            Orientationd(keys="image", axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        ]
        if self.normalize:
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

    def extract_axis_slices(self, volume: torch.Tensor) -> dict[str, dict[str, object]]:
        """Cut a 3D volume along axes 1, 2, and 3."""
        axis_data: dict[str, dict[str, object]] = {}
        for axis in self.axes:
            moved = torch.movedim(volume, axis, 0)
            indices = self.select_slice_indices(moved.shape[0])
            slices = moved[indices].contiguous()
            axis_data[f"axis_{axis}"] = {
                "tensor": slices,
                "slice_indices": indices,
                "shape": tuple(slices.shape),
            }
        return axis_data

    def augment_axis_slices(
        self,
        base_axes: dict[str, dict[str, object]],
        rigid_params: list[dict[str, float]],
        flip_params: list[dict[str, object]],
        nonlinear_seeds: list[int],
        image_id: str,
        modality: str,
    ) -> dict[str, dict[str, object]]:
        """Create views; each transform is shared by all slices of the same image."""
        views: dict[str, dict[str, object]] = {}
        for axis_name, axis_payload in base_axes.items():
            axis_number = int(axis_name.split("_")[-1]) + 1
            axis_suffix = f"axe{axis_number}"
            tensor = torch.as_tensor(axis_payload["tensor"]).float()
            common = {
                "slice_indices": axis_payload["slice_indices"],
                "source_shape": axis_payload["shape"],
                "axis": axis_number,
            }

            view = f"basic_{axis_suffix}"
            views[view] = self.named_view_payload(
                common=common,
                tensor=tensor,
                transform="identity",
                image_id=image_id,
                modality=modality,
                view=view,
                params={},
            )

            for rigid_number, params in enumerate(rigid_params, start=1):
                rigid = self.rigid_slice_stack(tensor, **params).contiguous()
                view = f"rotate_{rigid_number}_{axis_suffix}"
                views[view] = self.named_view_payload(
                    common=common,
                    tensor=rigid,
                    transform="rigid_rotation_translation",
                    image_id=image_id,
                    modality=modality,
                    view=view,
                    params=params,
                )

            for flip_number, params in enumerate(flip_params, start=1):
                flip_dims = params["flip_dims"]
                flipped = torch.flip(tensor, dims=flip_dims).contiguous()
                view = f"flip_{flip_number}_{axis_suffix}"
                views[view] = self.named_view_payload(
                    common=common,
                    tensor=flipped,
                    transform="flip",
                    image_id=image_id,
                    modality=modality,
                    view=view,
                    params=params,
                )

            for nonlinear_number, seed in enumerate(nonlinear_seeds, start=1):
                elastic = self.elastic_slice_stack(tensor, seed=seed).contiguous()
                view = f"nonlinear_{nonlinear_number}_{axis_suffix}"
                nonlinear_params = {"alpha": self.elastic_alpha, "sigma": self.elastic_sigma, "seed": seed}
                views[view] = self.named_view_payload(
                    common=common,
                    tensor=elastic,
                    transform="elastic_non_linear",
                    image_id=image_id,
                    modality=modality,
                    view=view,
                    params=nonlinear_params,
                )
        return views

    def named_view_payload(
        self,
        common: dict[str, object],
        tensor: torch.Tensor,
        transform: str,
        image_id: str,
        modality: str,
        view: str,
        params: dict[str, object],
    ) -> dict[str, object]:
        """Build a view payload with readable name and structured metadata."""
        payload = self.view_payload(common, tensor, transform, params)
        image_name = self.make_image_name(image_id=image_id, modality=modality, view=view, params=params)
        metadata = self.make_view_metadata(
            image_id=image_id,
            modality=modality,
            view=view,
            transform=transform,
            image_name=image_name,
            params=params,
            axis=int(common["axis"]),
            slice_indices=list(common["slice_indices"]),
        )
        payload["image_name"] = image_name
        payload["metadata"] = metadata
        return payload

    def view_payload(
        self,
        common: dict[str, object],
        tensor: torch.Tensor,
        transform: str,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build one serializable view dictionary."""
        payload = {
            **common,
            "tensor": tensor.to(dtype=self.output_dtype),
            "shape": tuple(tensor.shape),
            "transform": transform,
        }
        if extra:
            payload.update(extra)
        return payload

    def make_image_name(self, image_id: str, modality: str, view: str, params: dict[str, object]) -> str:
        """Create a readable generated image/view name with transform parameters."""
        safe_modality = self.safe_token(modality)
        suffix_parts = [image_id, safe_modality, view]
        if "angle_degrees" in params:
            suffix_parts.extend(
                [
                    f"deg{self.format_float(float(params['angle_degrees']))}",
                    f"tx{self.format_float(float(params['translate_x']))}",
                    f"ty{self.format_float(float(params['translate_y']))}",
                ]
            )
        elif "flip_name" in params:
            suffix_parts.append(str(params["flip_name"]))
        elif "alpha" in params:
            suffix_parts.extend(
                [
                    f"alpha{self.format_float(float(params['alpha']))}",
                    f"sigma{self.format_float(float(params['sigma']))}",
                    f"seed{params['seed']}",
                ]
            )
        return "__".join(suffix_parts)

    def make_view_metadata(
        self,
        image_id: str,
        modality: str,
        view: str,
        transform: str,
        image_name: str,
        params: dict[str, object],
        axis: int,
        slice_indices: list[int],
    ) -> dict[str, object]:
        """Structured metadata used for plotting titles and downstream inspection."""
        return {
            "image_id": image_id,
            "modality": modality,
            "view": view,
            "axis": axis,
            "slice_indices": slice_indices,
            "transform": transform,
            "params": dict(params),
            "image_name": image_name,
            "display_title": self.make_display_title(view=view, modality=modality, params=params),
        }

    def make_display_title(self, view: str, modality: str, params: dict[str, object]) -> str:
        """Compact title with transform parameters for notebook visualization."""
        if "angle_degrees" in params:
            detail = (
                f"rot={float(params['angle_degrees']):.1f} deg, "
                f"tx={float(params['translate_x']):.3f}, "
                f"ty={float(params['translate_y']):.3f}"
            )
        elif "flip_name" in params:
            detail = str(params["flip_name"])
        elif "alpha" in params:
            detail = f"alpha={float(params['alpha']):.3f}, sigma={float(params['sigma']):.1f}, seed={params['seed']}"
        else:
            detail = "original"
        return f"{modality} | {view}\n{detail}"

    @staticmethod
    def format_float(value: float) -> str:
        """Format floats for filenames without punctuation that is annoying in paths."""
        return f"{value:.4f}".replace("-", "m").replace(".", "p")

    @staticmethod
    def safe_token(value: str) -> str:
        """Make modality strings safe for generated names."""
        return "".join(char if char.isalnum() else "_" for char in value).strip("_")

    @staticmethod
    def flip_name(flip_dims: tuple[int, ...]) -> str:
        """Human-readable flip description for a 2D slice tensor."""
        if flip_dims == (-1,):
            return "flip_width"
        if flip_dims == (-2,):
            return "flip_height"
        return "flip_" + "_".join(str(dim) for dim in flip_dims)


    def rigid_slice_stack(
        self,
        slices: torch.Tensor,
        angle_degrees: float,
        translate_x: float,
        translate_y: float,
    ) -> torch.Tensor:
        """Apply rotation and translation to a slice stack [S, H, W]."""
        angle = math.radians(angle_degrees)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        theta = torch.tensor(
            [[cos_a, -sin_a, translate_x], [sin_a, cos_a, translate_y]],
            dtype=slices.dtype,
            device=slices.device,
        ).unsqueeze(0).repeat(slices.shape[0], 1, 1)
        images = slices.unsqueeze(1)
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(1)

    def elastic_slice_stack(self, slices: torch.Tensor, seed: int) -> torch.Tensor:
        """Apply a smooth non-linear 2D deformation to [S, H, W]."""
        if self.elastic_alpha <= 0:
            return slices.clone()

        images = slices.unsqueeze(1)
        _, _, height, width = images.shape
        generator = torch.Generator(device=slices.device).manual_seed(seed)
        noise = torch.empty(1, 2, height, width, dtype=slices.dtype, device=slices.device).uniform_(
            -1.0,
            1.0,
            generator=generator,
        )
        kernel_size = max(3, int(round(self.elastic_sigma)) * 2 + 1)
        displacement = F.avg_pool2d(noise, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        displacement = displacement / displacement.abs().amax().clamp_min(1e-6)
        displacement = displacement * self.elastic_alpha

        y_coords = torch.linspace(-1.0, 1.0, height, dtype=slices.dtype, device=slices.device)
        x_coords = torch.linspace(-1.0, 1.0, width, dtype=slices.dtype, device=slices.device)
        base_y, base_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0).repeat(slices.shape[0], 1, 1, 1)
        grid = base_grid.clone()
        grid[..., 0] = grid[..., 0] + displacement[:, 0]
        grid[..., 1] = grid[..., 1] + displacement[:, 1]
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False).squeeze(1)

    def sample_rigid_params(self, rng: random.Random) -> list[dict[str, float]]:
        """Sample three random rigid transforms for one image."""
        return [
            {
                "angle_degrees": rng.uniform(0.0, 180.0),
                "translate_x": rng.uniform(-self.max_translation, self.max_translation),
                "translate_y": rng.uniform(-self.max_translation, self.max_translation),
            }
            for _ in range(self.augmentation_count)
        ]

    def sample_flip_params(
        self,
        rng: random.Random,
        avoid: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        """Sample three random flip transforms for one image."""
        choices = [
            {"flip_dims": (-1,), "flip_name": "flip_width"},
            {"flip_dims": (-2,), "flip_name": "flip_height"},
            {"flip_dims": (-2, -1), "flip_name": "flip_height_width"},
        ]
        sampled = choices[:]
        rng.shuffle(sampled)
        if avoid is not None:
            derangements = [
                list(candidate)
                for candidate in itertools.permutations(choices, len(choices))
                if all(left != right for left, right in zip(candidate, avoid))
            ]
            sampled = rng.choice(derangements)
        return sampled[: self.augmentation_count]

    def transform_params(
        self,
        rigid_params: list[dict[str, float]],
        flip_params: list[dict[str, object]],
        nonlinear_seeds: list[int],
    ) -> dict[str, list[dict[str, object]]]:
        """Return per-image transform parameters used for every axis/slice."""
        return {
            "rotate": [dict(params) for params in rigid_params],
            "flip": [dict(params) for params in flip_params],
            "nonlinear": [
                {"alpha": self.elastic_alpha, "sigma": self.elastic_sigma, "seed": seed}
                for seed in nonlinear_seeds
            ],
        }

    def select_slice_indices(self, num_slices: int) -> list[int]:
        """Select slices explicitly or with skip/exclusion parameters."""
        if self.slice_skip < 1:
            raise ValueError("slice_skip must be >= 1")
        if self.skip_first < 0 or self.skip_last < 0:
            raise ValueError("skip_first and skip_last must be >= 0")

        if self.slice_indices is not None:
            invalid = [index for index in self.slice_indices if index < 0 or index >= num_slices]
            if invalid:
                raise ValueError(f"Slice indices out of range for volume with {num_slices} slices: {invalid}")
            return list(self.slice_indices)

        start = min(self.skip_first, num_slices)
        stop = max(start, num_slices - self.skip_last)
        indices = list(range(start, stop, self.slice_skip))
        if not indices:
            raise ValueError(
                f"No slices selected: num_slices={num_slices}, skip_first={self.skip_first}, "
                f"skip_last={self.skip_last}, slice_skip={self.slice_skip}"
            )
        last_index = num_slices - 1
        if self.include_last and last_index >= start and indices[-1] != last_index:
            indices.append(last_index)
        return indices

    def config(self, data_root: Path, train_pairs_csv: Path, max_pairs: int | None) -> dict[str, object]:
        """Return preprocessing settings saved in the .pt file."""
        return {
            "data_root": str(data_root),
            "train_pairs_csv": str(train_pairs_csv),
            "max_pairs": max_pairs,
            "slice_skip": self.slice_skip,
            "skip_first": self.skip_first,
            "skip_last": self.skip_last,
            "slice_indices": self.slice_indices,
            "axes": self.axes,
            "view_pattern": "{basic|rotate_N|flip_N|nonlinear_N}_axeM",
            "transform_policy": (
                "For one image, each random transform is sampled once and applied "
                "consistently to all selected slices and axes. Across different "
                "individuals, and between query/T1 and target/T2 in a correct pair, "
                "random rigid rotation/translation, flips, and non-linear "
                "deformation parameters are sampled independently."
            ),
            "augmentation_count": self.augmentation_count,
            "normalize": self.normalize,
            "include_last": self.include_last,
            "seed": self.seed,
            "max_translation": self.max_translation,
            "elastic_alpha": self.elastic_alpha,
            "elastic_sigma": self.elastic_sigma,
            "output_dtype": str(self.output_dtype).replace("torch.", ""),
            "loader": "MONAI LoadImaged + EnsureChannelFirstd + Orientationd(RAS)",
        }

    @staticmethod
    def read_csv(path: Path) -> list[dict[str, str]]:
        """Read a CSV manifest as dictionaries."""
        with path.open(newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def resolve_path(data_root: Path, image_path: str) -> Path:
        """Resolve image paths relative to data_root."""
        path = Path(image_path)
        return path if path.is_absolute() else data_root / path

    @staticmethod
    def infer_data_root(
        train_pairs_csv: Path,
        data_root: str | Path | None,
        first_row: dict[str, str] | None = None,
    ) -> Path:
        """Infer or normalize the root folder containing `dataset1`."""
        if data_root is not None:
            return Path(data_root).resolve()
        if train_pairs_csv.parent.name.startswith("dataset"):
            return train_pairs_csv.parent.parent.resolve()
        if first_row is not None:
            relative_image = Path(first_row["query_image"])
            for candidate in (train_pairs_csv.parent, train_pairs_csv.parent.parent, Path.cwd()):
                if (candidate / relative_image).exists():
                    return candidate.resolve()
        return train_pairs_csv.parent.resolve()

    @staticmethod
    def parse_int_list(value: str | None) -> list[int] | None:
        """Parse '80' or '40,80,120' into a list."""
        if value is None or value.strip() == "":
            return None
        return [int(item.strip()) for item in value.split(",") if item.strip()]

    @staticmethod
    def resolve_dtype(value: str | torch.dtype) -> torch.dtype:
        """Resolve output tensor dtype for saved dataset size control."""
        if isinstance(value, torch.dtype):
            return value
        aliases = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        key = value.lower()
        if key not in aliases:
            raise ValueError(f"Unsupported output_dtype {value!r}. Use float32, float16, or bfloat16.")
        return aliases[key]

    @staticmethod
    def resolve_output_format(output_path: Path, output_format: str | None) -> str:
        """Resolve output format from explicit value or file suffix."""
        if output_format is not None:
            fmt = output_format.lower().lstrip(".")
        else:
            fmt = output_path.suffix.lower().lstrip(".")
        aliases = {"pth": "pt", "npi": "npy"}
        fmt = aliases.get(fmt, fmt)
        if fmt not in {"pt", "npz", "npy"}:
            raise ValueError("Unsupported output format. Use pt, npz, npy, or npi.")
        return fmt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a paired T1/T2 MRI tensor dataset from train_pairs.csv.")
    parser.add_argument("--train-pairs-csv", "--pair-csv", dest="train_pairs_csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-format", type=str, default=None, choices=("pt", "npz", "npy", "npi"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--slice-skip", type=int, default=1)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--skip-last", type=int, default=0)
    parser.add_argument("--slice-indices", type=str, default=None)
    parser.add_argument("--include-last", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--max-translation", type=float, default=0.12)
    parser.add_argument("--elastic-alpha", type=float, default=0.06)
    parser.add_argument("--elastic-sigma", type=float, default=16.0)
    parser.add_argument("--output-dtype", type=str, default="float32", choices=("float32", "float16", "bfloat16"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocessor = MRIPairPreprocessor(
        slice_skip=args.slice_skip,
        normalize=not args.no_normalize,
        include_last=args.include_last,
        skip_first=args.skip_first,
        skip_last=args.skip_last,
        slice_indices=MRIPairPreprocessor.parse_int_list(args.slice_indices),
        seed=args.seed,
        max_translation=args.max_translation,
        elastic_alpha=args.elastic_alpha,
        elastic_sigma=args.elastic_sigma,
        output_dtype=args.output_dtype,
    )
    dataset = preprocessor.create_dataset(
        train_pairs_csv=args.train_pairs_csv,
        data_root=args.data_root,
        output_path=args.output,
        max_pairs=args.max_pairs,
        output_format=args.output_format,
    )
    first = dataset["examples"][0]
    print(f"Saved: {args.output}")
    print(f"Pairs: {len(dataset['examples'])}")
    print(f"First pair: {first['pair_id']}")
    print(f"Query shape: {first['query_volume_shape']} Target shape: {first['target_volume_shape']}")
    for view_name in first["query_views"]:
        print(
            f"{view_name}: query {first['query_views'][view_name]['shape']} "
            f"target {first['target_views'][view_name]['shape']}"
        )


if __name__ == "__main__":
    main()
