"""Tweet cleaning, Harvey multi-head inference, and semantic raster utilities."""
from __future__ import annotations

import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

HARVEY_TARGET_NAMES = [
    "homeOwnersInsurance", "floodInsurance", "destroyed", "floodDamage",
    "roofDamage", "tsaEligible", "tsaCheckedIn", "rentalAssistanceEligible",
    "repairAssistanceEligible", "replacementAssistanceEligible",
    "personalPropertyEligible",
]


def set_reproducibility(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_tweet_text(value: Any) -> str:
    """Normalize a tweet while retaining words carried by hashtags."""
    if pd.isna(value):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " HTTPURL ", str(value))
    text = re.sub(r"@\w+", " @USER ", text).replace("#", "")
    return re.sub(r"\s+", " ", text).strip()


def prepare_tweet_bounding_boxes(
    df: pd.DataFrame,
    text_col: str,
    lon_min_col: str,
    lat_min_col: str,
    lon_max_col: str,
    lat_max_col: str,
    texas_extent: Sequence[float],
    grid_size: float,
):
    """Clean and grid tweet boxes; return dataframe, dimensions, transform, CRS.

    ``texas_extent`` is ``(lon_min, lon_max, lat_min, lat_max)``.
    Areas are computed after clipping by projecting box polygons to EPSG:3083.
    """
    import geopandas as gpd
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    from shapely.geometry import box

    if df.empty:
        raise ValueError("Tweet dataframe is empty.")
    required = [text_col, lon_min_col, lat_min_col, lon_max_col, lat_max_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing tweet columns: {missing}")
    if grid_size <= 0:
        raise ValueError("grid_size must be positive.")
    lon_min, lon_max, lat_min, lat_max = map(float, texas_extent)
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("Invalid Texas extent.")

    out = df.copy()
    out["tweet_uid"] = out.index
    if out["tweet_uid"].duplicated().any():
        # A stable positional suffix preserves every record when the source index repeats.
        out["tweet_uid"] = [f"{idx}__{pos}" for pos, idx in enumerate(out.index)]
    out["clean_text"] = out[text_col].map(clean_tweet_text)
    coord_cols = [lon_min_col, lat_min_col, lon_max_col, lat_max_col]
    out[coord_cols] = out[coord_cols].apply(pd.to_numeric, errors="coerce")
    out = out.dropna(subset=coord_cols)
    out = out[out["clean_text"].str.len().gt(0)]
    out = out[
        out[lon_min_col].le(out[lon_max_col])
        & out[lat_min_col].le(out[lat_max_col])
        & out[lon_max_col].ge(lon_min)
        & out[lon_min_col].le(lon_max)
        & out[lat_max_col].ge(lat_min)
        & out[lat_min_col].le(lat_max)
    ].copy()
    if out.empty:
        raise ValueError("No valid tweet text/bounding boxes intersect the Texas extent.")

    out["lon_min_clip"] = out[lon_min_col].clip(lon_min, lon_max)
    out["lon_max_clip"] = out[lon_max_col].clip(lon_min, lon_max)
    out["lat_min_clip"] = out[lat_min_col].clip(lat_min, lat_max)
    out["lat_max_clip"] = out[lat_max_col].clip(lat_min, lat_max)
    ncols = int(math.ceil((lon_max - lon_min) / grid_size))
    nrows = int(math.ceil((lat_max - lat_min) / grid_size))
    out["col_min"] = np.floor((out["lon_min_clip"] - lon_min) / grid_size).astype(int)
    out["col_max"] = np.floor((out["lon_max_clip"] - lon_min) / grid_size).astype(int)
    out["row_min"] = np.floor((lat_max - out["lat_max_clip"]) / grid_size).astype(int)
    out["row_max"] = np.floor((lat_max - out["lat_min_clip"]) / grid_size).astype(int)
    for col, upper in (("row_min", nrows - 1), ("row_max", nrows - 1),
                       ("col_min", ncols - 1), ("col_max", ncols - 1)):
        out[col] = out[col].clip(0, upper).astype(np.int32)
    if ((out.row_min > out.row_max) | (out.col_min > out.col_max)).any():
        raise ValueError("Invalid raster indices remain after clipping.")

    geometry = [
        box(a, b, c, d) for a, b, c, d in zip(
            out.lon_min_clip, out.lat_min_clip, out.lon_max_clip, out.lat_max_clip
        )
    ]
    areas = gpd.GeoSeries(geometry, crs="EPSG:4326").to_crs("EPSG:3083").area / 1e6
    out["tweet_box_area_km2"] = areas.to_numpy(dtype=np.float64)
    out["spatial_precision_weight"] = (
        1.0 / np.sqrt(1.0 + out["tweet_box_area_km2"])
    ).astype(np.float32)
    if out["tweet_uid"].duplicated().any():
        raise ValueError("tweet_uid is not unique.")
    return (
        out,
        nrows,
        ncols,
        from_origin(lon_min, lat_max, grid_size, grid_size),
        CRS.from_epsg(4326),
    )


class TweetInferenceDataset(Dataset):
    """Tokenized, label-free tweet dataset."""

    def __init__(self, texts: Sequence[str], tokenizer: Any, max_length: int):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Calling the tokenizer is supported by current Transformers releases and
        # older Hugging Face tokenizers.  ``encode_plus`` was removed from some
        # recent tokenizer implementations exposed in Colab.
        encoded = self.tokenizer(
            self.texts[index],
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()
                if k in {"input_ids", "attention_mask", "token_type_ids"}}


class BERTMultiDeepHeadClassifier(nn.Module):
    """Harvey classifier supporting both observed multi-head checkpoint layouts.

    The supplied training notebook defines ``768 -> 512 -> 128 -> 1`` heads, while
    some exported fold checkpoints use ``768 -> 256 -> 1`` heads.  ``head_hidden_dims``
    makes that difference explicit; :func:`predict_tweet_impacts` infers it from the
    checkpoint before constructing the model.
    """

    outputs_are_logits = False

    def __init__(
        self,
        num_targets: int = 11,
        bert_model: nn.Module | None = None,
        pretrained_name: str = "bert-base-uncased",
        head_hidden_dims: Sequence[int] = (512, 128),
    ):
        super().__init__()
        if bert_model is None:
            from transformers import BertModel
            bert_model = BertModel.from_pretrained(pretrained_name)
        self.bert = bert_model
        self.drop = nn.Dropout(0.3)
        hidden_dims = tuple(int(value) for value in head_hidden_dims)
        if not hidden_dims or any(value <= 0 for value in hidden_dims):
            raise ValueError("head_hidden_dims must contain positive integers.")

        def make_head() -> nn.Sequential:
            dimensions = (self.bert.config.hidden_size, *hidden_dims, 1)
            layers: list[nn.Module] = []
            for layer_index, (input_dim, output_dim) in enumerate(
                zip(dimensions[:-1], dimensions[1:])
            ):
                layers.append(nn.Linear(input_dim, output_dim))
                if output_dim == 1:
                    layers.append(nn.Sigmoid())
                else:
                    layers.append(nn.ReLU())
                    # The deep notebook layout includes dropout between hidden layers.
                    # The one-hidden-layer 768->256->1 checkpoints have Linear keys
                    # at indices 0 and 2, so no dropout is inserted in that layout.
                    if len(hidden_dims) > 1:
                        layers.append(nn.Dropout(0.3))
            return nn.Sequential(*layers)

        self.head_hidden_dims = hidden_dims
        self.heads = nn.ModuleList([make_head() for _ in range(num_targets)])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                token_type_ids: torch.Tensor | None = None) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        pooled = self.drop(self.bert(**kwargs).pooler_output)
        return torch.cat([head(pooled) for head in self.heads], dim=1)


def _load_state_dict(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location=device)
    for key in ("state_dict", "model_state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {path}")
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[7:]: v for k, v in state.items()}
    return state


def _infer_checkpoint_head_hidden_dims(
    state: dict[str, torch.Tensor], target_count: int
) -> tuple[int, ...]:
    """Infer shared hidden dimensions from ``heads.<target>.<layer>.weight``."""
    layouts: list[tuple[tuple[int, int, int], ...]] = []
    for head_index in range(target_count):
        prefix = f"heads.{head_index}."
        weights = []
        for key, value in state.items():
            if (
                key.startswith(prefix)
                and key.endswith(".weight")
                and isinstance(value, torch.Tensor)
                and value.ndim == 2
            ):
                remainder = key[len(prefix):]
                try:
                    module_index = int(remainder.split(".", 1)[0])
                except ValueError:
                    continue
                weights.append((module_index, int(value.shape[0]), int(value.shape[1])))
        weights.sort()
        if not weights:
            raise ValueError(f"Checkpoint has no linear weights for head {head_index}.")
        layouts.append(tuple(weights))
    if any(layout != layouts[0] for layout in layouts[1:]):
        raise ValueError("Disaster-impact heads do not share one checkpoint architecture.")
    layout = layouts[0]
    if layout[-1][1] != 1:
        raise ValueError(f"Final head layer must output one value; observed {layout[-1]}.")
    for previous, current in zip(layout, layout[1:]):
        if previous[1] != current[2]:
            raise ValueError(f"Disconnected checkpoint head layers: {previous}, {current}.")
    hidden_dims = tuple(layer[1] for layer in layout[:-1])
    if not hidden_dims:
        raise ValueError("Checkpoint head has no hidden layer.")
    return hidden_dims


def predict_tweet_impacts(
    tweets: pd.DataFrame, text_col: str, tokenizer: Any, model_class: type[nn.Module],
    checkpoint_paths: Sequence[str | Path], target_names: Sequence[str],
    device: str | torch.device, batch_size: int, max_length: int,
) -> pd.DataFrame:
    """Average fold predictions in dataframe order, applying sigmoid at most once."""
    if tweets.empty:
        raise ValueError("Tweet dataframe is empty.")
    if text_col not in tweets:
        raise KeyError(f"Missing text column: {text_col}")
    if not target_names:
        raise ValueError("target_names is empty.")
    paths = [Path(p) for p in checkpoint_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files: " + ", ".join(missing))
    if not paths:
        raise ValueError("No checkpoint paths supplied.")
    loader = DataLoader(
        TweetInferenceDataset(tweets[text_col].astype(str).tolist(), tokenizer, max_length),
        batch_size=batch_size, shuffle=False,
    )
    device = torch.device(device)
    folds = []
    for path in paths:
        state = _load_state_dict(path, device)
        head_hidden_dims = _infer_checkpoint_head_hidden_dims(state, len(target_names))
        try:
            model = model_class(
                num_targets=len(target_names),
                head_hidden_dims=head_hidden_dims,
            ).to(device)
        except TypeError:
            # Compatibility path for a user-supplied model class. Such a class must
            # already match the checkpoint because strict loading remains enabled.
            model = model_class(len(target_names)).to(device)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint architecture mismatch for {path}. "
                f"Inferred hidden dimensions: {head_hidden_dims}. "
                "Confirm that all folds came from the same Harvey model implementation."
            ) from exc
        if hasattr(model, "heads") and len(model.heads) != len(target_names):
            raise ValueError(f"Target-count mismatch for {path}.")
        print(f"Loaded {path.name}: head hidden dimensions={head_hidden_dims}")
        model.eval()
        batches = []
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                if output.ndim != 2 or output.shape[1] != len(target_names):
                    raise ValueError(
                        f"Model-output shape mismatch: {tuple(output.shape)}; "
                        f"expected (batch, {len(target_names)})."
                    )
                if getattr(model, "outputs_are_logits", False):
                    output = torch.sigmoid(output)
                batches.append(output.detach().cpu().numpy())
        fold_pred = np.concatenate(batches).astype(np.float32)
        if fold_pred.shape[0] != len(tweets):
            raise ValueError("Prediction/dataframe row mismatch.")
        folds.append(fold_pred)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    averaged = np.mean(np.stack(folds), axis=0, dtype=np.float32)
    if not np.isfinite(averaged).all():
        raise ValueError("Model predictions contain NaN or infinity.")
    result = pd.DataFrame(averaged, columns=list(target_names), index=tweets.index)
    print("Prediction shape:", result.shape)
    print(result.describe().T)
    return result


def rasterize_semantic_predictions(
    tweets: pd.DataFrame, prediction_columns: Sequence[str], nrows: int, ncols: int,
    aggregation: str = "mean",
):
    """Distribute weighted K-vectors uniformly across inclusive tweet boxes."""
    if aggregation != "mean":
        raise ValueError("Only aggregation='mean' is currently supported.")
    required = list(prediction_columns) + [
        "row_min", "row_max", "col_min", "col_max", "spatial_precision_weight"
    ]
    missing = [c for c in required if c not in tweets]
    if missing:
        raise KeyError(f"Missing rasterization columns: {missing}")
    semantic_sum = np.zeros((len(prediction_columns), nrows, ncols), np.float32)
    semantic_weight = np.zeros((nrows, ncols), np.float32)
    tweet_count = np.zeros((nrows, ncols), np.float32)
    for row in tweets.itertuples(index=False):
        r0, r1 = int(row.row_min), int(row.row_max)
        c0, c1 = int(row.col_min), int(row.col_max)
        if not (0 <= r0 <= r1 < nrows and 0 <= c0 <= c1 < ncols):
            raise ValueError("Invalid raster indices.")
        vector = np.asarray([getattr(row, c) for c in prediction_columns], np.float32)
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite semantic prediction.")
        covered = (r1 - r0 + 1) * (c1 - c0 + 1)
        weight = np.float32(row.spatial_precision_weight) / np.float32(covered)
        semantic_sum[:, r0:r1 + 1, c0:c1 + 1] += vector[:, None, None] * weight
        semantic_weight[r0:r1 + 1, c0:c1 + 1] += weight
        tweet_count[r0:r1 + 1, c0:c1 + 1] += 1
    semantic_mean = np.zeros_like(semantic_sum)
    np.divide(
        semantic_sum, semantic_weight[None, :, :], out=semantic_mean,
        where=semantic_weight[None, :, :] > 0,
    )
    return semantic_sum, semantic_weight, semantic_mean, tweet_count


def write_multiband_geotiff(
    path: str | Path, array: np.ndarray, transform: Any, crs: Any,
    band_names: Sequence[str], nodata: float = 0.0,
) -> Path:
    """Write a channel-first float32 cube with descriptions and QC statistics."""
    import rasterio
    cube = np.asarray(array, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None]
    if cube.ndim != 3 or cube.shape[0] != len(band_names):
        raise ValueError("Array must be (bands, height, width), matching band_names.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=cube.shape[1], width=cube.shape[2],
        count=cube.shape[0], dtype="float32", crs=crs, transform=transform,
        nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(cube)
        for band, name in enumerate(band_names, 1):
            dst.set_band_description(band, str(name))
    print(f"Saved: {path} | dimensions={cube.shape[2]}x{cube.shape[1]} | bands={cube.shape[0]}")
    for name, band in zip(band_names, cube):
        finite = band[np.isfinite(band)]
        stats = (finite.min(), finite.max(), finite.mean(), finite.std()) if finite.size else (np.nan,) * 4
        print(f"{name}: min={stats[0]:.6g}, max={stats[1]:.6g}, mean={stats[2]:.6g}, "
              f"std={stats[3]:.6g}, nonzero={100*np.count_nonzero(band)/band.size:.2f}%")
    return path


def resample_multiband_to_match(
    source: np.ndarray, src_transform: Any, src_crs: Any, target_profile: dict,
    resampling: Any = None,
) -> np.ndarray:
    """Align a channel-first cube to a reference grid, skipping identical grids."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    cube = np.asarray(source, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None]
    target_shape = (int(target_profile["height"]), int(target_profile["width"]))
    dst_transform, dst_crs = target_profile["transform"], target_profile["crs"]
    same = (
        cube.shape[1:] == target_shape and src_crs == dst_crs
        and src_transform.almost_equals(dst_transform)
    )
    print("Source:", cube.shape, src_crs, src_transform)
    print("Target:", (cube.shape[0], *target_shape), dst_crs, dst_transform)
    if same:
        print("Grids already match; resampling skipped.")
        return cube
    method = Resampling.bilinear if resampling is None else resampling
    destination = np.zeros((cube.shape[0], *target_shape), np.float32)
    for band in range(cube.shape[0]):
        reproject(
            cube[band], destination[band], src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs, src_nodata=np.nan,
            dst_nodata=0.0, resampling=method,
        )
    if destination.shape[1:] != target_shape:
        raise ValueError("Semantic raster remains misaligned.")
    return destination


def normalize_channels(array: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score each channel independently and safely."""
    cube = np.asarray(array, dtype=np.float32)
    if cube.ndim == 2:
        cube = cube[None]
    out = np.zeros_like(cube)
    for i, channel in enumerate(cube):
        mask = np.isfinite(channel)
        if valid_mask is not None:
            mask &= np.asarray(valid_mask, dtype=bool)
        if not mask.any():
            continue
        mean, std = float(channel[mask].mean()), float(channel[mask].std())
        if np.isfinite(std) and std > 1e-6:
            out[i] = (channel - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
