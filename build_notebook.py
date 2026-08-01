"""Build DIEP_semantic_downscaling.ipynb from reviewed, reusable modules."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md("""# DIEP Semantic Disaster-Impact Flood Downscaling

This refactor preserves the source notebook's environmental preprocessing, three-level U-Net,
masked MSE plus smoothness objective, spatial block five-fold validation, overlap prediction,
Texas clipping, and GeoTIFF export. It replaces the single tweet feature with interpretable
Harvey BERT multi-head disaster-impact probability rasters and adds controlled ablations.

Cells that need private datasets or checkpoints are explicitly guarded. Fill the central
configuration only; do not scatter paths through later cells.""")

md("## 1. Configuration and Imports")
code("""from pathlib import Path
import ast, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject
import torch

from semantic_tweet_raster import (
    HARVEY_TARGET_NAMES, BERTMultiDeepHeadClassifier, clean_tweet_text,
    normalize_channels, predict_tweet_impacts, prepare_tweet_bounding_boxes,
    rasterize_semantic_predictions, resample_multiband_to_match,
    set_reproducibility, write_multiband_geotiff,
)
from flood_unet import (
    UNet, build_spatial_fold_map, predict_full_map_overlap,
    run_spatial_cv_experiment, train_final_model,
)

# None means unresolved: set it here before running dependent sections.
CONFIG = {
    "tweet_csv": None,
    "water_depth_target_0025": None,
    "coarse_water_depth_01": None,
    "precipitation_raster": None,
    "wind_raster": None,
    "dem_tiles": [],
    "texas_boundary_file": None,
    # Explicitly preserved from harvey_bert_multihead_deep_5fold.ipynb:
    "model_checkpoint_dir": Path("Training_results/Harvey_5foldcv_BERT_Multihead_Deeper_512"),
    "checkpoint_pattern": "bert_model_fold_{fold}.pth",
    "tokenizer_name": "bert-base-uncased",
    "text_col": "text",
    "place_col": "place",
    "lon_min_col": "lon_min", "lat_min_col": "lat_min",
    "lon_max_col": "lon_max", "lat_max_col": "lat_max",
    "texas_extent": (-107.0, -93.0, 25.0, 37.0),  # lon min/max, lat min/max
    "grid_resolution": 0.025,
    "target_names": HARVEY_TARGET_NAMES,
    "inference_batch_size": 24,
    "max_token_length": 512,
    "num_folds": 5,
    "patch_size": 64, "stride": 32,
    "unet_batch_size": 4,
    "learning_rate": 1e-4, "epochs": 50,
    "early_stopping_patience": 10, "min_delta": 1e-4,
    "smoothness_weight": 0.001,
    "random_seed": 42,
    "semantic_mean_path": Path("outputs/texas_semantic_mean_0025.tif"),
    "semantic_sum_path": Path("outputs/texas_semantic_sum_0025.tif"),
    "semantic_weight_path": Path("outputs/texas_semantic_weight_0025.tif"),
    "tweet_count_path": Path("outputs/texas_tweet_count_0025.tif"),
    "cv_results_path": Path("outputs/ablation_fold_results.csv"),
    "cv_summary_path": Path("outputs/ablation_summary.csv"),
    "final_model_path": Path("outputs/final_unet_all_data.pth"),
    "prediction_path": Path("outputs/pred_full_texas.tif"),
    "model_output_dir": Path("outputs/models"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG["device"] = DEVICE
print("Device:", DEVICE)""")

md("## 2. Reproducibility")
code("""set_reproducibility(CONFIG["random_seed"])
print("Seed:", CONFIG["random_seed"])""")

md("## 3. Load Tweets")
code("""def require_path(value, label):
    if value is None:
        raise FileNotFoundError(f"Set CONFIG[{label!r}] before running this section.")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path

def parse_bbox(value):
    \"\"\"Parse the source notebook's dict-like `place` or four-number bbox.\"\"\"
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, dict) and len(parsed.get("bbox", [])) == 4:
            return tuple(map(float, parsed["bbox"]))
    except (ValueError, SyntaxError, TypeError):
        pass
    try:
        parts = [float(x.strip("()[] '\\\"")) for x in str(value).split(",")]
        return tuple(parts) if len(parts) == 4 else None
    except ValueError:
        return None

tweet_path = require_path(CONFIG["tweet_csv"], "tweet_csv")
raw_tweets = pd.read_csv(tweet_path, low_memory=False)
if raw_tweets.empty:
    raise ValueError("Tweet CSV is empty.")
if CONFIG["place_col"] in raw_tweets and not {
    CONFIG["lon_min_col"], CONFIG["lat_min_col"], CONFIG["lon_max_col"], CONFIG["lat_max_col"]
}.issubset(raw_tweets.columns):
    parsed = raw_tweets[CONFIG["place_col"]].map(parse_bbox)
    valid = parsed.notna()
    raw_tweets = raw_tweets.loc[valid].copy()
    raw_tweets[[CONFIG["lon_min_col"], CONFIG["lat_min_col"],
                CONFIG["lon_max_col"], CONFIG["lat_max_col"]]] = pd.DataFrame(
        parsed.loc[valid].tolist(), index=raw_tweets.index
    )
print("Loaded tweets:", len(raw_tweets))""")

md("## 4. Clean Text and Bounding Boxes")
code("""tweets, nrows, ncols, texas_transform, texas_crs = prepare_tweet_bounding_boxes(
    raw_tweets, CONFIG["text_col"], CONFIG["lon_min_col"], CONFIG["lat_min_col"],
    CONFIG["lon_max_col"], CONFIG["lat_max_col"], CONFIG["texas_extent"],
    CONFIG["grid_resolution"],
)
assert tweets["tweet_uid"].is_unique
print(tweets[["tweet_uid", "clean_text", "tweet_box_area_km2", "spatial_precision_weight"]].head())
print("Clean tweets:", len(tweets), "| grid:", nrows, "x", ncols)""")

md("""## 5. Load BERT Multi-Head Model

The target order is preserved exactly from `harvey_bert_multihead_deep_5fold.ipynb`.
The checked-in loader also supports the two observed Harvey checkpoint head layouts:
`768→512→128→1` from the notebook and exported `768→256→1` fold files. It infers the
hidden dimensions from each checkpoint and still uses strict state-dict loading, so unrelated
or inconsistent checkpoints fail clearly. Both layouts end each head with a sigmoid, so
`BERTMultiDeepHeadClassifier.outputs_are_logits = False`; inference does not apply a second
sigmoid.""")
code("""from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(CONFIG["tokenizer_name"])
checkpoint_paths = [
    Path(CONFIG["model_checkpoint_dir"]) / CONFIG["checkpoint_pattern"].format(fold=fold)
    for fold in range(1, CONFIG["num_folds"] + 1)
]
missing_checkpoints = [str(p) for p in checkpoint_paths if not p.is_file()]
if missing_checkpoints:
    raise FileNotFoundError("Missing Harvey fold checkpoints: " + ", ".join(missing_checkpoints))
print(*checkpoint_paths, sep="\\n")""")

md("""## 6. Tweet-Level Disaster-Impact Inference

> **Methodological limitation:** this is a **tweet-level application of a model trained using
> ZIP-level aggregated disaster-impact labels**. The inference unit has changed. These
> probabilities must not be described as equivalent to predictions from a model trained and
> validated on individual tweet labels.""")
code("""impact_predictions = predict_tweet_impacts(
    tweets=tweets, text_col="clean_text", tokenizer=tokenizer,
    model_class=BERTMultiDeepHeadClassifier, checkpoint_paths=checkpoint_paths,
    target_names=CONFIG["target_names"], device=DEVICE,
    batch_size=CONFIG["inference_batch_size"], max_length=CONFIG["max_token_length"],
)
if len(impact_predictions) != len(tweets):
    raise ValueError("Prediction/dataframe row mismatch.")
tweets = tweets.join(impact_predictions)
assert np.isfinite(tweets[CONFIG["target_names"]].to_numpy()).all()""")

md("## 7. Semantic Rasterization")
code("""semantic_sum, semantic_weight, semantic_mean, tweet_count_raster = rasterize_semantic_predictions(
    tweets, CONFIG["target_names"], nrows, ncols, aggregation="mean"
)
assert semantic_mean.shape == (len(CONFIG["target_names"]), nrows, ncols)
write_multiband_geotiff(CONFIG["semantic_mean_path"], semantic_mean, texas_transform, texas_crs, CONFIG["target_names"])
write_multiband_geotiff(CONFIG["semantic_weight_path"], semantic_weight, texas_transform, texas_crs, ["semantic_weight"])
write_multiband_geotiff(CONFIG["tweet_count_path"], tweet_count_raster, texas_transform, texas_crs, ["tweet_count"])
# Optional diagnostic sum cube:
write_multiband_geotiff(CONFIG["semantic_sum_path"], semantic_sum, texas_transform, texas_crs, CONFIG["target_names"])""")

md("## 8. Semantic Raster Quality Control")
code("""extent_plot = [CONFIG["texas_extent"][0], CONFIG["texas_extent"][1],
               CONFIG["texas_extent"][2], CONFIG["texas_extent"][3]]
def show_raster(array, title, cmap="viridis"):
    plt.figure(figsize=(10, 7))
    plt.imshow(array, extent=extent_plot, origin="upper", cmap=cmap)
    plt.title(title); plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.colorbar(shrink=.8); plt.tight_layout(); plt.show()

show_raster(tweet_count_raster, "Tweet count", "magma")
show_raster(semantic_weight, "Semantic spatial weight", "magma")
for name, band in zip(CONFIG["target_names"], semantic_mean):
    show_raster(band, name)
show_raster(semantic_mean.sum(axis=0), "Total semantic activity", "magma")
for selected in ("floodDamage", "destroyed", "roofDamage", "repairAssistanceEligible"):
    if selected in CONFIG["target_names"]:
        show_raster(semantic_mean[CONFIG["target_names"].index(selected)], f"Selected: {selected}")""")

md("## 9. Environmental Raster Preprocessing")
code("""def read_raster(path_value, label):
    path = require_path(path_value, label)
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        profile, transform, crs, nodata = src.profile.copy(), src.transform, src.crs, src.nodata
    if nodata is not None:
        array[array == nodata] = np.nan
    array[array <= -9999] = np.nan
    return array, profile, transform, crs

def resample_single(array, transform, crs, target_profile):
    cube = resample_multiband_to_match(array[None], transform, crs, target_profile)
    return cube[0]

water_025, target_profile, water_025_transform, water_025_crs = read_raster(
    CONFIG["water_depth_target_0025"], "water_depth_target_0025")
water_01, _, water_01_transform, water_01_crs = read_raster(
    CONFIG["coarse_water_depth_01"], "coarse_water_depth_01")
precip_raw, _, precip_transform, precip_crs = read_raster(
    CONFIG["precipitation_raster"], "precipitation_raster")
wind_raw, _, wind_transform, wind_crs = read_raster(CONFIG["wind_raster"], "wind_raster")

def build_dem_resampled_from_tiles(paths, profile):
    if not paths:
        raise FileNotFoundError("Set CONFIG['dem_tiles'] to one or more DEM GeoTIFFs.")
    total = np.zeros((profile["height"], profile["width"]), np.float32)
    count = np.zeros_like(total)
    for path in paths:
        dem, _, transform, crs = read_raster(path, "dem_tile")
        aligned = resample_single(dem, transform, crs, profile)
        valid = np.isfinite(aligned); total[valid] += aligned[valid]; count[valid] += 1
    result = np.full_like(total, np.nan); np.divide(total, count, out=result, where=count > 0)
    return result

water_01_up = resample_single(water_01, water_01_transform, water_01_crs, target_profile)
precip_025 = resample_single(precip_raw, precip_transform, precip_crs, target_profile)
wind_025 = resample_single(wind_raw, wind_transform, wind_crs, target_profile)
dem_025 = build_dem_resampled_from_tiles(CONFIG["dem_tiles"], target_profile)
dy, dx = np.gradient(dem_025, 550.0)
slope_025 = np.sqrt(dx**2 + dy**2).astype(np.float32)""")

md("## 10. Raster Alignment")
code("""semantic_aligned = resample_multiband_to_match(
    semantic_mean, texas_transform, texas_crs, target_profile, Resampling.bilinear)
tweet_count_aligned = resample_multiband_to_match(
    tweet_count_raster, texas_transform, texas_crs, target_profile, Resampling.bilinear)[0]
semantic_weight_aligned = resample_multiband_to_match(
    semantic_weight, texas_transform, texas_crs, target_profile, Resampling.bilinear)[0]
print("Semantic:", semantic_aligned.shape, target_profile["crs"], target_profile["transform"])
print("Target:", water_025.shape, water_025_crs, water_025_transform)
if semantic_aligned.shape[1:] != water_025.shape:
    raise ValueError("Semantic and target arrays remain misaligned.")
if texas_crs != water_025_crs or not texas_transform.almost_equals(water_025_transform):
    print("Source semantic grid differed; aligned output now uses the target profile.")
assert target_profile["width"] == water_025.shape[1] and target_profile["height"] == water_025.shape[0]""")

md("""## 11. Feature Normalization

This first working implementation uses global per-channel z-scores. For strict spatial
cross-validation, fit normalization statistics on training-fold pixels only and apply those
statistics unchanged to the validation fold.""")
code("""environmental_raw = np.stack([water_01_up, precip_025, wind_025, dem_025, slope_025])
environmental_features = normalize_channels(environmental_raw)
semantic_features_norm = normalize_channels(semantic_aligned)
tweet_count_norm = normalize_channels(tweet_count_aligned)[0]

# Optional E2 input: load the existing single flood-relevance raster if available.
single_flood_relevance_norm = None
if CONFIG.get("single_flood_relevance_raster") is not None:
    sfr, _, sfr_transform, sfr_crs = read_raster(
        CONFIG["single_flood_relevance_raster"], "single_flood_relevance_raster")
    single_flood_relevance_norm = normalize_channels(
        resample_single(sfr, sfr_transform, sfr_crs, target_profile))[0]

X_full = np.concatenate([environmental_features, semantic_features_norm], axis=0).astype(np.float32)
y_full = water_025.astype(np.float32)
channel_names = ["coarse_water_depth", "precipitation", "wind", "DEM", "slope"] + list(CONFIG["target_names"])
print("Environmental feature shape:", environmental_features.shape)
print("Semantic feature shape:", semantic_features_norm.shape)
print("Final X_full shape:", X_full.shape)
print("Target y_full shape:", y_full.shape)
print("Channel names:", channel_names)
assert all(a.shape == y_full.shape for a in environmental_raw)
assert X_full.shape[1:] == y_full.shape
assert len(channel_names) == X_full.shape[0]
assert not np.isinf(X_full).any()
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)""")

md("## 12. U-Net Definition")
code("""model_shape_check = UNet(in_channels=X_full.shape[0]).to(DEVICE)
with torch.no_grad():
    sample_output = model_shape_check(
        torch.zeros(1, X_full.shape[0], CONFIG["patch_size"], CONFIG["patch_size"], device=DEVICE))
assert sample_output.shape == (1, 1, CONFIG["patch_size"], CONFIG["patch_size"])
del model_shape_check
if torch.cuda.is_available(): torch.cuda.empty_cache()
print("Dynamic in_channels:", X_full.shape[0])""")

md("## 13. Spatial Cross-Validation")
code("""# Target leakage guard: only coarse depth is in X; the fine-resolution target is y.
assert "fine_water_depth_target" not in channel_names
fold_map = build_spatial_fold_map(
    y_full, n_folds=CONFIG["num_folds"], block_size=CONFIG["patch_size"],
    seed=CONFIG["random_seed"])
show_raster(fold_map, "Spatial five-fold map", "tab10")""")

md("## 14. Ablation Experiments")
code("""experiments = {
    "E0_environmental": environmental_features,
    "E1_environmental_tweet_count": np.concatenate(
        [environmental_features, tweet_count_norm[None]], axis=0),
    "E3_environmental_semantic": np.concatenate(
        [environmental_features, semantic_features_norm], axis=0),
    "E4_environmental_semantic_tweet_count": np.concatenate(
        [environmental_features, semantic_features_norm, tweet_count_norm[None]], axis=0),
}
if single_flood_relevance_norm is None:
    warnings.warn("E2 skipped: set CONFIG['single_flood_relevance_raster'] to enable it.")
else:
    experiments["E2_environmental_flood_relevance"] = np.concatenate(
        [environmental_features, single_flood_relevance_norm[None]], axis=0)

all_results, all_histories, all_states = [], {}, {}
for name, features in experiments.items():
    result, histories, states = run_spatial_cv_experiment(name, features, y_full, fold_map, CONFIG)
    all_results.append(result); all_histories[name] = histories; all_states[name] = states
comparison = pd.concat(all_results, ignore_index=True)
summary = comparison.groupby("experiment").agg(
    mean_MAE=("MAE", "mean"), std_MAE=("MAE", "std"),
    mean_RMSE=("RMSE", "mean"), std_RMSE=("RMSE", "std"),
    mean_R2=("R2", "mean"), std_R2=("R2", "std"),
).reset_index()
CONFIG["cv_results_path"].parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(CONFIG["cv_results_path"], index=False)
summary.to_csv(CONFIG["cv_summary_path"], index=False)
display(comparison); display(summary)""")

md("## 15. Final Model Training")
code("""final_X = experiments["E4_environmental_semantic_tweet_count"]
final_model, final_loss_history = train_final_model(final_X, y_full, CONFIG)
CONFIG["final_model_path"].parent.mkdir(parents=True, exist_ok=True)
torch.save(final_model.state_dict(), CONFIG["final_model_path"])
plt.figure(figsize=(9, 5)); plt.plot(final_loss_history)
plt.xlabel("Epoch"); plt.ylabel("Training loss"); plt.title("Final-model loss"); plt.grid(True); plt.show()""")

md("## 16. Full-Map Overlap Prediction")
code("""pred_full = predict_full_map_overlap(
    final_model, final_X, CONFIG["patch_size"], CONFIG["stride"],
    CONFIG["unet_batch_size"], DEVICE)
assert pred_full.shape == y_full.shape
show_raster(pred_full, "Full-map overlap prediction", "turbo")""")

md("## 17. Texas Boundary Clipping")
code("""import geopandas as gpd
boundary_path = require_path(CONFIG["texas_boundary_file"], "texas_boundary_file")
texas = gpd.read_file(boundary_path).to_crs(water_025_crs)
texas_union = texas.geometry.union_all() if hasattr(texas.geometry, "union_all") else texas.unary_union
texas_mask = geometry_mask([texas_union], out_shape=pred_full.shape,
                           transform=water_025_transform, invert=True)
pred_full_tx = np.where(texas_mask, pred_full, np.nan)
show_raster(pred_full_tx, "Prediction clipped to Texas", "turbo")""")

md("## 18. GeoTIFF Export")
code("""profile = target_profile.copy()
profile.update(driver="GTiff", dtype="float32", count=1, nodata=-9999.0, compress="lzw")
CONFIG["prediction_path"].parent.mkdir(parents=True, exist_ok=True)
saved = np.where(np.isfinite(pred_full_tx), pred_full_tx, -9999.0).astype(np.float32)
with rasterio.open(CONFIG["prediction_path"], "w", **profile) as dst:
    dst.write(saved, 1)
    dst.set_band_description(1, "U-Net fine-resolution flood depth")
print("Saved:", CONFIG["prediction_path"])""")

md("## 19. Results and Diagnostics")
code("""error = pred_full - y_full
valid = np.isfinite(error)
print("Prediction finite cells:", np.isfinite(pred_full).sum())
if valid.any():
    print("Full-map MAE:", np.mean(np.abs(error[valid])))
    print("Full-map RMSE:", np.sqrt(np.mean(error[valid] ** 2)))
show_raster(error, "Prediction error", "coolwarm")
for experiment, folds in all_histories.items():
    plt.figure(figsize=(9, 5))
    for fold, history in folds.items():
        plt.plot(history["val_rmse"], label=f"fold {fold}")
    plt.title(f"{experiment}: validation RMSE"); plt.xlabel("Epoch"); plt.ylabel("RMSE")
    plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()""")

md("""## 20. Methodological Limitations

- This is a **tweet-level application of a model trained using ZIP-level aggregated
  disaster-impact labels**. The inference unit differs from training.
- Bounding-box distribution assumes uniform spatial support inside each clipped box.
- Bilinear resampling smooths continuous semantic channels if source and target grids differ.
- Global normalization is implemented for the first working version; training-fold-only
  statistics are preferable for strict leakage-free cross-validation.
- Tweet availability, geolocation precision, demographic/platform bias, and event-time
  mismatch can all affect downstream estimates.
- The semantic model outputs correlated proxy features, not causal measurements of damage.

No raw 768-dimensional BERT embeddings enter the U-Net; only the 11 interpretable head
probabilities are used.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}
nbf.write(nb, "DIEP_semantic_downscaling.ipynb")
print(f"Wrote DIEP_semantic_downscaling.ipynb with {len(cells)} cells")
