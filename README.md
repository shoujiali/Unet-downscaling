# DIEP semantic flood downscaling

This project converts geolocated Hurricane Harvey tweet text into 11 interpretable
disaster-impact probability rasters, then combines them with environmental variables in a
supervised U-Net that predicts fine-resolution flood depth.

## Workflow

Tweet text is cleaned and passed through the five Harvey BERT multi-head checkpoints. Fold
probabilities are averaged in tweet order, spatially weighted by bounding-box precision,
distributed across covered cells, and saved as a channel-first semantic cube. The cube is
aligned with the flood target grid, normalized per channel, and concatenated with coarse water
depth, precipitation, wind, DEM, and slope. The original spatial block five-fold workflow,
masked MSE plus smoothness loss, early stopping, overlap prediction, Texas clipping, and
GeoTIFF export are retained.

No raw 768-dimensional BERT embeddings are used as U-Net inputs.

## Inputs

Set all paths in the first configuration cell of
`DIEP_semantic_downscaling.ipynb`. The tweet table must contain:

- `text` (configurable)
- `lon_min`, `lat_min`, `lon_max`, `lat_max` (configurable), or the original `place` field

Environmental inputs are the 0.025-degree target depth, 0.1-degree coarse depth,
precipitation, wind, one or more DEM tiles, and a Texas boundary dataset. An optional existing
single flood-relevance raster enables experiment E2.

## Harvey model files

The source Harvey notebook defines `bert-base-uncased`, 11 sigmoid heads, and the checkpoint
convention:

```text
Training_results/Harvey_5foldcv_BERT_Multihead_Deeper_512/
  bert_model_fold_1.pth
  ...
  bert_model_fold_5.pth
```

The target order is:

```text
homeOwnersInsurance, floodInsurance, destroyed, floodDamage, roofDamage,
tsaEligible, tsaCheckedIn, rentalAssistanceEligible, repairAssistanceEligible,
replacementAssistanceEligible, personalPropertyEligible
```

The training notebook shows a `768→512→128→1` head, but some exported fold files use
`768→256→1`. The loader detects the hidden dimensions from each checkpoint and then performs
strict loading. It does not silently ignore missing or unexpected parameters.

## Running

1. Install `requirements.txt` (Colab: `pip install -r requirements.txt`).
2. Place `semantic_tweet_raster.py` and `flood_unet.py` beside the notebook.
3. Fill unresolved paths in the central `CONFIG` dictionary.
4. Run the notebook top to bottom. Missing private inputs fail early with explicit messages.

The code is compatible with CPU and CUDA. If CUDA memory is exhausted, lower
`unet_batch_size` or `patch_size`.

## Raster outputs

Outputs use the target grid and channel-first arrays internally. Semantic mean and optional
sum cubes are multiband float32, LZW-compressed GeoTIFFs with one named band per target.
Tweet count and semantic weight are single-band diagnostics. The final clipped flood-depth
prediction uses the original target profile with `-9999` outside valid coverage.

## Ablations

- E0: environmental variables
- E1: environmental variables + tweet count
- E2: environmental variables + existing single flood-relevance raster
- E3: environmental variables + 11 BERT semantic channels
- E4: environmental variables + BERT semantic channels + tweet count

Fold metrics include best epoch, MAE, RMSE, and R². A second table reports mean and standard
deviation by experiment.

## Methodological limitation

This is a **tweet-level application of a model trained using ZIP-level aggregated
disaster-impact labels**. It changes the inference unit and is not equivalent to tweet-level
training or validation. Semantic outputs should be treated as model-derived proxy features.
Global normalization is provided as a first implementation; training-fold-only statistics are
preferable for strict cross-validation.
