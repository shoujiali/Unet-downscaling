# Change log

## Preserved

- Original Texas extent (−107 to −93, 25 to 37) and 0.025-degree grid
- Tweet bounding-box-to-grid convention
- Five environmental predictors and DEM-derived slope
- Three-level U-Net, non-negative output, masked MSE, and smoothness penalty
- Spatial block five-fold validation, fresh model per fold, and early stopping
- Overlap-averaged full-map inference, Texas boundary mask, and GeoTIFF profile handling

## Replaced

- Scattered constants and paths with one configuration dictionary
- Single tweet-count/flood-relevance U-Net feature with configurable multi-head semantic inputs
- Duplicate notebook class/function definitions with two reusable modules
- Fixed `in_channels = 6` with dynamic input-channel discovery
- Ad hoc fold training with a reusable ablation-capable CV function

## Added

- Stable tweet IDs, text validation, bbox clipping, EPSG:3083 area, and precision weights
- Five-checkpoint batched ensemble inference with shape, file, and finite-value checks
- Strict checkpoint-driven support for both observed `768→512→128→1` and `768→256→1` heads
- Weighted multichannel rasterization and named multiband GeoTIFF output
- Multiband alignment, per-channel normalization, semantic diagnostics, and five ablations
- Fold histories, summary statistics, defensive validation, and CUDA OOM guidance

## Assumptions

- Harvey checkpoint state dictionaries match the exact architecture in the supplied notebook.
- Model heads already include sigmoid; inference must not apply it again.
- Continuous semantic channels may be bilinearly resampled when grids differ.
- Uniform distribution within each bounding box is an acceptable first spatial allocation.

## Unresolved issues

- Private tweet, environmental, DEM, boundary, and checkpoint files are not present here.
- E2 cannot run until the existing single flood-relevance raster path is supplied.
- Full end-to-end training and output validation require those external assets and compute.
- Training-fold-only normalization remains recommended future work for strict CV.
