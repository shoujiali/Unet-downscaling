# Hurricane Flood Depth Downscaling Using Deep Learning

## Overview

This repository presents a deep learning framework for predicting hurricane-induced flood water depth using multiple environmental and meteorological variables. The workflow integrates precipitation, temperature, wind, storm surge, and terrain information to estimate localized flood depth at the ZIP Code level.

The model is implemented using a U-Net convolutional neural network that learns the nonlinear relationship between hurricane characteristics and observed flood depth. The notebook provides an end-to-end pipeline including data preparation, model training, prediction, and visualization.

---

## Workflow

```
Environmental Datasets
        │
        ▼
 Data Preparation
        │
        ▼
 Feature Engineering
        │
        ▼
   U-Net Training
        │
        ▼
 Flood Depth Prediction
        │
        ▼
 Visualization & Evaluation
```

---

## Data Sources

The model combines observed flood information with multiple environmental datasets describing hurricane characteristics and terrain conditions.

### Target Variable

**Observed Water Depth**

Observed flood water depth was obtained from the **OpenFEMA** Individual Assistance Housing Registrants dataset.

- Variable: **waterDepth**
- Unit: Inches
- Description: Reported flood water depth associated with FEMA assistance records. Most observations are recorded in inches, while a small number were originally reported in feet and converted during preprocessing.
- Geographic identifiers include ZIP Code, Census Tract, Census Block Group, and Census Block.

Dataset DOI: https://doi.org/10.7266/n90stx2y

---

### Predictor Variables

| Variable | Source | Description |
|----------|---------|------------|
| Hurricane_Totals | PRISM Climate Group | Total accumulated precipitation during each hurricane event |
| Hurricane_Wind | NOAA NCEI | Maximum wind speed |
| Hurricane_TMax | NOAA NCEI | Maximum air temperature |
| Hurricane_TMin | NOAA NCEI | Minimum air temperature |
| Hurricane_Pre | NOAA NCEI | Total precipitation measured by weather stations |
| Hurricane_CERA | Coastal Emergency Risks Assessment (CERA) | Maximum modeled storm surge inundation |
| Elevation | USGS 1-arcsecond Digital Elevation Model | Bare-earth elevation |

---

### Geographic Processing

ZIP Code boundaries were standardized using both the **2010** and **2020 Census ZIP Code Tabulation Areas (ZCTAs)** obtained from the ESRI Federal Data collection.

- Hurricanes occurring before 2020 were matched using the 2010 ZIP Code boundaries.
- Hurricanes occurring during and after 2020 used the updated 2020 ZIP Code boundaries.

This approach ensures spatial consistency across historical hurricane events.

---

## Data Preparation

Prior to model training, all datasets were processed into a common analysis framework.

### Precipitation

Daily precipitation rasters were downloaded from the **PRISM Climate Group** for the duration of each hurricane event.

The rasters were

- clipped to the Texas state boundary,
- accumulated using raster calculations,
- converted into total storm precipitation for each hurricane.

---

### Weather Variables

Weather observations were obtained from the **National Centers for Environmental Information (NCEI)** for all NOAA climate stations in Texas between **2010 and 2024**.

For each hurricane event, the following statistics were calculated:

- Maximum wind speed
- Maximum temperature
- Minimum temperature
- Total precipitation

Continuous raster surfaces were generated using **Inverse Distance Weighting (IDW)** interpolation. Only stations with valid (>0) observations were included during interpolation.

---

### Storm Surge

Maximum storm surge inundation layers were obtained from the **Coastal Emergency Risks Assessment (CERA)** program.

Each hurricane is represented by a single raster describing the maximum modeled coastal inundation depth during the entire event.

---

### Elevation

Terrain information was derived from the **USGS 1-arcsecond (~30 m) Digital Elevation Model**, which primarily incorporates LiDAR-derived elevation where available.

---

### Feature Preparation

Before model training, the notebook performs several preprocessing operations, including

- loading environmental rasters,
- aligning spatial datasets,
- handling missing values,
- feature normalization,
- constructing predictor matrices,
- preparing training samples for deep learning.

---

## Deep Learning Model

Flood depth estimation is formulated as a supervised image regression problem.

A **U-Net** convolutional neural network is trained to learn the relationship between environmental conditions and observed flood depth.

### Input Features

The U-Net model uses six raster-based input channels:

1. **Coarse-resolution water depth** (`water_01`)  
   Flood water depth aggregated at a spatial resolution of 0.1°.

2. **Cumulative precipitation** (`precip`)  
   Total storm precipitation derived from PRISM climate data.

3. **Maximum wind speed** (`wind`)  
   Maximum hurricane-related wind speed over the event duration.

4. **Elevation** (`DEM`)  
   Bare-earth elevation derived from the USGS Digital Elevation Model.

5. **Slope**  
   Terrain slope calculated from the elevation raster.

6. **Tweet density**  
   A raster representation of geolocated tweet counts or density.

### Target Variable

The prediction target is the observed flood water depth raster at a spatial
resolution of 0.025°.

![1](https://github.com/shoujiali/Disaster-Impact-Estimation-Downscaling/blob/main/Unet/workflow.png)

---

## Prediction

After training, the model predicts flood water depth using the same environmental predictor variables.

The prediction workflow consists of

1. Loading the trained model.
2. Preparing predictor variables.
3. Generating prediction inputs.
4. Estimating flood water depth.
5. Producing spatial prediction maps.
6. Comparing predicted and observed flood depths.

---

## Outputs

The notebook produces

- Predicted flood depth maps

![1](https://github.com/shoujiali/Disaster-Impact-Estimation-Downscaling/blob/main/Unet/Unet%20predicaition.png)

---

## Repository Structure

```
.
├── Downscale.ipynb          # Complete modeling workflow
└── README.md
```

---
