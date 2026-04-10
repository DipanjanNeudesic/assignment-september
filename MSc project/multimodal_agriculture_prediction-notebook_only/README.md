# Sustainable Crop Yield Prediction via Multi-Modal AI

**MSc AI Dissertation Project — University of Hull**
**Student:** Dipanjan Das (202441750) | **Supervisor:** Dr Muhammad Khalid

---

## Project Overview

This project develops and evaluates a **Gated Multimodal Deep Learning architecture** for robust county-level soybean crop yield prediction. The central research challenge is that real-world agricultural data is inherently incomplete — satellite imagery is obscured by cloud cover, weather sensors fail, and temporal misalignment between data sources is common. Rather than treating these as edge cases, this project models them explicitly.

The architecture fuses two independent data streams:

| Stream | Source | Model Branch |
|--------|--------|--------------|
| **Visual / Spectral** | Sentinel-2 satellite imagery (NDVI, RGB) | ResNet-18 + LSTM (3D-CNN encoder) |
| **Environmental** | WRF-HRRR daily weather reanalysis | Bidirectional LSTM (Bi-LSTM) |

A **Gated Multimodal Unit (GMU)** learns to dynamically re-weight the contribution of each stream per sample, gracefully degrading under simulated sensor failure rather than failing catastrophically.

---

## Research Questions

1. What is the quantifiable performance of unimodal architectures (3D-CNN satellite-only vs Bi-LSTM weather-only) and under what conditions do they break down?
2. Does a gated multimodal architecture significantly reduce performance degradation under simulated modality dropout?
3. Can an automated quality check allow accurate predictions when 50% of satellite images are unusable due to cloud cover?
4. Does a GMU reduce error variance (RMSE) compared to static fusion baselines under non-stationary noise?
5. Is deep feature-level fusion superior to raw data-level fusion for spatiotemporal agricultural prediction?
6. Can the dual-stream architecture autonomously learn to separate data quality from feature importance without explicit supervision?

---

## Dataset — CropNet Benchmark (Lin et al., 2024)

| Modality | Format | Spatial Resolution | Temporal Resolution |
|----------|--------|--------------------|---------------------|
| Sentinel-2 (AG / NDVI) | HDF5 (224×224 tiles) | ~10 m/pixel | 14-day revisit |
| WRF-HRRR meteorological | GRIB2 | ~3 km grid | Daily snapshot |
| USDA Soybean Yields | CSV | County-level | Annual |

**Coverage:** 15 US counties across 15 states, years 2016–2022 (full USDA); 2021–2022 (Sentinel-2 imagery).

---

## Repository Structure

```
multimodal_agriculture_prediction-notebook_only/
│
├── test.ipynb                              ← Main experiment notebook (all steps)
├── README.md                               ← This file
│
├── data/
│   └── CropNet/
│       ├── HRRR/
│       │   ├── realtime_wrf/               ← GRIB2 weather files (2016–2022)
│       │   ├── precip_wrf/                 ← Precipitation-specific GRIB2 files
│       │   └── hrrr/                       ← Subset HRRR files
│       ├── Sentinel/
│       │   └── data/
│       │       ├── AG/                     ← RGB satellite tiles (HDF5)
│       │       └── NDVI/                   ← NDVI satellite tiles (HDF5)
│       ├── USDA/
│       │   └── data/Soybean/               ← Annual county-level yield CSVs
│       └── labels/
│           ├── weather_train.csv           ← Exported Phase 1 train labels
│           ├── weather_test.csv            ← Exported Phase 1 test labels
│           ├── hrrr_scaler.pkl             ← Fitted StandardScaler artefact
│           └── hrrr_imputer.pkl            ← Fitted SimpleImputer artefact
│
├── multimodal_agriculture_prediction_complete.ipynb
├── Sentinel_2_Imagery.ipynb
├── USDA_Crop_Dataset.ipynb
└── WRF_HRRR_Computed_Dataset.ipynb
```

---

## Experimental Phases

The project proceeds in three sequential model-training phases:

| Phase | Description | Data Source | Status |
|-------|-------------|-------------|--------|
| **Phase 1** | Weather-only Bi-LSTM regressor | USDA + HRRR (2016–2022, 634 counties) | Active |
| **Phase 2** | Satellite encoder (ResNet-18 + LSTM) | Sentinel-2 + USDA label intersection | Pending data alignment |
| **Phase 3** | GMU fine-tuning (dual-stream gated fusion) | Phase 1 + Phase 2 encoders | Pending Phase 2 |

---

## Notebook Walkthrough — `test.ipynb`

---

### Step 0 — Environment & Reproducibility Setup

**Cells: 1–11** | *Run this step at the start of every session*

#### Step 0.0 — Data Download (Cell 1–2)
**What it does:** Downloads the CropNet dataset programmatically using the `cropnet` library. Downloads USDA Soybean CSV files, Sentinel-2 HDF5 imagery (AG and NDVI), and WRF-HRRR GRIB2 weather files for 15 county FIPS codes across years 2021–2022.

**Why:** Raw data acquisition must be reproducible and auditable. Using the official CropNet downloader ensures correct directory structure and file naming conventions that all downstream steps depend upon.

**Contribution to objective:** Establishes the complete multi-modal data corpus. The three modalities (satellite, weather, yield labels) are the direct inputs to the GMU architecture.

---

#### Step 0.1 — Package Verification (Cell 4)
**What it does:** Checks whether all required Python packages (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `sklearn`, `plotly`, `h5py`, `torch`, `torchvision`, `shap`) are installed. Installs any missing ones automatically. Sets `matplotlib` to the non-interactive `Agg` backend to prevent GUI crashes in headless/Windows environments.

**Why:** Ensures a fully reproducible environment regardless of where the notebook is executed (local GPU workstation, university HPC, or Google Colab). The `Agg` backend prevents `NSInternalInconsistencyException` and similar OS-level renderer errors.

**Gain:** Eliminates "works on my machine" failures and provides a clean dependency manifest for the dissertation's reproducibility appendix.

---

#### Step 0.1b — Matplotlib Smoke Test (Cell 5)
**What it does:** Renders a trivial 3-point line chart and confirms it saves to an in-memory PNG buffer correctly.

**Why:** Validates the `Agg` backend configuration end-to-end before any real visualisations are generated. A failing smoke test surfaces configuration errors immediately rather than mid-analysis.

**Gain:** Confidence that all 20+ subsequent plots will render without exception.

---

#### Step 0.2 — Global Random Seed (Cell 6)
**What it does:** Sets `SEED = 42` across `random`, `numpy`, and `torch` (including all CUDA streams). Declares the compute device (`cuda` or `cpu`).

**Why:** Scientific reproducibility is a first-class requirement. Any stochastic operation — dataset splitting, weight initialisation, dropout, data augmentation — must produce identical results across runs. Fixing the seed at the start of the session achieves this without restricting any algorithm's expressiveness.

**Gain:** Every result in the dissertation is reproducible to bit-level precision. Eliminates variance from run-to-run when reporting RMSE, MAE, and R².

---

#### Step 0.3 — Path Constants (Cell 6, continued)
**What it does:** Declares `pathlib.Path` constants for every data root: `CROPNET`, `USDA_ROOT`, `SENTINEL_ROOT`, `HRRR_ROOT`. Verifies each path exists on disk and prints an `OK` / `MISSING` status.

**Why:** A single source of truth for every file path prevents brittle string literals scattered across 20+ cells. The existence check surfaces data-download failures early, before the notebook reaches processing steps that would silently produce empty DataFrames.

**Gain:** Makes the notebook portable — changing `BASE` is sufficient to relocate the entire project. The status report provides an unambiguous system health check.

---

#### Step 0.4 — Build `usda_index` (Cell 7)
**What it does:** Iterates over all 7 annual USDA Soybean CSV files (2016–2022), loads each, drops rows with missing yield values, and constructs a 5-digit FIPS code from `state_ansi` + `county_ansi`. Concatenates all years into a single `usda_index` DataFrame (~3,430 county-year records, 634 counties). Prints a detailed coverage summary: records per year, counties with all seven years of data, states covered.

**Why:** This is the **primary regression target** table. Every model phase ultimately predicts `yield_bu_acre` (or its log-transform). Centralising all 7 years into one index enables stratified train/test splitting across the full temporal range, which is critical for measuring time-generalisation — a key evaluation criterion in the dissertation's stress-testing design.

**Key engineering detail:** Explicit `float→int→str` FIPS conversion prevents malformed codes like `"5.01.0"` that occur when pandas infers float64 for integer columns containing nulls.

**Contribution to objective:** Defines the ground-truth output space for the weather-only Bi-LSTM (Phase 1). Training on 7 years of multi-state soybean data gives the weather branch sufficient temporal diversity to learn non-trivial yield-climate relationships before fine-tuning begins.

---

#### Step 0.5 — Build `sentinel_index` (Cell 8)
**What it does:** Recursively scans all Sentinel-2 HDF5 files under `SENTINEL_ROOT/AG/`. For each file, opens it with `h5py`, reads FIPS keys, date keys, and tile array dimensions. Assembles a `sentinel_index` DataFrame recording: FIPS, year, state, quarter date range, file path, number of dates, and number of tiles.

**Why:** Sentinel-2 imagery is delivered as HDF5 archives with a nested `fips → date → data` hierarchy. Building a read-once index avoids reopening dozens of large binary files repeatedly during EDA and training. The index also serves as the definitive record of what satellite data is actually available on disk — essential for diagnosing the Phase 2/3 data alignment gap.

**Gain:** O(1) lookup for "does county X have satellite data for year Y?" throughout all downstream steps.

---

#### Step 0.4b — USDA–Sentinel Intersection (Cell 9)
**What it does:** Cross-references `sentinel_index` FIPS codes against `usda_index`. Reports how many USDA yield records have matching satellite imagery. Explains the root cause if the intersection is empty: the Sentinel imagery was downloaded for county ANSI 001 (e.g., `01001`, `05001`) but the USDA soybean dataset predominantly covers counties with higher ANSI codes that have significant soybean farming.

**Why:** Identifying the data alignment gap is critical before any multimodal model can be built. Running Phase 2 (satellite encoder) or Phase 3 (GMU) without labelled satellite samples would be methodologically invalid. This cell documents the gap explicitly and prints corrective instructions rather than silently producing empty training sets.

**Contribution to objective:** Directly addresses Research Question 3 — the ability to maintain predictions under missing modality scenarios. The gap itself motivates the GMU's graceful degradation design.

---

#### Step 0.6 — Train/Test Splits (Cell 10)
**What it does:**
- **Phase 1 (Weather branch):** Stratified 80/20 split of `usda_index` using 5-bin yield quintile stratification. Produces `weather_train` (2,744 records) and `weather_test` (686 records).
- **Phase 2/3 (Satellite/GMU branch):** Same stratification (2-bin median split) applied to `sentinel_labels`. Creates empty placeholder DataFrames (`multimodal_index`, `train_set`, `test_set`) when the intersection is empty, preventing `NameError` in downstream cells.

**Why:** Yield values follow a near-normal distribution spanning ~20–80 BU/ACRE. A naive random split could leave the test set skewed toward high or low yields, producing artificially good or bad RMSE. Quintile stratification ensures both halves see the same yield distribution, making RMSE, MAE, and R² metrics directly comparable.

**Gain:** Unbiased performance estimation. Using the same `SEED=42` and split method across all model phases ensures fair comparison between unimodal baselines and the GMU.

---

### Step 1 — USDA Soybean Exploratory Data Analysis (Target Variable)

**Cells: 12–18** | *Characterises the regression target before any model is built*

---

#### Step 1.1–1.3 — County Inventory, Quality Check, FIPS Resolution (Cell 13)
**What it does:** Builds a wide pivot table (FIPS × year) showing yield values for every county across all 7 years. Flags counties missing data for any year. Prints the three-way intersection report: USDA counties, Sentinel counties, multimodal overlap.

**Why:** Missing labels are a silent source of training bias. A county that only appeared in 2016 and 2022 introduces a confound if the model experiences those years differently. The pivot matrix makes missingness immediately visible.

**Gain:** A full audit trail for which county-years are included in training and which are excluded — essential for the dissertation's transparency and reproducibility claims.

---

#### Step 1.4 — Target Distribution Analysis (Cell 14)

**What it does:** Produces four diagnostic plots:
1. **Histogram of raw yield** — visualises the overall distribution shape.
2. **Histogram of log(yield)** — tests whether log-transformation achieves normality.
3. **Boxplot by year** — reveals inter-annual variability and outliers at a glance.
4. **Q-Q plot** — directly tests normality of the raw yield distribution.

Runs a **Shapiro-Wilk normality test** on both raw and log-transformed yield. Sets the global flag `USE_LOG` based on the result, which determines whether the model target is `yield_bu_acre` or `log(yield_bu_acre)` throughout the rest of the notebook.

**Why:** Linear regression and many loss functions assume a reasonably symmetric target. Agricultural yield data commonly has a positive skew due to rare bumper-harvest years. If Shapiro-Wilk rejects normality (p ≤ 0.05), log-transforming the target reduces heteroscedasticity and prevents the model from being disproportionately penalised on high-yield outliers.

**Contribution to objective:** Directly affects model quality. An incorrectly specified target scale can inflate RMSE by 10–30% and distort the Relative Degradation Index (RDI) comparisons between the GMU and baselines.

---

#### Step 1.5 — Year-over-Year Yield Change Analysis (Cell 15)
**What it does:** Computes absolute and relative yield change for every county between consecutive years (2016→2017, 2017→2018, …, 2021→2022). Produces:
1. Distribution histogram of all year-over-year (YoY) deltas.
2. Bar chart of mean delta by year-pair, colour-coded green (improvement) / red (decline).
3. Horizontal bar chart of mean YoY delta per state.

Identifies the top-5 most improving and top-5 most declining counties by average annual delta.

**Why:** Temporal drift is a fundamental challenge in agricultural ML. If a model is trained on 2016–2020 and tested on 2021–2022, it must generalise across years that may have systematically different yield levels (due to climate trends, seed genetics, or farming practice changes). Understanding the YoY structure allows us to design a temporally-aware train/test split and to interpret degradation curves correctly during stress-testing.

**Gain:** Provides a baseline understanding of "natural" yield variance. This is used to contextualise model RMSE — an RMSE lower than the average YoY standard deviation indicates the model is learning genuine signal rather than just memorising means.

---

#### Step 1.6 — Geographic Coverage Visualisation (Cell 16)
**What it does:** Attempts to render an interactive **Plotly choropleth map** of county mean soybean yields (2016–2022) with orange markers showing the 15 Sentinel-2 coverage counties. Falls back to a **static two-panel matplotlib chart** (mean yield by state + stacked county record counts by year) when the required GeoJSON file is unavailable offline.

**Why:** Geography is a latent confound in yield prediction. The Corn Belt (Illinois, Iowa, Indiana) consistently outperforms the Deep South (Alabama, Louisiana) by 20–30 BU/ACRE. A model that learns county centroids without understanding geographic context may simply memorise regional baselines rather than capturing weather–satellite interactions. The coverage overlay also reveals whether the 15 Sentinel counties are geographically representative or clustered in one region.

**Contribution to objective:** Supports Research Question 6 — geographic holdout validation. Understanding which states are in the training set vs the test set is required for reporting geographic generalisation performance.

---

#### Step 1.7 — Freeze and Export Label Tables (Cell 17)
**What it does:** Applies the `USE_LOG` decision from Step 1.4 to add the model target column (`log_yield_bu_acre` or `yield_bu_acre`) to the train and test DataFrames. Saves three CSV files to `data/CropNet/labels/`:
- `weather_train.csv` — Phase 1 training labels
- `weather_test.csv` — Phase 1 test labels
- `multimodal_index.csv` — Phase 2/3 labels (if intersection non-empty)

**Why:** Separating data preparation from model training is a best practice that prevents accidental data leakage. Once labels are persisted, Step 3 (Bi-LSTM training) and all subsequent steps load them fresh from disk, guaranteeing they see only the data that was prepared before any model was built.

**Gain:** A permanent, auditable snapshot of the exact training targets used for every model in the dissertation. If USDA data is updated or re-downloaded, the saved CSVs still reproduce the original experiment.

---

### Step 2 — HRRR Weather Feature Engineering

**Cells: 18–23** | *Transforms raw GRIB2 meteorological files into a normalised PyTorch Dataset*

---

#### Step 2.1 — GRIB2 File Audit (Cell 19)
**What it does:** Scans every `realtime_wrf` GRIB2 file across all 7 years using `cfgrib`. For each file reports: file size, grid shape, latitude/longitude range, total variable count, which agronomically relevant variables are present (`t2m`, `d2m`, `r2`, `u10`, `v10`, `tp`, `sp`, `tcc`, `mstav`), and which are missing. Computes the intersection of variables present in **all years** (stored as `HRRR_EXTRACT_VARS`).

**Why:** HRRR file formats are not guaranteed to be consistent across years. A variable present in 2019 files may be absent from 2016 files due to NOAA model version changes. Building a model on 9 features and then discovering 2 of them are NaN for 2016–2017 would silently degrade Phase 1 training quality. This audit makes the available feature space explicit before any extraction begins.

**Gain:** `HRRR_EXTRACT_VARS` is the single authoritative source for which weather variables will be used in all downstream steps, preventing silent feature mismatch between training years.

---

#### Step 2.2 — County-Level HRRR Value Extraction (Cell 20)
**What it does:** For each unique (FIPS, state) combination in `usda_index`, looks up the county's geographic centroid from an embedded `STATE_CENTROIDS` dictionary. Uses a **nearest grid point** algorithm (minimising squared Euclidean distance in lat/lon space) to identify the single HRRR grid cell closest to the county centroid. Extracts all `HRRR_EXTRACT_VARS` values from that cell for each year's GRIB2 file. Converts HRRR longitudes from 0–360° to −180–180° before matching. Converts temperatures from Kelvin to Celsius (`t2m_c`, `d2m_c`).

**Why:** HRRR data is a spatial grid covering the continental US at ~3 km resolution. County centroid matching is the standard spatial join technique used when county-level analysis does not require area-weighted averaging (which would require polygon intersection and significantly more complexity). For soybean counties (~thousands of km²), the nearest-grid-point approximation introduces < 5 km error — well within acceptable bounds for seasonal climate signal extraction.

**Contribution to objective:** This is the core data pipeline for the **weather-only Bi-LSTM branch** (Phase 1). Every county-year in `usda_index` must have a weather feature vector for Phase 1 training to proceed.

---

#### Step 2.3 — Feature Engineering and Merge (Cell 21)
**What it does:** Derives three physically meaningful composite features from the raw HRRR variables:
- **`wind_speed`** = √(u10² + v10²) — scalar wind speed from U and V components (m/s)
- **`sp_hpa`** = sp / 100 — surface pressure converted from Pa to hPa (standard meteorological units)
- **`vpd_hpa`** (Vapour Pressure Deficit) = computed using the **Tetens formula**: es(T) − es(Td), where es is the saturation vapour pressure at temperature T and dew point Td.

Merges the resulting `hrrr_features` table with `usda_index` on `(fips, year)` to produce `weather_features` — the single merged table used for all Phase 1 modelling.

**Why VPD matters:** VPD is one of the most powerful predictors of crop stress in agronomy. When VPD is high (hot, dry air), plants close their stomata to conserve water, directly reducing photosynthesis and final grain fill. Neither raw temperature nor raw relative humidity alone captures this effect — it requires their non-linear combination. By computing VPD explicitly we give the Bi-LSTM access to an agronomically grounded feature that pure deep learning cannot derive from raw GRIB2 values without additional domain context.

**Wind speed** is relevant because high wind accelerates soil moisture evaporation and can physically damage crop canopy. The U and V components separately carry directional information that is less useful than scalar speed for yield prediction.

**Contribution to objective:** Feature engineering with domain knowledge is a deliberate methodological choice aligned with the dissertation's Analytical Process — "identify growth trends in images and stress factors (like drought) in the weather data." VPD directly quantifies drought stress intensity.

---

#### Step 2.4 — Z-Score Normalisation (Cell 22)
**What it does:** Trains a `SimpleImputer` (strategy=`mean`) on the **training split only** to fill NaN values (e.g., `r2` missing for year 2016). Then trains a `StandardScaler` (μ=0, σ=1) on the training split only and applies it to all rows. Adds the model target column (`target` = log(yield) or yield). Persists both artefacts as `hrrr_scaler.pkl` and `hrrr_imputer.pkl` for inference-time use.

**Why this ordering matters (avoiding data leakage):** The scaler and imputer are fit exclusively on training data. If they were fit on all data including the test set, the model would have indirect access to test set statistics during training — a form of data leakage that artificially inflates test performance. Fitting on train only guarantees a strict information boundary.

**Why StandardScaler specifically:** The Bi-LSTM uses gradient-based optimisation (Adam/SGD). Features on vastly different scales (e.g., surface pressure ~100,000 Pa vs relative humidity 0–100%) cause gradients to be dominated by the large-magnitude features, slowing convergence and potentially preventing convergence to a useful minimum. Z-scoring brings all features to comparable scale without distorting their relative distributions.

**Gain:** Reproducible, leak-free preprocessing. The saved `.pkl` files mean that at inference time (or during stress-testing) any new sample can be transformed identically to training data with a single `scaler.transform()` call.

---

#### Step 2.5 — `HRRRWeatherDataset` PyTorch Dataset (Cell 23)
**What it does:** Implements a `torch.utils.data.Dataset` subclass that:
1. Filters out any rows with NaN in target or features.
2. Returns each item as `(x, y, meta)` where `x` is a `FloatTensor` of shape `(seq_len, n_features)`, `y` is a scalar yield target, and `meta` is a dict with `{fips, year}` for traceability.
3. Wraps it in `DataLoader` objects with `batch_size=64` for both train and test.
4. Runs a **smoke test** on the first batch to confirm tensor shapes, dtypes, and value ranges.

The `seq_len=1` design is intentional: the current implementation uses one annual HRRR snapshot per county-year. The Dataset is designed to scale transparently to `seq_len=200` (daily data across a full growing season) by replacing the single-snapshot GRIB2 file with a daily time-series loader — a planned extension for the full dissertation.

**Why a `Dataset` class:** PyTorch's `DataLoader` protocol provides automatic batching, shuffling, parallel data loading, and integration with the training loop. Encapsulating all preprocessing inside the Dataset means the training loop (Step 3) sees only clean tensors and never needs to call pandas or numpy — keeping concerns separated and making the model code architecture-framework agnostic.

**Contribution to objective:** This Dataset is the direct input interface for the **Phase 1 Bi-LSTM weather regressor** (Step 3, coming). Getting this right is foundational: the same Dataset pattern will be reused for the satellite branch (Phase 2) and the GMU fusion model (Phase 3), with only the feature extraction changed.

---

## Evaluation Framework

| Metric | Symbol | Purpose |
|--------|--------|---------|
| Root Mean Squared Error | RMSE | Primary accuracy measure; penalises large errors |
| Mean Absolute Error | MAE | Robust accuracy; less sensitive to outliers |
| Coefficient of Determination | R² | Proportion of yield variance explained by the model |
| Mean Absolute Percentage Error | MAPE | Scale-free relative error |
| Relative Degradation Index | RDI | Performance drop as modality corruption increases |
| Robustness Retention Score | RRS | Fraction of clean accuracy retained under corruption |

Statistical validation: paired t-tests, 5-fold cross-validation, geographic holdout.

---

## Robustness Stress Testing (Planned — Steps 4–6)

Four degradation scenarios will be applied at four intensity levels (10%, 25%, 50%, 75%):

| Scenario | Description |
|----------|-------------|
| **Image dropout** | Random cloud masking on satellite tiles |
| **Weather dropout** | Random feature masking on HRRR variables |
| **Structured dropout** | Seasonal (growing-season) data removal |
| **Combined dropout** | Simultaneous multi-modality corruption |

The GMU's gate weights will be monitored throughout to verify that the model shifts reliance from the degraded modality to the intact one — reproducing the expert-agronomist heuristic described in the project rationale.

---

## Model Architecture Roadmap

### Phase 1 — Bi-LSTM Weather Regressor (Step 3, upcoming)
- Input: `(batch, seq_len, n_features)` weather tensor from `HRRRWeatherDataset`
- Architecture: 2-layer Bidirectional LSTM → Linear head
- Output: Scalar yield prediction
- Purpose: Unimodal baseline + pre-trained weather encoder for GMU

### Phase 2 — Satellite Encoder (Step 4, upcoming)
- Input: Sentinel-2 HDF5 tiles `(batch, T, C, H, W)`
- Architecture: ResNet-18 backbone → Temporal LSTM
- Output: County-level yield prediction from imagery alone
- Purpose: Unimodal baseline + pre-trained visual encoder for GMU

### Phase 3 — Gated Multimodal Unit (Step 5, upcoming)
- Input: Both encoders' intermediate feature representations
- Architecture: GMU gate = σ(W_z · [h_weather; h_satellite]) — dynamic per-sample weighting
- Output: Fused yield prediction more robust than either unimodal branch
- Purpose: Primary novel contribution of the dissertation

### Baselines (Steps 4–5)
- Early Fusion: Concatenate raw features before any encoder
- Late Fusion: Independent encoders → MLP fusion (no gate)

---

## Current Data Status

| Component | Status | Notes |
|-----------|--------|-------|
| USDA Soybean Labels (2016–2022) | ✅ Complete | 3,430 county-year records, 634 counties |
| HRRR Weather Features | ✅ Extracted | 7 variables × 7 years × all USDA counties |
| Sentinel-2 Imagery | ✅ Downloaded | 2021–2022, 15 counties (AG + NDVI) |
| USDA–Sentinel FIPS Intersection | ⚠ Gap identified | County ANSI 001 not in USDA soybean data |
| Phase 1 Labels (CSV) | ✅ Exported | `weather_train.csv`, `weather_test.csv` |
| Preprocessing Artefacts | ✅ Saved | `hrrr_scaler.pkl`, `hrrr_imputer.pkl` |
| Bi-LSTM Training | 🔲 Pending | Step 3 |
| Satellite Encoder | 🔲 Pending | Requires USDA–Sentinel alignment |
| GMU Architecture | 🔲 Pending | Requires Phase 1 + Phase 2 |
| Stress Testing | 🔲 Pending | Requires GMU |

---

## How to Reproduce

### Prerequisites
- Python 3.9+ with Anaconda or venv
- GPU strongly recommended (CUDA 11.8+) for Steps 3–6
- ~15 GB disk space for full CropNet download

### Setup

```bash
conda create -n cropnet python=3.9
conda activate cropnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn scipy scikit-learn plotly h5py cfgrib shap
pip install cropnet   # CropNet official downloader
```

### Execution Order

Open `test.ipynb` in VS Code or Jupyter and run all cells **sequentially top-to-bottom**. Do not skip cells — each step stores variables consumed by later steps.

1. Set your Sentinel Hub credentials in Cell 1 before downloading data.
2. Cell 4 automatically installs any missing packages.
3. After a fresh download, all subsequent steps are fully automatic.

---

## Key References

1. Lin, F., et al. (2024). *An Open and Large-Scale Dataset for Multi-Modal Climate Change-aware Crop Yield Predictions.* [arXiv:2406.06081](https://arxiv.org/pdf/2406.06081)
2. Han, Z., et al. (2022). *Multimodal Dynamics: Dynamical Fusion for Trustworthy Multimodal Classification.* CVPR 2022.
3. Maimaitijiang, M., et al. (2020). *Soybean yield prediction from UAV using multimodal data fusion and deep learning.* RSE. [DOI](https://doi.org/10.1016/j.rse.2019.111599)
4. Johnson, M. D. & Hsieh, W. W. (2016). *Crop yield forecasting on the Canadian Prairies by remotely sensed vegetation indices and machine learning methods.* [DOI](https://doi.org/10.1016/j.agrformet.2015.11.003)
5. Schlenker, W. & Roberts, M. J. (2009). *Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change.* PNAS. [DOI](https://www.pnas.org/doi/epdf/10.1073/pnas.0906865106)

---

*MSc AI Dissertation — University of Hull — 2026*
