# Project Plan: Sustainable Crop Yield Prediction via Multi-modal AI

**MSc Dissertation — University of Hull**
**Last updated: April 12, 2026**

---

## Status Overview

| Step | Title | Status |
|------|-------|--------|
| 0 | Environment & Reproducibility Setup | ✅ Complete |
| 1 | USDA Soybean EDA | ✅ Complete |
| 2 | HRRR Weather Feature Engineering | ✅ Complete |
| **3** | **Ethics, Bias Audit & Fairness Specification** | **→ Next** |
| 4 | Phase 1 Model: Bi-LSTM Weather Encoder | 4.1 ✅, 4.2–4.6 pending |
| 5 | Sentinel-2 Imagery EDA | Pending |
| 6 | Phase 2: Satellite Encoder (ResNet18 + LSTM) | ⚠️ Blocked |
| 7 | Phase 3: GMU Fine-tuning | ⚠️ Blocked |
| 8 | Robustness to Modality Dropout | Pending |
| 9 | Explainability (SHAP) | Pending |
| 10 | Final Evaluation & Dissertation Reporting | Pending |

---

## Step 0 — Environment & Reproducibility Setup ✅ Complete

Sets global seeds, verifies packages, defines all path constants, matplotlib Agg backend.

- **0.1** Package verification + `matplotlib.use('Agg')` + core imports
- **0.2** Global random seeds (SEED=42, NumPy, Python, PyTorch)
- **0.3** Path constants (`USDA_ROOT`, `SENTINEL_ROOT`, `HRRR_ROOT`)
- **0.4** Load all USDA Soybean CSVs 2016–2022 → `usda_index` (3,430 records, 634 FIPS)
- **0.4b** Programmatic Sentinel-USDA FIPS intersection → `sentinel_labels` (empty — data gap)
- **0.5** Scan all Sentinel HDF5 files → `sentinel_index` (15 FIPS, 3,884 tiles)
- **0.6** Define 80/20 stratified splits — Phase 1: 2,744 train / 686 test; Phase 2&3: skipped

---

## Step 1 — USDA Soybean EDA ✅ Complete

- **1.1–1.3** County inventory, quality check, FIPS resolution, intersection report
- **1.4** Yield distribution: histogram, boxplot by year, Q-Q plot, Shapiro-Wilk → `USE_LOG=True`
- **1.5** Year-over-year yield change per county (2,467 pairs; 2019→20 best, 2018→19 worst)
- **1.6** Interactive Plotly choropleth — 634 USDA counties, 15 Sentinel markers overlaid
- **1.7** Export `weather_train.csv` (2,744 rows) and `weather_test.csv` (686 rows) with `log_yield_bu_acre` target

---

## Step 2 — HRRR Weather Feature Engineering ✅ Complete

- **2.1** GRIB2 audit — 9 agronomic variables; `r2` missing in 2016 only; all 7 years present
- **2.2** Nearest-grid-point county extraction via state centroid proxy → `hrrr_raw` (4,438 rows, 634 FIPS)
- **2.3** Derived features: `wind_speed`, `sp_hpa`, `vpd_hpa` + USDA merge → `weather_features` (3,430 rows, all matched)
- **2.4** `SimpleImputer` (for 2016 `r2` NaN) + `StandardScaler` — fit on train only; artefacts saved to `data/CropNet/labels/`
- **2.5** `HRRRWeatherDataset` — `(64, 1, 9)` batches; `ds_train=2744`, `ds_test=686`; target y ∈ [3.207, 4.358] (log-scale)

**Feature columns:** `t2m_c`, `d2m_c`, `r2`, `wind_speed`, `tp`, `sp_hpa`, `tcc`, `mstav`, `vpd_hpa`

---

## Step 3 — Ethics, Bias Audit & Fairness Specification ← NEXT TO IMPLEMENT

> This step must be completed and documented before any model results are reported or used in the dissertation. It defines the fairness contract the model must satisfy.

### 3.1 Data Provenance & Representativeness Audit

| Dimension | Finding | Risk Level |
|-----------|---------|-----------|
| Geographic coverage | 10 of ~35 soybean-producing US states; Corn Belt heavily over-represented | High |
| County coverage | 634 of ~1,400 US soybean counties (~45%) | High |
| Temporal scope | 2016–2022 only; 2019 confounded by trade war + Midwest flooding | Medium |
| Weather spatial resolution | State-level centroid proxy — within-state heterogeneity erased | High |
| Modality gap | Sentinel-2 on county ANSI 001; USDA covers ANSI 003+; zero intersection | High |
| Surveyor bias | USDA NASS over-represents large commercial farms; subsistence/organic excluded | Medium |

### 3.2 Bias Metrics (computed at Steps 4.3, 6.4, and 7.4)

**Sliced RMSE** — must be reported per model, not just aggregate:

$$\text{RMSE}_g = \sqrt{\frac{1}{|g|}\sum_{i \in g}(\hat{y}_i - y_i)^2}$$

Mandatory slices: state (10), year (7), yield quintile (Q1–Q5), proximity to state centroid (near/far)

**Mean prediction bias per quintile:**

$$\text{Bias}_q = \frac{1}{|q|}\sum_{i \in q}(\hat{y}_i - y_i)$$

If Bias(Q1) > 0 and Bias(Q5) < 0, the model is regressing to the mean — underestimating crop failure and overestimating bumper yields (systematic fairness failure).

**Coefficient of variation of errors per state:**

$$\text{CV-Error}_s = \frac{\sigma(\hat{y}_s - y_s)}{\mu(y_s)}$$

States with CV-Error > 0.20 are flagged as poorly served by the model.

**Geographic coverage gap:**

$$\text{Coverage Gap} = 1 - \frac{|\text{FIPS in model}|}{|\text{FIPS in USDA full dataset}|}$$

Currently ~55%. Any deployment must document which counties are excluded and why.

**Jensen's inequality correction** (required when `USE_LOG=True`):

$$\hat{y}_{\text{BU/ACRE}} = \exp\!\left(\hat{y}_{\log} + \frac{\hat{\sigma}^2}{2}\right)$$

where $\hat{\sigma}^2$ is the residual variance on the train split. Without this, all back-transformed predictions are systematically underestimates.

### 3.3 Identified Biases, Mitigations & Tradeoffs

| # | Bias | Root Cause | Severity | Mitigation | Tradeoff |
|---|------|-----------|---------|-----------|---------|
| B1 | State-centroid weather proxy | All counties in a state share identical HRRR features | High | Replace with per-county TIGER centroid lat/lon | +compute; requires coordinate table |
| B2 | Corn Belt geographic dominance | IL, IA, IN, MN, NE account for >60% of records | High | Inverse-frequency state loss weighting: $w_s = N / (K \cdot N_s)$ | May increase aggregate RMSE |
| B3 | Low-yield observation scarcity | Q1 counties have fewer multi-year records | Medium | Oversample Q1 train records; stratify by quintile | Risk of overfitting on rare events |
| B4 | Log-target Jensen bias | $\mathbb{E}[\exp(\hat{y})] \neq \exp(\mathbb{E}[\hat{y}])$ | Medium | Apply correction $\exp(\hat{y} + \sigma^2/2)$ at inference | Requires storing residual variance |
| B5 | HRRR single-snapshot proxy | Jan 1 init file ≠ growing-season weather | High | Flag as explicit limitation; future work uses daily files | Acknowledged limitation; not fixable with current data |
| B6 | USDA reporting threshold omission | Counties below minimum acreage absent from data | Medium | Flag FIPS with < 3 years; exclude from policy claims | Reduces stable dataset to ~478 counties |
| B7 | Random 80/20 temporal mixing | Test set contains all years; no held-out future year | Low–Medium | Add leave-one-year-out (LOYO) evaluation for 2022 as supplement | Smaller effective test size |
| B8 | Sentinel-2 AZ 2022 tile anomaly | AZ 2022: 1,524 tiles vs ~60 average — likely archive artefact | Medium | Exclude AZ 2022 from Phase 2/3 training | Removes one county-year from already small satellite dataset |
| B9 | USDA surveyor bias | Large commercial operations over-represented in NASS survey | Medium | Document; model must not be used for smallholder or OFR-exempt farms | Cannot be corrected without external survey data |

### 3.4 Key Tradeoffs

**Accuracy ↔ Fairness**
Inverse-frequency state weighting (B2) redistributes learning capacity from dominant corn-belt states to minority states. Expected outcome: +0.5–2.0 BU/ACRE RMSE increase on IL/IA, matched by −3–5 BU/ACRE RMSE reduction on AL/DE. Both weighted and unweighted metrics must be reported.

**Model Complexity ↔ Interpretability**
The GMU (Step 7) is the highest-accuracy architecture but least interpretable. The Ridge Regression baseline (Step 4.4) is fully interpretable and should be presented as the recommended option for any policy or advisory context. SHAP values for the Bi-LSTM (Step 9) are approximate — KernelExplainer, not DeepExplainer.

**Data Coverage ↔ Data Quality**
Including all 7 years maximises the training signal but injects a confounded 2019 observation (trade war + Midwest flooding). Recommendation: train on all years but evaluate 2019 and 2022 in isolation in the LOYO supplement.

**Log Transform ↔ Direct Interpretability**
`USE_LOG=True` (statistically justified by Shapiro-Wilk) benefits optimisation but makes direct interpretation harder. Always report final metrics back-transformed to BU/ACRE with Jensen correction applied.

**Modality Richness ↔ Deployability**
The GMU model requires both weather and satellite inputs. The Sentinel-USDA FIPS gap means Phase 2/3 is currently untrainable. The Phase 1 weather-only model is the only currently deployable system. This is a data infrastructure finding — not a model deficiency — and must be framed as such in the dissertation.

### 3.5 Ethical Use Constraints

- **No individual-farm inference.** The model predicts county-aggregate yield only. Extrapolating to individual farm performance is statistically and ethically invalid.
- **Food security embargo.** Predictions must not be released before the corresponding official USDA NASS report to prevent commodity market speculation.
- **Geographic equity disclosure.** Any deployment must list which US counties are excluded and acknowledge that predictions for under-represented states carry materially higher uncertainty.
- **Climate nonstationarity.** The model trained on 2016–2022 must not be assumed valid beyond 2025 without recalibration. Yield distributions shift under progressive climate change.

---

## Step 4 — Phase 1 Model: Bi-LSTM Weather Encoder

| Sub-step | Description | Status |
|----------|-------------|--------|
| 4.1 | `BiLSTMRegressor`: 2-layer BiLSTM, hidden=64, 128-dim encoding, 146,049 params | ✅ Done |
| 4.2 | Training loop — 80 epochs, Adam lr=1e-3, ReduceLROnPlateau, MSE on log-yield | Pending |
| 4.3 | Evaluate: aggregate + **sliced RMSE/MAE/R²** by state, year, quintile; Jensen correction applied | Pending |
| 4.4 | Baseline comparison: Ridge Regression vs Gradient Boosting vs Bi-LSTM | Pending |
| 4.5 | **Bias audit report**: CV-Error by state; mean bias by quintile; flag B1/B2/B3 | Pending |
| 4.6 | Save encoder weights (`bilstm_encoder.pt`) for Step 7 GMU warm-start | Pending |

---

## Step 5 — Sentinel-2 Imagery EDA

| Sub-step | Description |
|----------|-------------|
| 5.1 | Load HDF5 tiles; visualise RGB composites and NDVI maps per state |
| 5.2 | Per-band statistics: mean, std, min, max across all tiles and dates |
| 5.3 | Temporal coverage analysis — missing quarters, cloud-fraction proxy |
| 5.4 | Data gap summary: FIPS-USDA mismatch; AZ 2022 tile anomaly (B8 — 1,524 tiles vs ~60 avg); recommended exclusions |

---

## Step 6 — Phase 2: Satellite Encoder (ResNet18 + LSTM) ⚠️ Blocked

**Blocker:** Requires USDA data for county ANSI 001 (currently absent from downloaded CSVs).

When unblocked:
- `SentinelTileDataset` with Multiple Instance Learning (MIL) tile aggregation
- ResNet18 backbone (ImageNet pretrained) → temporal LSTM → 256-dim embedding
- Tile-level bias check: AZ 2022 excluded per B8
- Sliced evaluation metrics mirror Step 4.3

---

## Step 7 — Phase 3: GMU Fine-tuning ⚠️ Blocked

**Blocker:** Requires Phase 2 completion.

When unblocked:
- Gated Multimodal Unit (GMU) fusing 128-dim weather + 256-dim satellite embeddings
- Warm-start from Step 4 + Step 6 encoder weights
- Fine-tune on `multimodal_index` (currently empty — data gap documented in Step 3.1)
- Sliced metrics mirror Step 4.3

---

## Step 8 — Robustness to Modality Dropout

- **Weather dropout:** zero-out HRRR inputs → measure GMU degradation
- **Satellite dropout:** zero-out Sentinel inputs → measure GMU degradation
- Report degradation curves for GMU vs Phase 1 alone
- **Fairness under dropout:** report degradation separately by state and yield quintile — low-yield counties may be disproportionately harmed

---

## Step 9 — Explainability (SHAP)

- SHAP KernelExplainer applied to Phase 1 Bi-LSTM (DeepExplainer not used — approximation not reliable for LSTM)
- Dominant features expected: `mstav`, `vpd_hpa`, `t2m_c`
- **Geographic proxy check:** If `sp_hpa` (surface pressure) ranks highly, it is likely acting as a geographic state identifier rather than a causal weather signal → flag B1, remove feature, retrain
- Beeswarm plots and feature importance bar charts exported at 300 DPI

---

## Step 10 — Final Evaluation & Dissertation Reporting

- Predictions vs actuals scatter plots; residual histograms
- **Full fairness table:** RMSE/MAE/R² per state, per year, per yield quintile for every model variant
- Jensen inequality correction applied to all back-transformed metrics
- Ethics section in dissertation:
  - Geographic equity gap
  - USDA surveyor bias (B9)
  - Food security embargo recommendation
  - Climate nonstationarity caveat
- All figures exported at 300 DPI for dissertation submission

---

## Data Summary

| Dataset | Coverage | Records | Notes |
|---------|---------|---------|-------|
| USDA Soybean | 2016–2022, 10 states | 3,430 county-year | 634 unique FIPS; 245 counties with all 7 years |
| HRRR (realtime_wrf) | 2016–2022 | 7 GRIB2 files (80–143 MB each) | 1059×1799 grid; Jan 1 snapshot only |
| Sentinel-2 AG | 2021–2022, 15 states | 116 HDF5 files | 3,884 tiles; county ANSI 001 only |
| Sentinel-2 NDVI | 2021–2022, 15 states | — | Parallel to AG |
| Labels (exported) | 2016–2022 | 2,744 train / 686 test | `weather_train.csv`, `weather_test.csv` |

## Known Data Gaps

1. **Sentinel-USDA FIPS mismatch** — Sentinel counties use ANSI 001; USDA CSVs cover ANSI 003+. Zero intersection. Phases 2 & 3 blocked until USDA county 001 data is downloaded.
2. **HRRR `r2` missing in 2016** — Imputed with training-set mean via `SimpleImputer`.
3. **`tp` (total precipitation) = 0 for all records** — Jan 1 initialisation file captures weather state, not accumulated precipitation. A limitation of the single-snapshot approach.
4. **AZ 2022 tile anomaly** — 1,524 tiles vs ~60 average. Likely a data archive artefact (B8). To be excluded from Phase 2/3.

## Saved Artefacts

| File | Location | Description |
|------|----------|-------------|
| `weather_train.csv` | `data/CropNet/labels/` | 2,744 train records with `log_yield_bu_acre` |
| `weather_test.csv` | `data/CropNet/labels/` | 686 test records with `log_yield_bu_acre` |
| `hrrr_scaler.pkl` | `data/CropNet/labels/` | `StandardScaler` fit on train split |
| `hrrr_imputer.pkl` | `data/CropNet/labels/` | `SimpleImputer` fit on train split |
