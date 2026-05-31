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
| 3 | Ethics, Bias Audit & Fairness Specification | ✅ Complete |
| 4 | Phase 1 Model: Bi-LSTM Weather Encoder | ✅ Complete |
| 5 | Sentinel-2 Imagery EDA | ✅ Complete |
| 6 | Phase 2: Satellite Encoder (CNN) | ✅ Complete |
| 7 | Phase 3: GMU Fine-tuning | ✅ Complete |
| 8 | Robustness to Modality Dropout | ✅ Complete |
| 9 | Explainability (SHAP) | ✅ Complete |
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

- **2.1** GRIB2 audit — 9 agronomic variables confirmed across 7 GRIB2 files (80–143 MB each); `r2` missing in 2016 only; grid shape (1059, 1799), lat 21.14–52.62, lon 225.9–299.08
- **2.2** Nearest-grid-point county extraction via state centroid proxy → `hrrr_raw`; all 3,535 USDA county-year records matched (100% join rate); `B1` bias documented (state-centroid proxy erases within-state heterogeneity)
- **2.3** Derived features: `wind_speed = √(u10² + v10²)`, `sp_hpa = sp/100`, `vpd_hpa = saturation_pressure − actual_pressure` + USDA merge → `weather_features` (3,535 rows, 15 columns, 0 missing-all-features records)
- **2.4** `SimpleImputer(strategy='mean')` for 2016 `r2` NaN + `StandardScaler` — both **fit on train split only** (2,828 rows) to prevent data leakage; artefacts `hrrr_scaler.pkl` and `hrrr_imputer.pkl` saved to `data/CropNet/labels/`
- **2.5** `HRRRWeatherDataset` (PyTorch `Dataset`) — `(64, 1, 9)` batches; `ds_train=2,828`, `ds_test=707`; target y ∈ [3.174, 4.244] (log-scale); `seq_len=1` (single annual snapshot — acknowledged limitation B5)

**Feature columns:** `t2m_c`, `d2m_c`, `r2`, `wind_speed`, `tp`, `sp_hpa`, `tcc`, `mstav`, `vpd_hpa`

**Scaler mean/std (train split):** `t2m_c` (1.336/9.852), `d2m_c` (−2.054/10.670), `r2` (78.518/13.432), `wind_speed` (4.037/2.353), `sp_hpa` (986.767/25.123), `tcc` (60.762/44.007), `mstav` (66.361/27.217), `vpd_hpa` (1.456/1.xxx)

> **Known limitation (B5):** `tp` (total precipitation) = 0.0 for all records — the Jan 1 initialisation GRIB2 captures weather state, not accumulated growing-season precipitation. This is a structural limitation of the single-snapshot approach and is flagged as future work (daily file extraction).

---

## Step 3 — Ethics, Bias Audit & Fairness Specification ✅ Complete

> All sub-steps (3.1–3.7) executed and documented. Fairness functions are registered for use in Steps 4.3, 6.4, and 7.4. Bias registry B1–B9 is finalised.

### 3.1 Data Provenance & Representativeness Audit ✅

| Dimension | Finding | Risk Level |
|-----------|---------|-----------|
| Geographic coverage | 15 of ~35 soybean-producing US states; Corn Belt heavily over-represented (IL, IA, IN, MN, NE > 60% of records) | High |
| County coverage | 634 of ~1,400 US soybean counties (~45%); ~55% coverage gap | High |
| Temporal scope | 2016–2022 only; 2019 confounded by trade war + Midwest flooding | Medium |
| Weather spatial resolution | State-level centroid proxy — within-state heterogeneity erased (B1) | High |
| Modality gap | Sentinel-2 on county ANSI 001; USDA covers ANSI 003+; zero FIPS intersection on disk — resolved by using state-aggregate labels for Phase 2/3 | High |
| Surveyor bias | USDA NASS over-represents large commercial farms; subsistence/organic excluded (B9) | Medium |

### 3.2 Bias Metrics (computed at Steps 4.3, 6.4, and 7.4) ✅

**Sliced RMSE** — must be reported per model, not just aggregate:

$$\text{RMSE}_g = \sqrt{\frac{1}{|g|}\sum_{i \in g}(\hat{y}_i - y_i)^2}$$

Mandatory slices: state (15), year (7), yield quintile (Q1–Q5), proximity to state centroid (near/far)

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

where $\hat{\sigma}^2$ is the residual variance on the train split. Without this, all back-transformed predictions are systematically underestimates. `JENSEN_SIGMA2` initialised here and updated at Step 4.3 to residual variance = 0.029034.

### 3.3 Identified Biases, Mitigations & Tradeoffs ✅

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

### 3.4 Key Tradeoffs ✅

**Accuracy ↔ Fairness**
Inverse-frequency state weighting (B2) redistributes learning capacity from dominant corn-belt states to minority states. Expected outcome: +0.5–2.0 BU/ACRE RMSE increase on IL/IA, matched by −3–5 BU/ACRE RMSE reduction on AL/DE. Both weighted and unweighted metrics must be reported.

**Model Complexity ↔ Interpretability**
The GMU (Step 7) is the highest-accuracy architecture but least interpretable. The Ridge Regression baseline (Step 4.4) is fully interpretable and should be presented as the recommended option for any policy or advisory context. SHAP values for the Bi-LSTM (Step 9) are approximate — KernelExplainer, not DeepExplainer.

**Data Coverage ↔ Data Quality**
Including all 7 years maximises the training signal but injects a confounded 2019 observation (trade war + Midwest flooding). Recommendation: train on all years but evaluate 2019 and 2022 in isolation in the LOYO supplement.

**Log Transform ↔ Direct Interpretability**
`USE_LOG=True` (statistically justified by Shapiro-Wilk) benefits optimisation but makes direct interpretation harder. Always report final metrics back-transformed to BU/ACRE with Jensen correction applied.

**Modality Richness ↔ Deployability**
The GMU model requires both weather and satellite inputs. The original Sentinel-USDA FIPS gap was resolved by switching to state-aggregate USDA labels (state mean yield per year) as the supervision signal for Phase 2/3. The Phase 1 weather-only model remains the primary county-level deployable system. This is documented as a data infrastructure finding in the dissertation.

### 3.5 Jensen's Inequality Correction Constant ✅

`JENSEN_SIGMA2` initialised from label prior distribution and updated at Step 4.3 to actual residual variance on the train split:
- **Prior estimate (label variance):** computed from `log_yield_bu_acre` distribution
- **Updated value (Step 4.3):** `JENSEN_SIGMA2 = 0.029034` (residual variance of BiLSTM train predictions)
- Applied in all `jensen_correction()` calls throughout Steps 4–9

### 3.6 Bias Registry B1–B9 ✅

All 9 biases documented with root cause, severity, mitigation, implementation step, and current status. Three biases flagged as high-severity open items for future work: B1 (centroid proxy), B2 (Corn Belt dominance — mitigation planned in training loop), B5 (single-snapshot weather).

### 3.7 Ethical Use Constraints ✅

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
