# Ethical Report: Multimodal Agricultural Crop Yield Prediction System
### Step 3 — Ethics, Bias Audit & Fairness Analysis

**Prepared by:** Senior AI Researcher & Ethics Expert  
**Date:** April 19, 2026  
**Project:** Multimodal Agriculture Prediction (Soybean Yield, USA 2016–2022)  
**Pipeline Step Under Review:** Step 3 (Ethics, Bias Audit & Fairness Specification)  
**Frameworks Applied:** EU AI Act (2024), NIST AI RMF 1.0, IEEE P7003, ACM FAccT Principles

---

## Executive Summary

This report provides a critical ethics review of Step 3 of the multimodal soybean yield prediction pipeline. The system integrates three data modalities — USDA NASS survey yields, WRF-HRRR atmospheric reanalysis, and Sentinel-2 satellite imagery — to train a county-level regression model for soybean yield prediction across the contiguous United States (2016–2022).

Step 3 is to be commended as a **proactive and structured ethics integration** within a technical pipeline. The inclusion of a bias registry (B1–B9), fairness metric functions, a Jensen correction constant, and explicit use constraints reflect a high level of methodological transparency. However, this report identifies several areas where the analysis requires **strengthening, clarification, or additional consideration** before the system can be released or cited in policy-relevant contexts.

**Overall Ethics Risk Tier:** High (per the EU AI Act's classification of AI systems affecting food security and agricultural planning).

---

## 1. Scope and Context

### 1.1 System Description

The pipeline predicts county-level soybean yield (bushels per acre) using:

| Modality | Source | Spatial Resolution | Temporal Scope |
|---|---|---|---|
| Yield labels | USDA NASS Survey | County aggregate | 2016–2022 |
| Weather features | WRF-HRRR GRIB2 | ~3 km grid (state centroid proxy) | 2016–2022 |
| Satellite imagery | Sentinel-2 (ESA) | 10 m | 2021–2022 only |

The models under development span three phases: (1) a Bi-LSTM weather regressor, (2) a ResNet18+LSTM satellite encoder, and (3) a Gated Multimodal Unit (GMU) fusion model. At the time of this report, only Phase 1 is trainable due to a data gap documented in Step 0.5a.

### 1.2 Intended and Declared Use

The system is intended for academic research and dissertation purposes only. Step 3.7 defines an explicit embargo against deployment for individual-farm advisory, commodity market signalling, or food security reporting without further validation. The report endorses this boundary as both technically justified and ethically necessary.

### 1.3 Ethics Review Methodology

This review examines Steps 3.1–3.7 of the notebook sequentially, applying:
- **Data ethics** lenses (provenance, consent, representativeness)
- **Algorithmic fairness** criteria (group equity, calibration, auditing)
- **Societal harm analysis** (downstream risk to farmers, markets, policy)
- **Research integrity** standards (reproducibility, transparency)

---

## 2. Data Provenance and Representativeness (Step 3.1)

### 2.1 Findings

Step 3.1 conducts a structured provenance audit across six dimensions: geographic coverage, county coverage, temporal scope, weather spatial resolution, modality gap, and surveyor bias.

| Dimension | Identified Risk | Severity |
|---|---|---|
| Geographic coverage | 15 of ~35 soybean-producing states in dataset | High |
| County coverage | ~45% of US soybean counties; 55% absent | High |
| Temporal scope | 7 years; 2019 confounded | Medium |
| Weather resolution | State-centroid proxy erases intra-state heterogeneity | High |
| Modality gap | Zero USDA yield labels for Sentinel-2 FIPS codes | High |
| Surveyor bias | USDA NASS over-represents large commercial operations | Medium |

### 2.2 Critical Assessment

**Geographic underrepresentation** is the most consequential risk. A model trained on 45% of US soybean counties — with pronounced Corn Belt concentration — will systematically underfit the southern and western production zones. This is not merely a performance issue; it is a **distributional justice issue**: farmers in under-represented states who may one day use derivative tools built on this research are less well served by the resulting model.

**The USDA NASS surveyor bias** (B9) deserves more prominent treatment than it receives. The NASS survey design explicitly excludes farms below a minimum acreage threshold and relies on voluntary participation, creating a systematic exclusion of:
- Subsistence and smallholder operations
- Organic and alternative-practice farms
- Farms operated by marginalised communities, who are statistically over-represented in small-farm categories (USDA Census of Agriculture, 2017)

The notebook acknowledges this bias and correctly states that the model must not be used for smallholder farms. However, it does not quantify the magnitude of this exclusion or recommend any external data source to partially correct it. This gap should be addressed in the dissertation's limitations chapter.

**The 55% county coverage gap** introduced by the USDA reporting threshold is correctly identified but incompletely mitigated. The coverage\_gap() function provides a scalar metric, yet no spatial map is produced in Step 3 to visualise _which_ counties are systematically absent. A choropleth visualisation would make this gap tractable for reviewers and increase transparency.

### 2.3 Data Consent and Licensing

All three data sources are used under appropriate open licences:
- **USDA NASS**: Licensed for academic use; publicly available
- **WRF-HRRR**: NOAA public domain
- **Sentinel-2**: ESA / Copernicus Open Access Hub (CC-BY 4.0)
- **CropNet pipeline**: MIT licence (Wang et al., 2022)

**No personally identifiable information (PII) is present.** All data are aggregated at the county level, which is the unit of analysis. This satisfies GDPR/CCPA data minimisation principles in the academic context.

**Licensing adequacy:** The CC-BY 4.0 licence requires attribution. The notebook references Wang et al. (2022) by name but does not include a full citation. A complete bibliographic entry (DOI or arXiv reference) should be added to the project's README and dissertation bibliography.

---

## 3. Geographic and Temporal Bias (Step 3.2)

### 3.1 Corn Belt Dominance

Step 3.2 quantifies that Illinois, Iowa, Indiana, Minnesota, and Nebraska together represent a disproportionate fraction of county-year records. This finding is operationalised as **Bias B2** and mitigated via inverse-frequency state loss weighting, scheduled for implementation in Step 4.2.

**Assessment:** The proposed mitigation is technically sound. Inverse-frequency weighting $w_s = N / (K \cdot N_s)$ where $N$ is the total record count, $K$ is the number of states, and $N_s$ is the number of records for state $s$, is a standard approach to group imbalance in regression. However, the notebook notes expected trade-offs of +0.5–2.0 BU/ACRE RMSE on Corn Belt states and −3–5 BU/ACRE RMSE reduction on minority states. These are _projections_ rather than empirical measurements. The final report must present actual measured trade-offs after training, not just projections.

**Ethical implication:** If the weighting is not implemented, the model effectively privileges Corn Belt farmers' predictive accuracy at the expense of accuracy for all other states. This is a direct fairness failure under the principle of **equal model performance across demographic groups** (analogous to geographic group membership).

### 3.2 Temporal Confounding in 2019

The year 2019 is flagged as confounded by the US–China trade war and unprecedented Midwest flooding. The notebook correctly identifies this year as a confounder but proposes no training treatment — only a note to evaluate it in isolation in the Leave-One-Year-Out (LOYO) supplement.

**Assessment:** This approach is reasonable for a research prototype. However, from an ethics standpoint, any publication or policy-adjacent use of the model must explicitly state that the training data includes a severe crop-loss event (2019 flooding) that is only partially represented in the label distribution. The risk is that the model **learns to under-predict severe crop failure** because 2019 is one of only seven years and its loss signal is diluted by six normal years. This is directly connected to Bias B3 (Q1 scarcity).

**Recommendation:** Add a sensitivity experiment: train once with 2019 included and once excluded; compare Q1 RMSE. Report both in the dissertation.

### 3.3 Climate Nonstationarity

Step 3.7 correctly constrains the model to its training window (2016–2022) and calls for recalibration before assuming validity beyond 2025. This is an important ethical safeguard.

**Additional consideration:** The training window 2016–2022 coincides with a warming trend and several record-high temperature events. A model trained in this window may already be miscalibrated for the cooler end of the historical climate distribution. Users should be advised that the model's accuracy in a cooler-than-average future year may be lower than the validation metrics suggest.

---

## 4. Label Imbalance and Yield Quintile Equity (Step 3.3)

### 4.1 Quantile Distribution Analysis

Step 3.3 assigns each county-year record to one of five yield quintiles (Q1–Q5) and evaluates whether low-yield observations are under-represented. This is operationalised as **Bias B3**.

**Assessment:** This is ethically the most important bias in the pipeline. A model that systematically under-predicts Q1 (low-yield/crop-failure) events will:
- **Underestimate insurance loss claims**, with direct financial harm to farmers
- **Fail to trigger early-warning signals** for food security interventions
- **Provide misleading comfort** to policymakers during incipient drought or flooding events

The notebook identifies this risk and proposes oversampling Q1 records during training and separate Q1 RMSE reporting. Both mitigations are appropriate.

**Unresolved gap:** The notebook does not specify a concrete oversampling ratio or method (e.g., SMOTE, repeat sampling, synthetic generation). For a research dissertation, the exact method used must be documented to enable reproducibility. If oversampling is not implemented, this must be stated as an explicit limitation.

### 4.2 Regression-to-the-Mean Risk

The `bias_per_group()` function includes the following criterion:

> If `Bias(Q1) > 0` and `Bias(Q5) < 0`, the model is regressing to the mean — underestimating crop failure and overestimating bumper yields.

This is technically rigorous. However, the regression-to-the-mean pathology is almost guaranteed in any neural network trained on a skewed target without explicit correction. The dissertation should report this metric and explicitly confirm or deny whether the trained model exhibits this pattern. A model that is known to exhibit regression-to-the-mean bias must carry a warning label in any applied report.

---

## 5. Fairness Metric Framework (Step 3.4)

### 5.1 Completeness Assessment

The fairness metric suite defined in Step 3.4 constitutes a solid foundation:

| Function | Metric | Group Slices |
|---|---|---|
| `sliced_rmse()` | RMSE in BU/ACRE | State, year, quintile |
| `bias_per_group()` | Mean bias in BU/ACRE | State, year, quintile |
| `cv_error_per_state()` | CV of prediction errors | State |
| `coverage_gap()` | Fraction of absent FIPS | National |
| `fairness_report()` | Composite report | All of the above |

**Positive finding:** The framework correctly flags states with CV-Error > 0.20 as "poorly served" and applies Jensen correction throughout. The threshold of 0.20 is reasonable for agricultural regression (20% of mean yield represents approximately 8–10 BU/ACRE, which is agronomically significant).

### 5.2 Missing Metrics

Several fairness metrics recommended in the literature are absent:

1. **Maximum Mean Discrepancy (MMD)** between the training and test feature distributions per state — would detect distribution shift caused by geographic imbalance.
2. **Disparate Impact Ratio**: $\text{RMSE}_{minority} / \text{RMSE}_{majority}$. The current framework requires manual comparison of sliced RMSEs; a single ratio would be more actionable.
3. **Calibration curve by quintile**: A reliability diagram plotting predicted vs actual yield per quintile would reveal systematic over/under-confidence in specific yield ranges.
4. **Temporal generalization gap**: Year-by-year RMSE trend line — if RMSE is increasing year-over-year, the model is not temporally robust.

**Recommendation:** Add at least the Disparate Impact Ratio and a calibration curve to the final results chapter. These are low-cost additions that significantly strengthen the fairness case.

### 5.3 Scope of Fairness Metrics

The current framework evaluates **predictive fairness** (equal error across groups) but does not address **allocative fairness** (whether systematic biases could cause unequal resource allocation, e.g., crop insurance payouts). For a purely academic prototype, this is acceptable. However, the dissertation should acknowledge the distinction and clarify that predictive fairness, while necessary, is not sufficient for deployment in policy contexts.

---

## 6. Jensen's Inequality Correction (Step 3.5)

### 6.1 Technical Correctness

The Jensen correction $\hat{y}_\text{BU/ACRE} = \exp\!\left(\hat{y}_\text{log} + \frac{\sigma^2}{2}\right)$ is correctly derived and implemented. Setting `JENSEN_SIGMA2` initially to the label variance (prior) and updating it to the residual variance at Step 4.3 is the methodologically correct two-stage approach.

### 6.2 Ethical Dimension

The Jensen correction is not merely a technical nicety — it has ethical weight. Without it:
- **All reported metrics are biased downward** in BU/ACRE, making the model appear more accurate than it is
- **Q1 (crop failure) predictions are selectively deflated**, since Jensen bias is largest where prediction uncertainty is highest

The notebook enforces the constraint that all reported metrics must use Jensen-corrected BU/ACRE values and states this as a mandatory constraint in Step 3.7. This is appropriate and should be verified by the dissertation supervisor.

**Risk:** The `JENSEN_SIGMA2` placeholder is initialised to `None`. If any evaluation cell is run before Step 4.3 updates this value, the `jensen_correction()` function will silently fall back to naive `exp()` (as documented in the code). This is a **research integrity risk**: results computed with `JENSEN_SIGMA2 = None` and results computed with the fitted residual variance are not comparable. The code should raise a warning (or assertion error) if `fairness_report()` is called while `JENSEN_SIGMA2` is still `None`.

---

## 7. Bias Registry B1–B9 (Step 3.6)

### 7.1 Registry Completeness

The bias registry is comprehensive and well-structured. The following table reproduces and annotates the nine identified biases:

| ID | Bias | Severity | Status | Ethics Comment |
|---|---|---|---|---|
| B1 | State-centroid weather proxy | High | Open — documented | Fundamentally compromises intra-state equity. Should be escalated from "future work" to a hard limitation warning on any reported results. |
| B2 | Corn Belt geographic dominance | High | Mitigation planned | Planned mitigation (inverse-frequency weighting) is appropriate but unverified. Must be confirmed empirically. |
| B3 | Low-yield observation scarcity | Medium | Mitigation planned | Highest downstream harm potential. Q1 underestimation has direct food-security implications. |
| B4 | Log-target Jensen bias | Medium | Correction defined | Correction is sound; risk is execution (see Section 6.2 above). |
| B5 | HRRR single-snapshot proxy | High | Acknowledged limitation | January 1 HRRR snapshot provides almost no growing-season agronomic signal. This severely limits predictive validity and must be stated prominently in the abstract. |
| B6 | USDA reporting threshold omission | Medium | Flagged in audit | Omission disproportionately affects small and minority-owned farms. Flagged correctly. |
| B7 | Random 80/20 temporal mixing | Low-Medium | Mitigation planned | LOYO supplement is the correct fix. The reported metrics without LOYO should be labelled "in-sample temporal validation" not "temporal generalization". |
| B8 | Sentinel-2 AZ 2022 tile anomaly | Medium | Mitigation planned | Likely data artefact. Correct approach is exclusion, but origin of the anomaly should be investigated before deciding. |
| B9 | USDA surveyor bias | Medium | Acknowledged | Cannot be corrected without external survey data. Constraint correctly applied. |

### 7.2 Missing Biases

The following biases are not registered but warrant consideration:

| Proposed ID | Bias | Severity | Rationale |
|---|---|---|---|
| B10 | Infrastructure disparity in sensor availability | Medium | Counties with fewer economic resources may have poorer Sentinel-2 cloud coverage or ground truthing, introducing a wealth-correlated data quality bias. |
| B11 | Model architecture inductive bias | Low | A Bi-LSTM assumes sequential temporal dependencies in what is effectively a single-snapshot annual feature vector (seq_len=1). This is a fundamental architectural mismatch that inflates model complexity without adding representational accuracy. |
| B12 | Climate-crop interaction nonstationarity | Medium | The relationship between HRRR weather features and yield is mediated by crop variety, soil type, and irrigation access — none of which are in the feature set. The model may capture spurious correlations rather than causal pathways. |

---

## 8. Ethical Use Constraints (Step 3.7)

### 8.1 Assessment of Declared Constraints

Step 3.7 specifies six ethical use constraints in tabular form. These are reproduced and evaluated below:

| Constraint | Assessment |
|---|---|
| **No individual-farm inference** | Correctly enforced. County-level aggregates cannot legally or statistically be disaggregated to farm level. This constraint is necessary and sufficient. |
| **Food security embargo** | Critically important. Releasing predictions ahead of official NASS reports is prohibited by 7 USC §2276 and constitutes a material risk of commodity market manipulation. The constraint should cite this legal provision explicitly. |
| **Geographic equity disclosure** | Essential. The ~55% coverage gap must appear in any results summary, not just in a supplementary ethics document. Recommendation: add a mandatory header to all output tables flagging coverage gap magnitude. |
| **Climate nonstationarity** | Well-framed. The 2025 recalibration deadline is specific and useful. |
| **LOYO supplement required** | Correctly stated but currently unimplemented. Any temporal-generalization claim must await LOYO results. |
| **Jensen correction mandatory** | Correct. Should be enforced programmatically (see Section 6.2). |

### 8.2 Missing Constraints

The following constraints should be added:

1. **No deployment in high-stakes policy contexts without external validation.** The Corn Belt dominance (B2) means the model's generalisation error in Southern states has not been validated against independent data. Any state-level policy use requires external validation by that state's agricultural extension service.

2. **No model stacking onto other decision systems without audit.** If the model is embedded as a component in a larger pipeline (e.g., an insurance pricing engine), a fresh fairness audit is required for the combined system.

3. **Mandatory uncertainty quantification at inference.** The current architecture produces a point estimate with no confidence interval. Any operational use must attach a predictive interval (e.g., quantile regression bands or bootstrap confidence intervals) to each county-year prediction.

4. **No use with varieties, practices, or geographies outside the training distribution.** The model has never encountered organic soybean production, drip-irrigated production, or tropical cultivation contexts. Extrapolating to these regimes is scientifically and ethically invalid.

---

## 9. Key Ethical Tradeoffs (Step 3.7 — Extended Analysis)

Step 3.7 identifies five key tradeoffs. Below is an extended analysis of each:

### Accuracy vs Fairness

The inverse-frequency weighting scheme is the primary mechanism for accuracy–fairness tradeoff management. The projected +0.5–2.0 BU/ACRE RMSE increase in Corn Belt states is the cost of more equitable representation of minority states. This is an **acceptable and deliberate tradeoff** consistent with Rawlsian fairness principles (maximising the welfare of the least-advantaged group). The dissertation must transparently report both weighted and unweighted metrics side-by-side and justify the weighting choice.

### Model Complexity vs Interpretability

The recommendation to use Ridge Regression for policy contexts is sound and reflects responsible AI practice. However, the report should go further: the narrative in the dissertation should distinguish between:
- **Research claims** (which may use the GMU and are justified by academic interest in multimodal fusion)
- **Policy claims** (which must exclusively use interpretable, auditable models that satisfy explainability requirements under EU AI Act Article 13)

### Data Coverage vs Data Quality

Including the confounded 2019 observation is a reasonable pragmatic choice for a small dataset (n=3,430). However, the dissertation should present a clear rationale for this choice and acknowledge that it represents a trade-off between training set size and label purity. The LOYO supplement for 2019 will serve as the empirical validation of this decision.

### Log Transform vs Direct Interpretability

The `USE_LOG=True` flag improves optimisation dynamics and is standard practice for right-skewed regression targets. The ethical requirement is that **all externally reported results are in BU/ACRE with Jensen correction applied**. Internal training can remain in log-space. The notebook enforces this correctly.

### Modality Richness vs Deployability

The Sentinel–USDA FIPS gap renders Phase 2 and 3 untrainable with the current data. This is documented as "a data infrastructure finding — not a model deficiency", which is accurate and appropriately frames expectations. The ethical dimension is that **this gap must not be minimised in the dissertation abstract or introduction**. The practical limitation that only Phase 1 can be trained should be prominently stated in the contributions section.

---

## 10. Research Integrity and Reproducibility

### 10.1 Strengths

- Global random seed (`SEED = 42`) set across `numpy`, `random`, and `torch` ensures reproducibility of the train/test split and model weight initialisation.
- Data split logic is stratified (yield quintile for Phase 1; yield median for Phase 2/3), reducing variance in evaluation metrics.
- Scaler and imputer objects are serialised to disk (`hrrr_scaler.pkl`, `hrrr_imputer.pkl`), preventing train/test leakage at inference time.
- The `USE_LOG` flag is a single global switch, preventing inconsistency between training and inference.

### 10.2 Concerns

1. **Hardcoded credentials (Cell 1):** The `sentinel_client_id` and `sentinel_client_secret` are stored as plaintext strings in the first notebook cell. This is a **critical security and research integrity vulnerability**. If the notebook is shared (e.g., with a dissertation examiner), these credentials will be exposed. They should be removed from the notebook immediately and loaded from environment variables or a `.env` file that is excluded from version control.

2. **Missing data versioning:** The notebook does not include checksums (MD5/SHA-256) for the downloaded GRIB2 or CSV files. If the source data files change (e.g., USDA NASS reissues a corrected survey), the results cannot be verified without re-running the entire pipeline. A data manifest with file hashes should be added.

3. **Single random seed:** Using a single seed (42) is reproducible but does not communicate _variance across seeds_. Results should be reported as the mean ± std across at least three random seeds to characterise model stability.

4. **`JENSEN_SIGMA2 = None` initialisation risk:** As noted in Section 6.2, evaluation cells that are run out-of-order will produce uncorrected metrics without warning. An assertion guard is recommended.

---

## 11. Alignment with Regulatory and Professional Frameworks

### 11.1 EU AI Act (2024)

Under the EU AI Act, agricultural yield prediction systems used in food security planning, agricultural insurance, or input-supply management would likely be classified as **high-risk AI systems** (Annex III: AI systems in critical infrastructure and essential services). The key requirements for compliance include:

| Requirement | Current Status | Gap |
|---|---|---|
| Risk management system (Article 9) | Partial — bias registry B1–B9 serves this function | Missing: risk severity scoring and residual risk assessment |
| Data governance (Article 10) | Partial — data sources documented | Missing: data provenance logs, update procedures |
| Technical documentation (Article 11) | Partial — notebook is documentation | Missing: structured model card |
| Transparency (Article 13) | Partially met | Missing: end-user-facing plain-language disclosure |
| Human oversight (Article 14) | Not addressed | No oversight mechanism defined |
| Accuracy, robustness (Article 15) | Partially addressed | LOYO evaluation pending |

**Recommendation:** Prepare a model card (Mitchell et al., 2019) for the dissertation appendix that covers all AI Act Article 11 requirements and links to the bias registry and ethical use constraints in this report.

### 11.2 NIST AI Risk Management Framework (RMF 1.0)

The NIST AI RMF maps onto four functions: **GOVERN, MAP, MEASURE, MANAGE**. Step 3 primarily addresses the MEASURE function (identifying and assessing AI risks). The GOVERN function (policies, accountability) and MANAGE function (risk response and monitoring) are not addressed in the current pipeline and should be treated in the dissertation's discussion chapter.

### 11.3 IEEE P7003 — Algorithmic Bias Considerations

IEEE P7003 requires that bias identification be coupled with a **bias impact statement** — a qualitative assessment of the harm that each identified bias could cause to affected stakeholders. The current bias registry (Step 3.6) documents technical mitigations but does not identify the **human stakeholders** whose interests are affected by each bias. A brief stakeholder map is recommended:

| Stakeholder | Affected Bias IDs | Potential Harm |
|---|---|---|
| Smallholder farmers | B9, B6 | Exclusion from model predictions; misrepresentation of farm-scale risk |
| Southern/Western state farmers | B1, B2 | Lower prediction accuracy; under-served by insurance and advisory systems |
| Crop failure-affected farmers | B3, B5 | Under-predicted yield loss; delayed or insufficient insurance payouts |
| Food policy analysts | B7, B5 | Misleading temporal generalisation claims |
| Commodity traders | B4, B5 | Systematic prediction bias if Jensen correction is not applied consistently |

---

## 12. Overall Ethical Risk Assessment

| Risk Domain | Rating | Key Evidence |
|---|---|---|
| Data representativeness | High | 55% county gap; Corn Belt dominance; USDA surveyor bias |
| Algorithmic fairness | Medium-High | B2 mitigation planned but unverified; Q1 scarcity risk |
| Temporal validity | Medium | LOYO supplement pending; 2019 confounding unmitigated in training |
| Transparency & reproducibility | Medium | Hardcoded credentials; missing data versioning; single seed |
| Legal and regulatory compliance | Medium | No model card; EU AI Act high-risk criteria partially met |
| Downstream societal harm | Medium-High | Food security and insurance use cases require formal deployment audit |

---

## 13. Recommendations

The following actions are recommended before the dissertation is finalised:

### Priority 1 — Immediate (Before Results Are Reported)

1. **Remove hardcoded Sentinel API credentials** from Cell 1. Use `os.environ.get()` or `python-dotenv`.
2. **Add assertion guard** in `fairness_report()`: raise `ValueError` if `JENSEN_SIGMA2 is None` and `USE_LOG=True`.
3. **Confirm Jensen correction** is applied in every evaluation cell (Steps 4.3, 6.4, 7.4) before recording metrics.

### Priority 2 — Before Dissertation Submission

4. **Produce a spatial coverage map** (choropleth) in Step 3.1 showing which counties are absent from the training set.
5. **Implement and report LOYO evaluation** for 2022 before any temporal-generalisation claims.
6. **Measure the actual Corn Belt accuracy trade-off** from inverse-frequency weighting (Step 4.2) and report observed vs projected RMSE changes.
7. **Report results across three random seeds** (not only seed 42) to characterise model stability.
8. **Add a 2019 sensitivity experiment**: train with and without 2019; compare Q1 RMSE.

### Priority 3 — For Full Ethical Compliance

9. **Prepare a model card** (following Mitchell et al., 2019) in the dissertation appendix covering all EU AI Act Article 11 requirements.
10. **Add the stakeholder harm map** (Section 11.3 above) to the dissertation ethics chapter.
11. **Add proposed Biases B10–B12** to the bias registry (infrastructure disparity, architectural mismatch, causal confounding).
12. **Cite 7 USC §2276** in the food-security embargo constraint.
13. **Add a data manifest** with file checksums for all downloaded data sources.

---

## 14. Conclusion

Step 3 of the multimodal agriculture prediction pipeline demonstrates a **commendable level of proactive ethical integration** that is rare in technical AI research. The inclusion of a structured bias registry, fairness metric functions, Jensen's inequality correction, and explicit use constraints reflects genuine methodological responsibility.

The primary ethical vulnerabilities identified in this review are:

1. **The 55% county coverage gap and Corn Belt dominance** create a system that, if deployed or cited beyond its training distribution, will systematically under-serve non-Corn-Belt farmers — a group that is already economically less-represented in agricultural policy.

2. **The Q1 (crop failure) scarcity risk** has the highest potential downstream harm, as it creates systematic under-prediction of yield loss events that drive insurance, food aid, and disaster relief decisions.

3. **The HRRR single-snapshot proxy** (Bias B5) is a high-severity acknowledged limitation that fundamentally constrains the model's agronomic validity. This must be prominently disclosed in all reporting.

4. **The hardcoded API credentials** in Cell 1 are an immediate security and research integrity risk that should be resolved before the notebook is shared with any third party.

With the Priority 1 actions implemented and the Priority 2 analyses added to the dissertation, this research will meet a high standard of ethical AI practice commensurate with its academic context.

---

## References

- USDA NASS (2022). *National Agricultural Statistics Service Survey Methodology.* [https://www.nass.usda.gov/](https://www.nass.usda.gov/)
- Wang, Y. et al. (2022). *CropNet: A Large-Scale Dataset and Benchmark for Multimodal Crop Yield Prediction.* arXiv preprint.
- Mitchell, M. et al. (2019). Model Cards for Model Reporting. *Proceedings of FAccT 2019.*
- European Parliament (2024). *EU Artificial Intelligence Act*, Regulation (EU) 2024/1689.
- NIST (2023). *AI Risk Management Framework 1.0.* National Institute of Standards and Technology.
- IEEE (2021). *P7003 Standard for Algorithmic Bias Considerations.*
- Jensen, J.L.W.V. (1906). Sur les fonctions convexes et les inégalités entre les valeurs moyennes. *Acta Mathematica*, 30, 175–193.
- USDA Census of Agriculture (2017). *Farm Demographics: U.S. Farmers by Gender, Age, Race, Ethnicity.* ACH12-3/February 2014.

---

*This report was prepared as an independent ethics review of Step 3 of the multimodal agriculture prediction pipeline. All technical claims are derived directly from the notebook source code and documented outputs in `test.ipynb`.*
