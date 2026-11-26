# Free full-text reading list — BOF / Problem B

Checked: 2026-11-26. This list contains publisher-hosted free full text or open-access papers that were actually inspected. It does not include papers for which only an abstract was available.

## Priority 1 — closest match to Problem B

### An et al. (2026): physics-aware cumulative decarburization
- **Title:** A Physics-Aware and Interpretable Framework for Predicting Cumulative Decarburization in Basic Oxygen Furnace (BOF) Steelmaking
- **Journal / DOI:** *Applied Sciences*, 16(6), 3059. [10.3390/app16063059](https://doi.org/10.3390/app16063059)
- **Free full text:** [MDPI HTML](https://www.mdpi.com/2076-3417/16/6/3059) · [PDF](https://www.mdpi.com/2076-3417/16/6/3059/pdf)
- **Access / license:** Publisher Open Access, CC BY.
- **Inspected scope:** acquisition, lag estimation, target construction, spectral feature extraction, PCA, heat-level split, model optimization and comparison.

**What the paper does**
- Acquires flame spectra and CO/CO2/O2 gas data at 0.5 s intervals; spectra span 300–1100 nm with 2048 channels.
- Estimates transport delay by averaging cross-correlation across all wavelength channels against the off-gas CO series. For example heat A1, the peak corresponds to about 24 s.
- Uses CO + CO2 instantaneous flow to construct total carbon-gas flow and integrates it with the composite trapezoidal rule (Δt = 0.5 s) to obtain cumulative decarburization.
- Selects characteristic lines for C, Si, Mn, P, Fe, O, Ca, Na and K using NIST spectral information; performs element-wise PCA retaining 90% cumulative explained variance.
- Builds a Stacking model with Random Forest, Extra Trees, XGBoost and CatBoost base learners and ElasticNet as the meta-learner; compares with SVR and FCNN.
- Splits by complete heats: converter A uses 6 heats for training and 1 for testing; converter B uses 5 for training and 1 for testing. Lag compensation and preprocessing are fitted using training heats.

**Why it matters**
This is the most directly relevant free paper found so far: the sensing modalities, 0.5 s sampling, high-dimensional flame spectra, off-gas delay and heat-wise validation resemble the competition.

**Important differences**
- Its prediction target is cumulative decarburization, not separate instantaneous CO and CO2 trajectories.
- Its within-converter delay is treated as relatively stable and estimated from training heats; the competition statement says delay varies with oxygen-lance flow. Adapt the method to flow-dependent delay rather than blindly using one constant lag.
- Industrial data are not released with the paper.

**Practical ideas to test**
1. Full-spectrum averaged cross-correlation as a stable lag baseline.
2. Mechanism-informed wavelength selection, then within-element PCA.
3. CO + CO2 integration as a cross-check for cumulative decarburization.
4. Leave-one-heat-out validation with all preprocessing fitted only on training heats.

## Priority 2 — direct SVR and DNN feature-selection comparison

### Liu et al. (2025): feature selection for BOF oxygen supply
- **Title:** Evaluation of Feature Selection Methods for Oxygen Supply Prediction in BOF Steelmaking
- **Journal / DOI:** *ISIJ International*, 65(10), 1454–1462. [10.2355/isijinternational.ISIJINT-2025-165](https://doi.org/10.2355/isijinternational.ISIJINT-2025-165)
- **Free full text:** [J-STAGE HTML](https://www.jstage.jst.go.jp/article/isijinternational/65/10/65_ISIJINT-2025-165/_html/-char/en) · [PDF](https://www.jstage.jst.go.jp/article/isijinternational/65/10/65_ISIJINT-2025-165/_pdf)
- **Access / license:** J-STAGE Open Access, CC BY-NC-ND 4.0.
- **Inspected scope:** feature-selection definitions, model setup, GWO search, validation design and results.

**What the paper does**
- Uses 2779 valid BOF production samples and 19 process features to predict total oxygen supply.
- Compares Pearson correlation (PCC), Spearman rank correlation (SCC), mutual information (MI) and maximal information coefficient (MIC).
- Selects the top eight features from each ranking and evaluates Elastic Net, SVR, XGBoost, DNN and KNN.
- Uses GWO with population 30 and 100 iterations for model hyperparameter search; applies stratified five-fold validation by oxygen-supply interval.
- Reports MIC average RMSE 197.4 m³ and R² 0.649. The PI-SHAP subset reduces DNN RMSE from 198.5 m³ to 194.6 m³.

**Why it matters**
This gives a concrete, reproducible comparison of SVR and DNN with different feature-selection methods, plus a GWO search setup and a PI-SHAP interpretation workflow.

**Important differences**
- Predicts total oxygen supply from tabular process variables, not CO/CO2 from flame spectra.
- Its sample-level stratified folds are not equivalent to holding out a complete heat. For the competition, leave-one-heat-out should remain the primary validation protocol.

## Priority 3 — BOF Transformer for tabular process data

### Khaksar Ghalati et al. (2025): TTB-BOFNet
- **Title:** Deep Transformers for Analyzing BOF Steelmaking Data
- **Journal / DOI:** *Metallurgical and Materials Transactions B*, 56, 4201–4217. [10.1007/s11663-025-03615-7](https://doi.org/10.1007/s11663-025-03615-7)
- **Free full text:** [Springer HTML](https://link.springer.com/article/10.1007/s11663-025-03615-7) · [PDF](https://link.springer.com/content/pdf/10.1007/s11663-025-03615-7.pdf)
- **Access / license:** Springer Open Access, CC BY 4.0.
- **Inspected scope:** attention equations, architecture, dataset, preprocessing, training objective and test results.

**What the paper does**
- Uses 10,168 BOF operations across 259 steel grades/sub-grades and three furnaces.
- Converts ten selected numeric features into EDA-informed categorical bins, especially sparse variables with many zeros.
- Uses column embeddings and Transformer blocks to learn contextual feature interactions; concatenates the contextual embeddings with remaining numeric features and predicts endpoint temperature with an MLP head.
- Reports test R² 0.8435 for TTB-BOFNet, compared with 0.8287 for TabTransformer and 0.8060 for MLP.

**Why it matters**
Useful for feature engineering when process variables are sparse or categorical, and for understanding how attention can model feature interactions.

**Important differences**
This is tabular endpoint-temperature regression, not time-series or spectral-to-gas prediction. Its industrial dataset is far larger than the ten competition heats; do not compare its R² directly with the competition's metrics.

## Priority 4 — Transformer for dynamic flame video

### Yang et al. (2026): dynamic flame image Transformer
- **Title:** Transformer-Based Dynamic Flame Image Analysis for Real-Time Carbon Content Prediction in BOF Steelmaking
- **Journal / DOI:** *Metals*, 16(2), 185. [10.3390/met16020185](https://doi.org/10.3390/met16020185)
- **Free full text:** [MDPI HTML](https://www.mdpi.com/2075-4701/16/2/185) · [PDF](https://www.mdpi.com/2075-4701/16/2/185/pdf)
- **Access / license:** MDPI Open Access, CC BY.
- **Inspected scope:** label construction, optical-flow equations, Transformer architecture, batch split, training settings and ablation results.

**What the paper does**
- Converts RGB flame frames into four-channel inputs by adding optical flow calculated with Farnebäck.
- Uses sparse expert carbon annotations (2–6 key time points plus endpoint sub-lance measurement) and cubic-spline interpolation to generate intermediate labels.
- Discretizes carbon into 36 classes and uses a six-layer Transformer encoder-decoder with eight attention heads, d_model 512, AdamW, learning rate 2e-5 and 100 epochs.
- Uses 12 of 13 batches for development and holds out one complete batch for independent testing.
- Reports validation top-1 accuracy 90.31%, ±0.02 tolerance accuracy 96.82% and weighted F1 0.9035. The abstract reports approximately 75% of independent-test errors within ±0.05.

**Why it matters**
Useful for temporal attention, positional encoding, dynamic flame information and ablation design.

**Important differences**
This paper uses flame video rather than spectral arrays and predicts carbon categories rather than gas flow. The model is large relative to a ten-heat competition dataset. Treat it as an adjacent-method reference, not a ready-made architecture.

## Also now freely readable: IPSO-RELM

### Zhang et al. (2019): IPSO-RELM for endpoint manganese
- **Title:** Improved prediction model for BOF end-point manganese content based on IPSO-RELM method
- **Journal / DOI:** *Chinese Journal of Engineering*, 41(8), 1052–1060. [10.13374/j.issn2095-9389.2019.08.011](https://doi.org/10.13374/j.issn2095-9389.2019.08.011)
- **Free full text:** [Publisher HTML](https://cje.ustb.edu.cn/article/doi/10.13374/j.issn2095-9389.2019.08.011?viewType=HTML) · [Publisher PDF](https://cje.ustb.edu.cn/cn/article/pdf/preview/10.13374/j.issn2095-9389.2019.08.011.pdf)
- **Inspected scope:** PSO/IPSO formulas, ELM and RELM equations, IPSO modifications, model design and results.

The paper combines regularized ELM with an improved particle swarm optimizer to tune the input weights and hidden-layer biases. It compares BP, ELM, RELM and IPSO-RELM on BOF endpoint manganese. Reported IPSO-RELM results: 94% hit rate within ±0.025%, MSE 2.18×10⁻⁸ and R² 0.72.

**Caution:** endpoint manganese is not a continuous CO/CO2 target. Use this as an ELM optimization reference, not as direct evidence of performance on Problem B.

## Access-status summary

| Paper | Full text free? | Main use for Problem B |
|---|---|---|
| An et al. 2026 | Yes — MDPI CC BY | Spectrum/off-gas alignment, physical spectral features, cumulative target, heat-wise split |
| Liu et al. 2025 | Yes — J-STAGE CC BY-NC-ND | SVR/DNN comparison, feature selection, GWO, PI-SHAP |
| Khaksar Ghalati et al. 2025 | Yes — Springer CC BY | Tabular Transformer, sparse-feature binning, SHAP |
| Yang et al. 2026 | Yes — MDPI CC BY | Flame-video temporal Transformer, optical flow, sequence classification |
| Zhang et al. 2019 | Yes — publisher-hosted free PDF/HTML | IPSO-RELM equations and optimization; license not independently verified |

## Current gaps

- A directly relevant, freely readable BOF paper on ELM optimization beyond the 2019 IPSO-RELM paper has not yet been verified.
- The 2026 flame-spectrum-to-gas Autoencoder + GWO-BiLSTM paper (DOI 10.1177/03019233251382772) remains restricted on the publisher page; no legal free author manuscript was confirmed in this search.
- The 2025 asynchronous FPN + Transformer BOF paper (DOI 10.1016/j.measurement.2025.117060) also has not yet been confirmed in a legal free full-text version.

Do not treat abstracts as full papers. Keep those records marked abstract_only until a free full text is located and inspected.
