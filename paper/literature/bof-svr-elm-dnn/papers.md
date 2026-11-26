# Paper cards and comparison

The structured record is in [papers.json](papers.json). The cards below summarize the verified scope and what can/cannot be transferred to APMCM Problem B.

## 1. BOF flame-spectrum and gas prediction

### Wu et al. (2026) — Autoencoder + Optuna + GWO-BiLSTM
- **Title:** An improved flue gas analysis model based on the correction of the converter mouth flame spectrum
- **Venue / DOI:** *Ironmaking & Steelmaking* (OnlineFirst); [10.1177/03019233251382772](https://doi.org/10.1177/03019233251382772)
- **Task:** BOF flame-spectrum-based gas-emission prediction, with process-stage segmentation.
- **Method:** Autoencoder dimensionality reduction with Optuna; BiLSTM temporal regression; Grey Wolf Optimization for BiLSTM parameter tuning.
- **Reported abstract metrics:** CO MAE 18.0017, CO RMSE 20.2151; CO2 MAE 1.7307, CO2 RMSE 2.1321. The accessible abstract does not state metric units.
- **Evidence:** Abstract only; publisher page and DOI verified, full text not accessed.
- **Transfer:** Closest match to spectrum-to-gas prediction. Inspect exact target definitions and validation split before adopting; the abstract does not establish cross-heat generalization.

### Shao et al. (2014) — phase-aware SVC + epsilon-SVR
- **Title:** BOF endpoint prediction based on the flame radiation by hybrid SVC and SVR modeling
- **Venue / DOI:** *Optik*, 125(11), 2491–2496; [10.1016/j.ijleo.2013.10.094](https://doi.org/10.1016/j.ijleo.2013.10.094)
- **Task:** BOF endpoint prediction from flame radiation.
- **Method:** SVC separates blowing phases; phase-aware epsilon-SVR performs regression.
- **Evidence:** Publisher abstract only; full text not accessed, numeric metrics not exposed.
- **Transfer:** Compare a single global SVR against phase-classifier-plus-SVR. Endpoint carbon is not the same target as continuous CO/CO2 flow.

### Gao et al. (2019) — wavelet weighted twin-SVR
- **Title:** End-Point Static Control of Basic Oxygen Furnace (BOF) Steelmaking Based on Wavelet Transform Weighted Twin Support Vector Regression
- **Venue / DOI:** *Complexity*, Article 7408725; [10.1155/2019/7408725](https://doi.org/10.1155/2019/7408725)
- **Task:** BOF endpoint carbon and temperature prediction in a static-control framework.
- **Method:** Wavelet denoising, normalized samples, weighted twin-SVR with an RBF kernel; two quadratic programming problems.
- **Reported full-text metrics:** 92% carbon hit rate within 0.005%; 96% temperature hit rate within 10 °C; double hit rate 90%.
- **Evidence:** Open publisher full text inspected.
- **Transfer:** Useful SVR variant and noise-handling reference. Its 220-sample endpoint task is not directly comparable with ten heats of spectral time-series data.

### Liu et al. (2021) — WNPSVR + WOA
- **Title:** End-point prediction of 260 tons basic oxygen furnace (BOF) steelmaking based on WNPSVR and WOA
- **Venue / DOI:** *Journal of Intelligent & Fuzzy Systems*, 41(2), 2923–2937; [10.3233/JIFS-210007](https://doi.org/10.3233/JIFS-210007)
- **Task:** 260-ton BOF endpoint carbon and temperature prediction.
- **Method:** K-nearest-neighbor sample weighting; weighted nonparallel SVR; Whale Optimization Algorithm for model parameters.
- **Reported abstract metrics:** 89% carbon hit rate within 0.005 wt.%; 95% temperature hit rate within 10 °C; double hit rate 85%.
- **Evidence:** Publisher abstract only; full text not accessed.
- **Transfer:** Candidate optimizer comparison. WOA is not guaranteed to improve this competition's model.

## 2. ELM and optimized ELM

### Han & Liu (2014) — evolutionary membrane algorithm + ELM
- **Title:** Endpoint prediction model for basic oxygen furnace steel-making based on membrane algorithm evolving extreme learning machine
- **Venue / DOI:** *Applied Soft Computing*, 19, 430–437; [10.1016/j.asoc.2013.09.012](https://doi.org/10.1016/j.asoc.2013.09.012)
- **Task:** BOF endpoint carbon and temperature prediction.
- **Method:** ELM with evolutionary membrane algorithm adjusting input weights and hidden-layer thresholds/parameters.
- **Evidence:** Publisher abstract and author-profile summary; full text not accessed; no numeric result in accessible abstract.
- **Transfer:** Compare ordinary ELM with optimized ELM over repeated seeds; report variation, not just best run.

### Zhang et al. (2019) — IPSO-RELM
- **Title:** Improved prediction model for BOF end-point manganese content based on IPSO-RELM method
- **Venue / DOI:** *Chinese Journal of Engineering*, 41(8), 1052–1060; [10.13374/j.issn2095-9389.2019.08.011](https://doi.org/10.13374/j.issn2095-9389.2019.08.011)
- **Task:** BOF endpoint manganese prediction.
- **Method:** Regularized ELM optimized by improved particle swarm optimization; compared with BP, ELM and RELM.
- **Reported abstract metrics:** 94% hit ratio within ±0.025%; MSE 2.18×10⁻⁸; reported fitting degree R² = 0.72.
- **Evidence:** Official journal abstract; full text not inspected.
- **Transfer:** A useful ablation template: ELM → RELM → IPSO-RELM. Its manganese target and error tolerance must not be conflated with CO/CO2 metrics.

### Qi et al. (2021) — JITL + vMF mixture + WELM
- **Title:** Just-in-time-learning based prediction model of BOF endpoint carbon content and temperature via vMF mixture model and weighted extreme learning machine
- **Venue / DOI:** *Computers & Chemical Engineering*, 154, Article 107488; [10.1016/j.compchemeng.2021.107488](https://doi.org/10.1016/j.compchemeng.2021.107488)
- **Task:** BOF endpoint prediction under changing raw-material batches and imbalanced process data.
- **Method:** vMF mixture model selects similar historical samples; weighted ELM performs local regression.
- **Evidence:** Publisher abstract only; numeric metrics not exposed.
- **Transfer:** Useful hypothesis for heat-to-heat distribution shift. A local/JITL model must select neighbors only from training heats.

## 3. Deep neural networks and temporal fusion

### Liu et al. (2020) — stacked autoencoder + sparse Bayesian regression
- **Title:** A Stacked Autoencoder With Sparse Bayesian Regression for End-Point Prediction Problems in Steelmaking Process
- **Venue / DOI:** *IEEE Transactions on Automation Science and Engineering*, 17(2), 550–561; [10.1109/TASE.2019.2935314](https://doi.org/10.1109/TASE.2019.2935314)
- **Task:** Steelmaking endpoint prediction.
- **Method:** Stacked autoencoder representation with sparse Bayesian regression.
- **Evidence:** Metadata plus a short abstract fragment only; IEEE publisher page was inaccessible in this session. Detailed input variables, architecture and metrics are not verified.
- **Transfer:** Possible encoder-plus-regression reference; obtain and read full text before reusing details.

### Hu et al. (2025) — multi-scale FPN + Transformer self-attention
- **Title:** Research on prediction model of converter steelmaking process based on multi-scale feature extraction and self-attention mechanism for multi-source domain asynchronous data
- **Venue / DOI:** *Measurement*, Article 117060; [10.1016/j.measurement.2025.117060](https://doi.org/10.1016/j.measurement.2025.117060)
- **Task:** BOF operating-condition classification from multi-source asynchronous process data.
- **Method:** Explicit time-offset parameter δt; Feature Pyramid Network for multi-scale feature extraction/fusion; Transformer self-attention for long-range dependencies.
- **Reported abstract metrics:** At 15-second window, precision 90.1%, recall 90.02%, F1 89.7%.
- **Evidence:** Publisher abstract only.
- **Transfer:** Very relevant to asynchronous sensor alignment and multi-scale fusion, but this is classification, not continuous gas-flow regression.

### Zhou et al. (2023) — domain-knowledge spatio-temporal deep LSTM
- **Title:** A soft sensor modeling framework embedded with domain knowledge based on spatio-temporal deep LSTM for process industry
- **Venue / DOI:** *Engineering Applications of Artificial Intelligence*, Article 106847; [10.1016/j.engappai.2023.106847](https://doi.org/10.1016/j.engappai.2023.106847)
- **Task:** Soft sensing in two mineral-processing case studies.
- **Method:** Time-delay analysis and variable alignment; domain-knowledge-based process decoupling; spatio-temporal structure embedded in deep LSTM.
- **Evidence:** Publisher abstract only; no numeric metrics exposed.
- **Transfer:** Cross-domain method reference for explicit delay estimation and process-informed sequence modeling, not BOF evidence.

### Zeng et al. (online 2025; issue 2026) — meta-learning stacked autoencoder
- **Title:** A Dynamically Adaptive Soft Sensor for End-Point Carbon Content and Temperature Prediction in Multimodal Basic Oxygen Furnace Steelmaking using a Meta-Learning Stacked Autoencoder
- **Venue / DOI:** *Steel Research International*, 97(5), 2805–2821; [10.1002/srin.202500757](https://doi.org/10.1002/srin.202500757)
- **Publication note:** Online first 2025-12-01; assigned to the 2026 issue.
- **Task:** BOF endpoint carbon and temperature prediction under multimodal operating modes and data drift.
- **Method:** Wasserstein-distance-weighted Dirichlet Process Gaussian Mixture Model for operating-mode clustering; meta-learned SAE; mutual-information-weighted Jensen-Shannon divergence for similar-sample selection and local fine-tuning.
- **Evidence:** Publisher abstract only; full text not retrieved and numeric metrics not exposed.
- **Transfer:** Strong motivation for mode-aware/global-versus-grouped models. It does not establish continuous CO/CO2 regression performance.

### Wu et al. (2026) — spectrum-to-gas BiLSTM
See the first card; this is the most directly aligned deep sequence-model study in this set.

## 4. Supporting spectral feature extraction and broader BOF context

### Zhao et al. (2023) — BEADS + GA rough-set feature reduction + BP
- **Title:** Research on prediction model of converter temperature and carbon content based on spectral feature extraction
- **Venue / DOI:** *Scientific Reports*, 13, Article 14409; [10.1038/s41598-023-41751-9](https://doi.org/10.1038/s41598-023-41751-9)
- **Task:** BOF endpoint temperature and carbon prediction from flame spectra.
- **Method:** BEADS baseline correction/noise reduction; GA-optimized rough-set attribute reduction; eight wavelength features; BP neural network.
- **Reported full-text metrics:** 978 records; random row-wise split 600 train / 378 test. Temperature RMSE 3.85 K; carbon RMSE about 0.021%; carbon hit rate 94.0% within ±0.05%.
- **Evidence:** Open-access full text inspected.
- **Transfer:** Strong reference for spectral preprocessing and channel reduction. Its row-wise random split is not a valid substitute for holding out complete heats.

### Liu et al. (2025) — dynamic flame video features + 3D multi-layer complex networks
- **Title:** Dynamic flame feature-driven prediction model for basic oxygen furnace steelmaking endpoint carbon content based on three-dimensional multi-layer complex networks
- **Venue / DOI:** *Engineering Applications of Artificial Intelligence*, 139, Article 109564; [10.1016/j.engappai.2024.109564](https://doi.org/10.1016/j.engappai.2024.109564)
- **Task:** BOF endpoint carbon prediction from flame video.
- **Method:** Dynamic multi-frame color/texture and spatial features represented with a three-dimensional multi-layer complex network; regression head.
- **Reported abstract metrics:** 87.91% within ±0.02%; R² 0.8547; RMSE 2.0959 (unit not exposed in abstract).
- **Evidence:** Publisher abstract/preview only.
- **Transfer:** Suggests dynamic flame features may matter, but video/image features are not the same as the contest's 2047-channel spectra.

### Ghalati et al. (2023) — BOF machine-learning review
- **Title:** Toward learning steelmaking—A review on machine learning for basic oxygen furnace process
- **Venue / DOI:** *Materials Genome Engineering Advances*, 1(1), e6; [10.1002/mgea.6](https://doi.org/10.1002/mgea.6)
- **Task:** Review of BOF machine-learning applications, data workflows, SVR, ANN and future directions.
- **Evidence:** Open-access full review inspected.
- **Transfer:** Best starting point for expanding the bibliography. Trace quantitative claims to primary studies rather than citing review summaries as new experiments.

## Comparison matrix

| Family | Representative papers | Strength suggested by literature | Main caution for Problem B |
|---|---|---|---|
| SVR | Shao 2014; Gao 2019; Liu 2021 | Strong nonlinear regression baseline; phase-aware and weighted variants exist | Most studies predict endpoint carbon/temperature, not continuous gas flow |
| ELM | Han 2014; Zhang 2019; Qi 2021 | Fast shallow regression; optimization, regularization and local weighting can be tested | Small number of independent heats makes optimizer overfitting a risk |
| Deep sequence models | Wu 2026; Hu 2025; Zhou 2023 | Learn spectral representation, temporal dependence and cross-modal fusion | More parameters and stronger overfit risk with only ten heats |
| Spectral preprocessing | Zhao 2023 | Baseline correction and feature selection reduce high-dimensional input | Random row split leaks heat-specific temporal structure |
| Cross-heat adaptation | Qi 2021; Zeng 2026 issue | Local selection and mode-aware adaptation address process drift | Must not use held-out heat data to select training neighbors or tune the model |

## Reference URLs

The DOI and publisher URLs are recorded in each entry in [papers.json](papers.json). Use the DOI link as the stable citation target.
