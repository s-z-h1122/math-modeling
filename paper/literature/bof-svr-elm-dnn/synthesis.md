# Problem-specific synthesis and proposed experiments

> This document separates (A) facts stated in the contest problem, (B) findings reported by cited papers, and (C) proposed experiments for the team. Proposed experiments are hypotheses to test, not established results.

## A. Contest requirements (source-derived)

The 2026 APMCM Problem B statement provides ten converter heats. Each heat contains:

- Flame spectral intensity sampled every 0.5 seconds, about 1400 rows by 2047 wavelength channels, spanning 340.535 nm to 1027.678 nm.
- CO2 and CO flue-gas time series sampled every 0.5 seconds.
- Cumulative oxygen consumption.

The statement explicitly defines the CO2 and CO values in the sample table as **instantaneous volume flow rates (m3/s)**, although the table column labels use concentration wording. The team should preserve the explicit unit definition and explain the naming ambiguity in the report.

Question 1 asks for outlier removal and continuous reconstruction of spectral, cumulative-oxygen and flue-gas signals, plus continuous 3D spectra for all ten heats. Question 2 asks for oxygen-consumption/gas relationships, total gas emissions and flow-dependent time delays. Question 3 asks for real-time gas prediction from converter-mouth spectra, validation on all ten attachments, and a globally combined or grouped model with good generalizability.

## B. What the literature supports

### SVR

- BOF flame-radiation endpoint regression has been studied with a phase classifier followed by epsilon-SVR (Shao et al., 2014).
- Wavelet-denoised weighted twin-SVR has been applied to BOF endpoint carbon and temperature prediction (Gao et al., 2019).
- Weighted nonparallel SVR with WOA has been applied to endpoint carbon and temperature prediction in a 260-ton BOF (Liu et al., 2021).

**Interpretation:** SVR-family models are reasonable classical nonlinear regression baselines. Phase-aware, weighted and optimizer-tuned variants are candidates for ablation, not automatic upgrades. The reviewed endpoint studies do not by themselves prove performance on continuous CO2/CO time-series targets.

### ELM

- Evolutionary membrane optimization has been applied to ELM for BOF endpoint prediction (Han & Liu, 2014).
- IPSO-RELM has been studied for BOF endpoint manganese prediction (Zhang et al., 2019).
- JITL with vMF-mixture sample selection and WELM has been studied for endpoint carbon and temperature prediction under changing operating conditions (Qi et al., 2021).

**Interpretation:** ELM offers a relatively compact shallow-regression baseline. Regularization, optimizer-based parameter selection and local sample weighting are literature-backed options to test. Their extra degrees of freedom can overfit if tuning is not confined to training heats.

### Deep neural networks and fusion

- A BOF flame-spectrum/gas study reports an autoencoder + Optuna + GWO-BiLSTM framework (Wu et al., 2026); this is the closest direct spectral-to-gas reference in this collection.
- A BOF asynchronous multi-source study uses explicit time offsets, multi-scale FPN features and Transformer self-attention (Hu et al., 2025), but its output is an operating-condition class, not continuous gas flow.
- A domain-knowledge spatio-temporal deep LSTM framework analyzes and aligns process-variable delays in mineral processing (Zhou et al., 2023); this is a cross-domain reference.
- A multimodal BOF meta-learning SAE paper addresses operating modes and data drift for endpoint carbon/temperature (Zeng et al., online 2025 / issue 2026).

**Interpretation:** The literature motivates a compact spectral encoder plus temporal model and explicit time alignment. It does not justify selecting a large neural network before establishing simpler baselines and heat-level generalization.

### Spectral preprocessing

Zhao et al. (2023) report BEADS baseline correction, GA-optimized rough-set feature reduction and a BP network for BOF endpoint temperature/carbon prediction. They report reducing 2048 channels to eight selected wavelength features. Their records were shuffled and split row-wise; this does not test generalization to an unseen heat and should not be copied as the main validation protocol.

## C. Proposed modeling plan for Problem B

### Stage 1 — inspect and reconstruct signals

1. Keep each heat as an independent unit with a stable heat identifier.
2. Plot raw spectra, oxygen consumption, CO2 and CO before cleaning.
3. Detect isolated spikes and suspicious segments separately. Do not delete physically plausible rapid transitions solely because they are statistically unusual.
4. Compare robust local filters / Hampel-style detection with domain-informed checks; record every replacement mask.
5. Reconstruct continuous curves with an explicitly chosen interpolant and preserve the original observations. Avoid using future samples in any preprocessing step intended to simulate real-time prediction.
6. Derive oxygen-lance flow from the cumulative-oxygen curve only after choosing a smoothing/reconstruction method; differentiation amplifies noise, so compare derivative methods and inspect units.

### Stage 2 — estimate the physical delay

Let O_h(t) be cumulative oxygen consumption for heat h, and G_h(t) be the measured flue-gas flow series. A candidate delay estimate is obtained by comparing physically motivated changes in oxygen-lance flow / oxygen consumption with later gas response, using a bounded cross-correlation or event-alignment procedure.

The contest explicitly states that higher oxygen-lance flow rates correspond to smaller gas-signal delays. Therefore:

- Do not assume one constant lag across the entire heat.
- Estimate delay within flow-rate regimes or with a smooth delay-versus-flow relationship.
- Report per-heat delay estimates and uncertainty/range, not only one pooled value.
- Check CO2 and CO separately; they need not have identical delay profiles.
- Avoid using the held-out heat's target gas series to tune a prediction model. Delay estimation used for model training must be based on training-fold data or a predeclared physical procedure that does not leak test labels.

A simple baseline is a discrete lag l in samples, where each sample is 0.5 s. Then physical delay is tau = 0.5 l seconds. This conversion follows from the contest sampling interval.

### Stage 3 — build comparable prediction baselines

Use the same training/test heat partitions and the same target alignment for all model families.

| Model | Candidate input | Candidate output | Role |
|---|---|---|---|
| RBF-SVR | Reduced spectral features at time t, optionally recent spectral window | CO2/CO at aligned future time t+lag | Classical nonlinear baseline |
| ELM / RELM | Same reduced features | Same aligned gas targets | Fast shallow model |
| IPSO-RELM / EMA-ELM | Same features | Same targets | Optimized ELM ablation |
| Autoencoder + BiLSTM | Spectral sequence/window | Gas sequence at flow-dependent lag | Deep temporal model |
| Compact encoder + attention | Spectral sequence and explicitly aligned time features | Gas sequence | Fusion/attention candidate |
| Phase-aware SVR/ELM | Features plus process-stage indicator or separate stage models | Gas targets | Test whether stage changes the mapping |

Do not compare models if they use different target shifts, different held-out heats or different leakage-prone preprocessing.

### Stage 4 — validate across heats

The primary generalization protocol should be **leave-one-heat-out cross-validation**:

1. Hold out one complete heat.
2. Fit all preprocessing, feature selection, scaling, dimensionality reduction, lag calibration, optimizer tuning and model parameters using only the other nine heats.
3. Predict the held-out heat without using its future gas labels for tuning.
4. Repeat for all ten heats.
5. Report each heat's metrics and the distribution across heats.

A second, clearly labeled analysis may compare leave-groups-of-heats-out or train-on-several-heats/test-on-the-rest. A random row-wise split can be included only as a non-primary comparison and must be labeled as potentially optimistic because neighboring time points are correlated.

### Stage 5 — metrics and reporting

For CO2 and CO separately, report at least:

- MAE
- RMSE
- R2, with a note if the held-out heat has near-constant target variance
- Per-heat metrics and pooled metrics
- Error by process stage and oxygen-flow regime
- Prediction latency / runtime if real-time feasibility is discussed

For total emissions, integrate the reconstructed instantaneous flow rate over time using the correct time step and units. Confirm the data's units and any conversion factors before reporting totals.

### Stage 6 — ablation and robustness

Recommended controlled comparisons:

1. Raw spectral channels vs. PCA/PLS vs. wavelength feature selection vs. autoencoder embedding.
2. Single global model vs. stage-specific model vs. heat-group model.
3. No lag correction vs. fixed lag vs. flow-dependent lag.
4. SVR vs. ELM vs. RELM vs. optimized RELM vs. BiLSTM.
5. Single-seed result vs. repeated-seed distribution for ELM and neural models.
6. With/without oxygen-flow and cumulative-oxygen features, only where the problem's prediction-time information makes them legitimately available.

## D. Key risks to call out in the paper

- **Small independent sample count:** there are ten heats. Thousands of rows within a heat do not make thousands of independent heats.
- **Temporal leakage:** adjacent rows are strongly related. Random row-wise train/test splitting can overstate generalization.
- **Preprocessing leakage:** fitting a scaler, feature selector, PCA/autoencoder or lag parameter on all ten heats before cross-validation leaks test-distribution information.
- **Future-information leakage:** centered smoothing, two-sided interpolation or sequence windows that include future spectra are not valid for a real-time claim unless explicitly framed as offline reconstruction.
- **Target ambiguity:** the statement calls the tabulated CO2/CO values instantaneous volume flow rates (m3/s), despite concentration-style column labels.
- **Target mismatch in literature:** many BOF papers predict endpoint carbon, manganese or temperature; these are methodological references, not direct performance evidence for continuous gas-flow forecasting.
- **Over-optimization:** metaheuristic search can overfit a tiny number of heats. Nested validation or a fixed, small search budget is preferable.
- **Distribution shift:** different heats may differ in operating mode and baseline intensity. Report per-heat errors and inspect domain shift.

## E. Questions for the modeling and programming teammates

1. Which outputs are required for Question 3: instantaneous CO2 and CO flow only, or also integrated emissions?
2. What real-time window is allowed: current spectrum only, past spectra, or a rolling sequence?
3. Does the prediction timestamp mean estimating the current bath state from a delayed gas measurement, or forecasting the analyzer's future readout? Write the time-index convention explicitly.
4. What physical constraints can be defended (non-negativity, smoothness, flow-dependent lag, stage transitions)?
5. Which preprocessing operations can be performed causally without seeing future observations?

## F. Bottom line

Start with signal reconstruction and explicit delay analysis. Establish an RBF-SVR baseline and a standard ELM/RELM baseline under leave-one-heat-out validation. Only then test phase-aware or optimized variants and a compact autoencoder–BiLSTM/attention model. The literature supports these as plausible candidates; the ten-heat dataset must decide which one generalizes.

## Source links

- Contest statement: user-provided 2026 APMCM Problem B.pdf.
- Literature records and publisher/DOI links: see [papers.json](papers.json) and [papers.md](papers.md).
- Last bibliography verification: 2026-11-26.
