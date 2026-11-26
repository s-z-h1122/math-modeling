# BOF Literature Knowledge Base: SVR, ELM and Deep Neural Networks

This folder collects research relevant to **2026 APMCM Problem B: Dual-modal Temporal Information Fusion Modeling for Converter Steelmaking Process**.

## Files

- [papers.json](papers.json) — machine-readable index of 18 verified bibliographic records, model summaries, evidence levels and source links.
- [papers.md](papers.md) — human-readable paper cards and comparison table.
- [synthesis.md](synthesis.md) — problem-specific synthesis, candidate experiments and validation cautions.
- [open-access-reading-list.md](open-access-reading-list.md) — verified free full-text papers and detailed reading notes.

## Scope

The collection focuses on:

1. **Support vector regression (SVR):** flame-spectrum regression, phase-aware SVC + SVR, twin-SVR variants, weighted/nonparallel SVR and WOA tuning.
2. **Extreme learning machine (ELM):** evolutionary optimization of ELM, regularized ELM with improved PSO, and just-in-time weighted ELM.
3. **Deep neural networks:** autoencoder + BiLSTM, stacked autoencoders, meta-learning SAE, FPN + Transformer self-attention, and domain-knowledge spatio-temporal LSTM.
4. **Supporting BOF studies:** spectral baseline correction and feature selection, flame-video temporal feature extraction, and BOF machine-learning reviews.

## Evidence and reading discipline

Each paper has an `evidence_level`:

- `full_text`: the publisher page or open full text was accessible and inspected.
- `abstract_only`: the publisher abstract/preview and bibliographic metadata were accessible, but the full article was not inspected.
- `metadata_only`: only bibliographic metadata and a short abstract fragment were accessible.

Do not treat an abstract-only record as a full replication recipe. Where an input variable, data split, optimizer setting or unit is not stated in the accessible source, it is explicitly marked as unknown/not verified. Metrics belong to each paper's own data and target definitions; they are not directly comparable across studies.

## Problem-specific cautions

- Problem B supplies **10 heats**, not 10 independent samples. Each heat contains roughly 1400 time rows and 2047 wavelength channels at 0.5-second intervals.
- The problem text says the CO2 and CO values are **instantaneous volume flow rates (m3/s)**, even though the table headings use concentration wording. Preserve this distinction when writing the paper.
- The gas analyzer observes delayed flue-gas signals. The delay varies with oxygen-lance flow rate; a single constant lag may be inadequate.
- Randomly splitting adjacent time rows can leak near-duplicate temporal information between train and test sets. Use heat-level validation (for example, leave-one-heat-out) as the primary generalization check.
- The papers' endpoint carbon/temperature studies are relevant methodological references, but endpoint targets are not the same as the competition's continuous CO2/CO time-series targets.

## Suggested reading order

1. Ghalati et al. (2023), BOF machine-learning review — map the field.
2. Wu et al. (2026), flame spectrum + autoencoder + GWO-BiLSTM — closest match to spectrum/gas modeling.
3. Hu et al. (2025), asynchronous multi-source FPN + self-attention — study time alignment and fusion (classification task).
4. Qi et al. (2021), JITL + vMF + WELM — study operating-mode shift and local modeling.
5. Shao et al. (2014), SVC + SVR — study phase-aware spectral regression.
6. Han & Liu (2014) and Zhang et al. (2019) — compare ELM optimization strategies.
7. Zeng et al. (2025 online / 2026 issue), meta-learning SAE — study cross-mode adaptation.
8. Zhao et al. (2023), spectral feature extraction — study baseline correction and wavelength reduction.

## Citation practice

Cite the original publisher/DOI page in any report. Do not copy long passages from papers. Distinguish:

- **Source-derived facts:** directly reported by the paper or the contest statement.
- **Research synthesis:** conclusions drawn by comparing multiple papers.
- **Proposed experiments:** ideas to test on the competition data, not established results.

Last bibliographic verification: **2026-11-26**. Four additional free full-text papers were added; the 2019 IPSO-RELM record was upgraded from abstract-only to full-text after inspecting the publisher PDF.
