# V-TWIN: a source-aware, confound-controlled evaluation framework for clinical machine-learning classifiers

Reference implementation for the study:

> **Speech-based detection of lung cancer is dominated by demographic and acquisition confounding: a source-aware evaluation framework and a multi-center placebo demonstration.**
> Ankışhan H. *et al.* (under review, *Journal of Biomedical Informatics*).

This repository contains the analysis code for **V-TWIN**, a reusable framework for
*confound-controlled* evaluation of clinical machine-learning classifiers. It was developed and
demonstrated for **speech-based lung-cancer (LC) detection**, but its components apply to any
classifier trained on multi-source clinical data.

The framework asks a single question: **how much of an apparently strong classifier reflects the
target disease, and how much reflects demographic, acquisition, site and collection confounding?**
Applied to speech and LC, it shows that the apparent signal is **confound-dominated** and does not
generalise across cohorts.

---

## What the framework provides

- **Confound-only negative controls (M0).** Predict the label from acquisition/quality descriptors
  alone, from age alone, and predict site/collection from acoustic features alone.
- **Feature-subset localisation.** Separate a physiological subset from a spectral/acquisition subset
  to localise where discrimination comes from.
- **Source- and collection-leakage quantification.** Measure how strongly the feature space encodes
  cohort/device/collection identity.
- **Pre-registered bidirectional cross-cohort transfer**, with a **post-adjustment leakage
  diagnostic** that flags "improved" transfer that is really re-encoded acquisition structure.
- **Unsupervised domain adaptation** (within-fold residualisation, rank-normalisation, CMVN-style
  alignment, CORAL).
- **Multi-center placebo control.** Separate disease-free groups across centers, and substitute
  controls across centers, to expose how apparent detection is inflated.
- **Guarded ensemble (M5)** with probability calibration, an acquisition/source discriminator,
  a conformal uncertainty layer and a **selective-prediction abstention rule**.

A single, fixed, pre-specified model is used throughout (`StandardScaler` + `LogisticRegression(C=1.0)`)
to avoid tuning toward a desired result. All evaluation is **subject-level**.

---

## Repository layout

```
da_common.py                  Core utilities: cohort loading, the FIXED model, subject-level CV,
                              cross-cohort transfer, source-leakage and CORAL primitives, feature subsets.
ibni_within_cohort_audit.py   Within-cohort, confound-controlled primary analysis (M0/M1/M4 + age matching).
ibni_cross_cohort.py          Bidirectional cross-cohort transfer + post-adjustment leakage diagnostic.
coral_transfer.py             Unsupervised CORAL domain adaptation with leakage recomputed in the aligned space.
paired_contrast_transfer.py   Paired/subgroup contrasts across cohorts.
placebo_site_confound.py      Multi-center placebo: sham (disease-free) contrasts + honest-vs-naive control swap.
m5_guarded_ensemble.py        Guarded ensemble with conformal abstention (risk-coverage, selective accuracy).
make_figures.py               Publication figures (300 dpi PNG).
_selftest.py                  Lightweight self-test of the shared primitives.

data_prep/                    Provenance scripts used to build the feature CSVs (environment-specific; reference only).
data/                         Place your own feature CSVs here (see data/README.md). No data are distributed.
results/                      Aggregate metric outputs (AUROC tables, risk-coverage, leakage) — no personal data.
figures/                      Generated publication figures.
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt
```

Developed with Python 3.9 and scikit-learn 1.6.1.

---

## Data

**No data are distributed with this repository.** The study used de-identified, feature-level data;
no raw audio, names, free-text, pathology reports or imaging are shared. De-identified, feature-level
data may be available from the corresponding author on reasonable request, subject to institutional
and ethical approvals.

To run the framework on your own data, supply a subject-level CSV per cohort with the **45-dimensional
Voice45** feature columns plus metadata. The expected schema is documented in
[`data/README.md`](data/README.md). The fixed 45-D layout is:

```
[0-12]  MFCC 0-12
[13-25] delta-MFCC
[26-37] spectral / RMS / ZCR / energy descriptors (mean + std)  -> device/channel sensitive
[38-39] F0 mean, F0 std
[40] jitter  [41] shimmer  [42] HNR  [43] voiced fraction  [44] breathiness   -> physiological subset
```

> Cohort and center identifiers in the code (e.g. site labels) are **dataset-specific** and correspond
> to the anonymised Cohort A / B / C used in the manuscript. Adapt them to your own data.

---

## Reproducing the analyses

Run from the repository root (scripts import `da_common`). Edit the data paths at the top of each
script to point to your CSVs, then:

```bash
python ibni_within_cohort_audit.py     # negative controls, feature subsets, age-matched sensitivity
python ibni_cross_cohort.py            # bidirectional transfer + post-adjustment leakage diagnostic
python coral_transfer.py               # CORAL domain adaptation (leakage recomputed in aligned space)
python placebo_site_confound.py        # multi-center placebo (sham + control-swap) demonstration
python m5_guarded_ensemble.py          # guarded ensemble: abstention, risk-coverage, selective accuracy
python make_figures.py                 # regenerate figures into figures/
python _selftest.py                    # sanity-check the shared primitives
```

Each analysis writes its metric tables to `results/`.

---

## Methodological principles (anti-HARKing)

1. The model is **fixed and pre-specified**; it is never re-tuned to obtain above-chance transfer.
2. **Target labels are never touched during training** in transfer/adaptation.
3. All standardisation and confound-adjustment steps are **fitted within training folds only**.
4. Cross-cohort and adaptation analyses are **run once** with pre-specified settings, and **all
   outcomes are reported irrespective of direction**.
5. Feature subsets are defined **by position** in the 45-D layout, identically across scripts.

---

## Citation

If you use this code, please cite the paper (see [`CITATION.cff`](CITATION.cff)). A formal reference
will be added once the article is published.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
