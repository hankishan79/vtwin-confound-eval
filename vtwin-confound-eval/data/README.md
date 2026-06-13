# Data

**No data are distributed with this repository.** The study used de-identified, feature-level data
only (no raw audio, names, free-text, pathology reports or imaging). De-identified, feature-level data
may be available from the corresponding author on reasonable request, subject to institutional and
ethical approvals.

## Expected input schema (per cohort)

A subject-level CSV with one row per subject (or per recording, aggregated to subject level), containing:

### Acoustic features — Voice45 (45 columns, fixed order)

| Index | Columns | Meaning |
|-------|---------|---------|
| 0–12  | MFCC 0–12 | mel-frequency cepstral coefficients |
| 13–25 | delta-MFCC | first differences |
| 26–37 | spectral / RMS / ZCR / energy (mean + std) | device/channel-sensitive descriptors |
| 38–39 | F0 mean, F0 std | fundamental frequency |
| 40–44 | jitter, shimmer, HNR, voiced fraction, breathiness | physiological subset |

Cepstral mean–variance normalisation (CMVN) is **not** applied at extraction, so that spectral
device/channel signatures are retained and can be *measured* (this is required for the leakage
analyses).

### Metadata / candidate confounders

- `label` — clinical diagnosis (the authoritative reference standard; e.g. LC / COPD / non-LC).
- `AGE` — participant age (used as a negative control and for age matching).
- Quality-control / acquisition (QC) descriptors — recording duration, estimated SNR, overall and
  speech loudness (dBFS), speech-frame fraction, number of selected segments, quality-pass flag.
- `site` / collection identifier — center or collection of origin (audited as a confounder).
- `sex` — optional, for matched sensitivity analyses.

Place your CSV(s) here and set the paths at the top of each analysis script. Files in this folder are
git-ignored by default so that data are never committed.
