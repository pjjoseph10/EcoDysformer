# Data Card — ETDD70 (as used in Eco-Dysformer v2, Stage 1)

## Dataset
- **Name:** ETDD70 — Eye-Tracking Dyslexia Dataset.
- **Source / citation:** Dostalova, N., Svaricek, R., Sedmidubsky, J., Culemann,
  W., Sasinka, C., Zezula, P., & Cenek, J. (2024). *ETDD70: Eye-tracking dyslexia
  dataset* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13332134
  Associated paper: Sedmidubsky, J., Dostalova, N., Svaricek, R., & Culemann, W.
  (2024). *ETDD70: Eye-tracking dataset for classification of dyslexia using
  AI-based methods.* SISAP 2024, Springer.
- **Population:** 70 Czech-speaking children aged 9–10; **35 dyslexic / 35
  non-dyslexic** (clinician-diagnosed). Collected with informed consent under
  ethical guidelines for research with minors.
- **Access:** public research dataset; no institutional data access needed to
  replicate this project.

## What each child contributes (verified on disk)
Every child reads **three passages** — the project's central asset, a genuine
within-subject linguistic-complexity manipulation:

| Task code | Passage (name) | complexity_rank | Stimulus | Content in `content` column |
|-----------|----------------|-----------------|----------|-----------------------------|
| `T1_Syllables` | syllables | 0 | `s7_stimuli_t1.jpg` | syllable matrix (e.g. `ma si mu …`) |
| `T4_Meaningful_Text` | meaningful | 1 | `s7_stimuli_t4.jpg` | real Czech narrative (`Malý Pepík …`) |
| `T5_Pseudo_Text` | pseudo | 2 | `s7_stimuli_t5.jpg` | legal nonwords (`Vůmice lobu …`) |

There are only **3 distinct stimuli**, shared across all children.

### Files per (child, passage) — 70 × 3 × 4 = 840 total
- `*_raw.csv` — 250 Hz gaze: `time, gaze_x/y_left, gaze_x/y_right, pupil_left/right, stimfile, subject_id`.
- `*_fixations.csv` — I2MC fixations (40 ms min): `start_ms, end_ms, duration_ms, fix_x/y, disp_x/y, aoi_subline, aoi_line, …`.
- `*_saccades.csv` — saccades: `duration_ms, ampl_x, ampl_y, ampl, avg_vel, peak_vel, start_x, end_x, start_y, end_y, …`.
- `*_metrics.csv` — trial-level aggregates (`n_fix_trial, mean_fix_dur_trial, n_regress_trial, ratio_progress_regress_trial, …`) + per-ROI dwell/revisit stats and the token `content`.
- `dyslexia_class_label.csv` — `subject_id, class_id (0/1), label`.
- `rois.zip`, `stimuli.zip` — ROI word/line boundaries (+ token text) and stimulus images.

## Features derived in this project
**Gaze (per child, per passage)** — recomputed from the event files, cross-checked
against `*_metrics.csv` (**1260/1260 strict comparisons within 2%**): fixation
count, mean/median/std fixation duration, total & summed fixation time, fixation
rate, dispersion; saccade count, amplitude mean/std, duration, peak/avg velocity;
regression ratio and within/between-line/return-sweep counts.

> **Regression definition (documented choice).** `ampl_x`/`ampl_y` are stored as
> *magnitudes*, so signed direction is derived from the coordinate columns. Our
> transparent geometric regression **over-counts** relative to the dataset's
> published AOI-based count (within-line only); both are provided
> (`regression_*` vs `regression_*_dataset`) and the gap is reported, not hidden.

**Linguistic (per passage)** — from the reconstructed Czech text via a
Czech-appropriate pipeline (UDPipe/Stanza): mean parse-tree depth, mean dependency
distance, lexical density; Zipf frequency (mean/std/min) via `wordfreq`; plus
surface statistics. Because there are only 3 stimuli, these are **passage-level
constants** (identical across children) — their role is the RQ2 *conditioning*
signal, not per-child discrimination. An optional XLM-R/RobeCzech embedding is
supplementary and off by default.

## Labels
Binary, subject-level: `class_id` 0 = non-dyslexic, 1 = dyslexic. Balanced 35/35.

## Known limitations & appropriate use
- Single language (Czech), single protocol, **n = 70** — generalization beyond
  ETDD70 is unverified (OOD validation is a Stage-2 item, not built here).
- Diagnosis labels are the dataset's clinician labels; this project treats
  classification as a **methodological** exercise and makes **no clinical or
  diagnostic claim**.
- Linguistic features vary only across 3 passages; any linguistic "signal" is a
  conditioning effect, not independent per-child evidence.
- The dataset's exact I2MC/AOI regression algorithm is not fully reproducible from
  the released event files; see the regression note above.
