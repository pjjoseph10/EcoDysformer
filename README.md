# Eco-Dysformer v2 — Stage 1 (CORE)

A small, honest, reproducible study for dyslexia screening from eye-tracking on
the **ETDD70** dataset (70 Czech children, 35 dyslexic / 35 typical, each reading
three passages: a **syllable matrix → narrative → pseudo-text** complexity
gradient). Stage 1 builds the CORE contribution only:

- **RQ2** — paired gaze × linguistic-complexity co-conditioning, using the genuine
  within-subject pairing (every child reads all three passages).
- **RQ1** — a Performer (linear-attention) encoder vs a **parameter-matched**
  quadratic (softmax) baseline on the *same* engineered features, plus an
  empirical sequence-length crossover on the raw event stream.
- **RQ4 (core path)** — pre-fusion explainability (LIME + attention weights) on
  **original interpretable features, never on PCA components**.

This is an **offline methodological study on a public research dataset**. It makes
**no clinical or diagnostic claim**. It is a rigor/methods project, not an
accuracy race — the target is the published ~90% range, reported honestly.

> Stage-2 items (handwriting branch, naive-vs-honest fusion / RQ3, OOD cohorts)
> are intentionally **not** built here. The code is structured so they slot in
> cleanly later. See `PlanC_v2_Proposal.md` for the full plan.

---

## Repository layout

```
configs/stage1.yaml         Single source of truth: paths, hyperparameters, seed, flags
requirements.txt            Pinned dependencies (Kaggle-tested reference env)
data_card.md  model_card.md Dataset & model documentation
src/eco_dysformer/
  config.py  seed.py        Config load/validate; deterministic global seeding
  data/                     inspect_dataset · loader · labels · stimuli · tensors
  features/                 gaze · linguistic · embeddings · assemble
  models/                   attention · blocks · gaze/linguistic encoders · fusion · heads · pipeline · build
  eval/                     cv · stats · metrics · operational · rq1_crossover · rq2_effects · run_cv · baselines_table
  explain/                  lime_explain · attention_extract
  run_stage1.py             Entry point: full Stage-1 result set + baseline table
tests/                      data layer · gaze features · no-leakage CV · stats/metrics · tensors · model smoke
dataset/ETDD70/             The dataset (attach on Kaggle; see below)
outputs/                    All metrics/tables (JSON/CSV) and figures are written here
```

### Design invariants (enforced in code, not just prose)
- **Subject-level folds.** A child's three passages never split across train/test.
  `eval/cv.py` builds subject-grouped, class-stratified nested folds and
  `assert_no_subject_leakage` re-checks every split. The core model unit is a
  child (one example = a 3-passage sequence), so folds are leak-free by
  construction *and* asserted.
- **No fabricated numbers.** Steps that need the real trained model produce output
  on Kaggle; where a dependency is missing they raise a clear error, never a
  placeholder metric.
- **Gaze features are recomputed** from the raw fixation/saccade events and
  **cross-checked** against the dataset's own `*_metrics.csv` (1260/1260 strict
  comparisons pass within 2%). The dataset's AOI-based regression count is carried
  as an explicit alternative column; the definitional gap is reported, not hidden.
- **Explainability on original features only.** `explain.on_features` must be
  `original`; asserted at config load.

---

## Running on Kaggle (the documented GPU target: P100 / T4)

1. **Create a notebook** and **attach the ETDD70 dataset** (Zenodo
   `10.5281/zenodo.13332134`) so it appears under `/kaggle/input/…`.
2. **Enable GPU** and **enable Internet** (needed to `pip install` and to download
   the Czech UDPipe/Stanza model, the Zipf table, and — if enabled — XLM-R).
3. **Point the config at the attached data.** Either edit `configs/stage1.yaml`
   `paths.*` to the `/kaggle/input/...` location, or symlink it to
   `dataset/ETDD70`. Confirm the four `paths` resolve with the inspector (step 5).
4. **Install deps:** `pip install -r requirements.txt`
   (torch / numpy / pandas / scikit-learn / lightgbm are preinstalled on Kaggle;
   this pins the rest).
5. **Inspect first** (prints the tree, sample schemas, reconstructed Czech text,
   label balance; writes `outputs/results/dataset_inspection.json`):
   ```bash
   PYTHONPATH=src python -m eco_dysformer.data.inspect_dataset
   ```
6. **Run all of Stage 1:**
   ```bash
   PYTHONPATH=src python -m eco_dysformer.run_stage1
   ```
   Produces (under `outputs/`): recomputed gaze features + cross-check, Czech
   linguistic features, RQ2 effect sizes, subject-level nested-CV results for all
   arms (`cv_results.json`), the RQ1 crossover curve + figure, LIME stability +
   fusion attention, and the published-baseline comparison table.

**Czech NLP engine:** `features.linguistic.engine` defaults to `udpipe`. Provide a
Czech model at `features.linguistic.udpipe_model` (e.g. `czech-pdt-ud-2.5`), or
set `engine: stanza` to auto-download the Czech Stanza model on first use. The
supplementary XLM-R / RobeCzech embedding is **off by default**
(`use_embedding: false`).

---

## Local development (bare Python, no GPU)

The data / feature / statistics layers run in a plain Python env
(numpy · pandas · scipy · pyyaml). Neural + LightGBM + Czech-NLP steps are
Kaggle-first.

```bash
# gaze features + cross-check + RQ2 effect sizes + baseline stub, no torch/NLP:
PYTHONPATH=src python -m eco_dysformer.run_stage1 --local

# tests (no pytest needed — each suite has a plain runner):
PYTHONPATH=src:tests python tests/test_cv_no_leakage.py
# or, with pytest installed:  PYTHONPATH=src pytest -q
```

`--local` runs inspect → gaze features → RQ2 effects → baseline table and skips
the model/crossover/explain stages with a clear note (no fabricated numbers).

---

## Reproducibility
- One global `seed` in the config seeds Python / NumPy / PyTorch / LightGBM and
  sets deterministic flags (`src/eco_dysformer/seed.py`).
- Every metric and table is written to `outputs/` as JSON/CSV; figures are saved.
  Nothing important lives only in stdout.
- Dependencies are pinned in `requirements.txt`.
