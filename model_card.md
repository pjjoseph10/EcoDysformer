# Model Card — Eco-Dysformer v2, Stage 1 (CORE)

## Overview
A lightweight, linear-complexity model for dyslexia **screening research** on
ETDD70 eye-tracking data. The core path fuses each child's **paired** gaze and
linguistic-complexity signals across the syllable → narrative → pseudo-text
gradient. **This is an offline methodological study; it is NOT a clinical or
diagnostic system and must not be used as one.**

## Architecture
Per child = one example = a length-3 sequence of passage tokens (ordered by
complexity).

- **Gaze encoder** — projects the per-passage engineered gaze feature vector,
  adds a learned passage/complexity position code, and applies self-attention
  over the 3 passage tokens. Attention is swappable: **Performer** (FAVOR+
  kernelized, O(N)) or **quadratic** (softmax, O(N²)). A separate `events` input
  mode encodes the raw fixation/saccade stream for the RQ1 crossover study.
- **Linguistic-complexity encoder** — projects the per-passage linguistic feature
  vector (dependency depth, lexical density, Zipf; optional supplementary
  embedding) into the same space; the RQ2 conditioning signal.
- **Paired cross-attention fusion** — the gaze tokens attend to the linguistic
  tokens for the *same* child/passages (genuinely paired), pooled to a per-child
  vector.
- **Heads** — a small auxiliary MLP (BCE) trains the encoders end-to-end; the
  **primary classifier is LightGBM**, fit on the frozen per-child embeddings.

**Arms (Stage 1):**
| Arm | Attention | Conditioning | Question |
|-----|-----------|--------------|----------|
| `performer_conditioned` | Performer | gaze × complexity | RQ1 core / RQ2 core |
| `quadratic_conditioned` | quadratic | gaze × complexity | RQ1 baseline (param-matched) |
| `performer_blind` | Performer | gaze only | RQ2 contrast (complexity-blind) |

The Performer and quadratic arms are **parameter-matched by construction**
(identical learned projections; the Performer's random-feature matrix is a fixed
non-trainable buffer) and `assert_param_matched` verifies it, so RQ1 isolates
attention complexity, not capacity.

## Training
- Full-batch Adam on the auxiliary BCE head; global-seeded; deterministic flags.
- Fold-safe standardization (fit on train only); Gaussian feature jitter as
  train-time augmentation (time-warp/segment-permutation apply to the raw-event
  regime).
- LightGBM head with pinned seeds; downstream on frozen embeddings.

## Evaluation protocol
- **Subject-level nested cross-validation** — outer loop = performance estimate,
  inner loop = LightGBM hyperparameter selection (`num_leaves` grid on frozen
  embeddings). A child's three passages never split across train/test; asserted.
- **Metrics** — accuracy, F1, AUROC, plus calibration (Brier, ECE). Operational:
  parameter count, peak GPU memory, training time/epoch, inference latency.
- **Statistics** — every "matches or exceeds" claim uses a **paired Wilcoxon
  signed-rank test across outer folds** plus **bootstrap CIs**; no single-split
  point estimates. RQ2 gaze-shift magnitude reported as **Cohen's d** across the
  complexity gradient (dyslexic vs typical).
- **RQ1 crossover** — empirical Performer-vs-quadratic forward-time/memory sweep
  over sequence length on the raw event stream; the crossover length is reported
  (an honest "no win on short sequences" is expected and acceptable).
- **Baselines** — a comparison table vs published ETDD70 numbers (dataset paper
  ~90%; SwinV2+SGA 92.45%; INSIGHT 86.65%; CatBoost/XGBoost ~80–83%), clearly
  marked as **cross-paper, different-validation, approximate**, with a separate
  column for this project's own protocol.

## Explainability (RQ4 core path)
LIME local explanations and cross-attention weights are computed on the
**original interpretable features** (fixation count, regression ratio, syntactic
depth, …), **never on PCA components**. Attribution **stability across outer
folds** is reported (top-k Jaccard, Spearman), with a biomarker face-validity
check. These support interpretability face-validity **only** — not clinical
subtype claims.

## Intended use & limitations
- **Intended use:** reproducible ML methods research on a public dataset;
  efficiency ("Eco") reporting; interpretability analysis.
- **Out of scope:** any clinical, diagnostic, or screening deployment;
  individual-level decisions about a child.
- **Limitations:** n = 70, single language/protocol; results are a first rigorous
  pass; generalization beyond ETDD70 is unverified (OOD is Stage 2). Linguistic
  features are passage-level constants (conditioning signal only). Small-N neural
  training is kept deliberately tiny and is prone to variance — hence the paired
  tests and CIs rather than point estimates.
