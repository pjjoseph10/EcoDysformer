# Eco-Dysformer v2 — Stage 1 Results

_Auto-generated from the run artifacts; all numbers pulled from disk. Offline methodological study on ETDD70 — no clinical or diagnostic claim._

## Per-arm performance (subject-level nested CV)

Outer folds: **10**, inner folds: 3; subject-level, leakage-checked: **True**.

| Arm | Accuracy [95% CI] | F1 | AUROC | ECE | Brier |
|---|---|---|---|---|---|
| performer_conditioned | 0.829 [0.754, 0.896] | 0.805 | 0.861 | 0.170 | 0.169 |
| quadratic_conditioned | 0.821 [0.742, 0.900] | 0.807 | 0.883 | 0.178 | 0.176 |
| performer_blind | 0.833 [0.742, 0.917] | 0.807 | 0.878 | 0.166 | 0.164 |

## RQ1 — Linear vs quadratic attention

**Accuracy — matched.** Performer 0.829 vs quadratic 0.821; paired Wilcoxon p = 0.786, paired-difference 95% CI [-0.088, 0.092] (straddles zero → no significant difference).

**Efficiency — crossover at N ≈ 1024.** Below it the Performer's constant overhead makes it slower; above it linear attention wins on both time and memory. ETDD70's engineered event streams (~130–350 events/passage) sit below the crossover, so quadratic is pragmatic for that representation; the raw 250 Hz stream (~18k samples/passage) sits far above it.
Both arms are parameter-matched (performer 141121 vs quadratic 141121 params).

|   seq_len |   performer_ms |   quadratic_ms | performer_faster   |   performer_peak_mb |   quadratic_peak_mb | device   |
|----------:|---------------:|---------------:|:-------------------|--------------------:|--------------------:|:---------|
|         3 |        2.58835 |        1.49067 | False              |             393.868 |             393.998 | cuda     |
|         8 |        2.57291 |        1.55711 | False              |             394.037 |             394.037 | cuda     |
|        16 |        2.55442 |        1.50313 | False              |             394.099 |             394.099 | cuda     |
|        32 |        2.52609 |        1.55703 | False              |             394.224 |             394.224 | cuda     |
|        64 |        2.612   |        1.50113 | False              |             394.474 |             394.474 | cuda     |
|       128 |        2.50617 |        1.50181 | False              |             395.732 |             395.849 | cuda     |
|       256 |        2.51586 |        1.50518 | False              |             398.49  |             402.724 | cuda     |
|       512 |        2.54936 |        1.5057  | False              |             404.006 |             428.474 | cuda     |
|      1024 |        2.53804 |        2.91164 | True               |             415.037 |             527.974 | cuda     |
|      2048 |        3.27227 |        9.17805 | True               |             437.1   |             918.974 | cuda     |
|      4096 |        5.30468 |       34.5561  | True               |             481.225 |            2468.97  | cuda     |

## RQ2 — Paired gaze × linguistic co-conditioning

**Classification — honest null.** Complexity-conditioned 0.829 vs complexity-blind 0.833; Wilcoxon p = 0.832, diff CI [-0.075, 0.067]. Conditioning does **not** improve classification — expected, since the linguistic features are passage-level constants and the gaze features already carry the signal.

**Descriptive interaction — the positive RQ2 evidence.** Per-subject gaze-feature slope across the syllable→narrative→pseudo gradient, dyslexic vs typical (Mann–Whitney, BH-FDR):

| Feature | d (slope) | p | q (FDR) |
|---|---|---|---|
| regression_ratio | 0.22 | 0.6896 | 0.7259 |
| fix_count | 1.05 | 0.0003 | 0.0010 |
| mean_fix_dur | 0.34 | 0.0213 | 0.0388 |
| total_read_time_ms | 1.44 | 0.0000 | 0.0000 |

Significant group×complexity interaction (q<0.05) for: **fix_count, mean_fix_dur, total_read_time_ms** — the dyslexic disadvantage in reading effort widens as text gets harder.

## RQ4 — Pre-fusion explainability (original features, never PCA)

Top pooled LIME features: ['syllables__median_fix_dur', 'syllables__mean_sacc_dur', 'pseudo__median_fix_dur', 'syllables__mean_fix_disp', 'pseudo__return_sweep_count']. All oculomotor (biomarker face-validity = 1.00). Attribution stability across folds: mean top-5 Jaccard 0.27 (exact set shuffles), mean Spearman 0.82 (broad ranking stable).

## Published-baseline comparison

> Cross-paper numbers use different validation and are approximate; only the *This work* rows share one protocol.

| system                                          |   accuracy | accuracy_ci    | validation                              | note                                                       |
|:------------------------------------------------|-----------:|:---------------|:----------------------------------------|:-----------------------------------------------------------|
| ETDD70 dataset paper (Sedmidubsky et al., 2024) |     0.9    | nan            | cross-paper (protocol differs)          | approximate; different validation protocol                 |
| SwinV2 + SGA                                    |     0.9245 | nan            | cross-paper (protocol differs)          | approximate; different validation protocol                 |
| INSIGHT                                         |     0.8665 | nan            | cross-paper (protocol differs)          | approximate; different validation protocol                 |
| CatBoost / XGBoost                              |     0.82   | nan            | cross-paper (protocol differs)          | approximate range 0.80-0.83; different validation protocol |
| This work — performer_conditioned               |     0.8292 | [0.754, 0.896] | subject-level nested CV (this protocol) | our protocol; directly comparable across our own arms      |
| This work — quadratic_conditioned               |     0.8208 | [0.742, 0.900] | subject-level nested CV (this protocol) | our protocol; directly comparable across our own arms      |
| This work — performer_blind                     |     0.8333 | [0.742, 0.917] | subject-level nested CV (this protocol) | our protocol; directly comparable across our own arms      |

## Honest limitations

- n=70, single language/protocol; generalization beyond ETDD70 unverified (OOD is Stage 2).
- RQ2's positive result is descriptive (effect sizes), not a classification gain; conditioning did not help the classifier.
- Calibration is mediocre (see ECE); models are somewhat overconfident.
- Recomputed geometric regression over-counts vs the dataset's AOI-based count (both reported); see data_card.
