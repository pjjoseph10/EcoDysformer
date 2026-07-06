"""Evaluation harness: subject-level nested CV, paired statistics, metrics.

The single most important correctness requirement lives here: a child's three
passages must NEVER be split across train and test. :mod:`.cv` enforces and
asserts subject-level grouping; every comparison claim is backed by a paired
Wilcoxon signed-rank test across outer folds plus bootstrap CIs (:mod:`.stats`),
never a single-split point estimate.
"""
