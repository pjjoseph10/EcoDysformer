"""Pre-fusion explainability (RQ4 core path).

LIME local explanations and attention-weight extraction are computed on the
ORIGINAL interpretable features (fixation count, regression ratio, syntactic
depth, ...) -- NEVER on PCA components. Attribution stability across outer folds
is reported so explanations are judged on consistency, not a single lucky split.

This is an offline methodological study: explanations are assessed as
interpretability face-validity against literature-documented biomarkers, never as
a clinical or diagnostic claim.
"""
