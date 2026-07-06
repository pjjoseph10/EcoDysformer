"""Eco-Dysformer v2 -- Stage 1 (CORE).

A small, honest, reproducible study for dyslexia screening from eye-tracking on
the ETDD70 dataset. This package contains the CORE-tier pipeline only:

    data  ->  features  ->  models (RQ1/RQ2)  ->  eval  ->  explainability

Stage-2 items (handwriting branch, naive-vs-honest fusion / RQ3, OOD cohorts)
are intentionally NOT implemented; the package is structured so they slot in
cleanly later.

This is an offline methodological study on a public research dataset. Nothing in
this package makes, or should be read as making, a clinical or diagnostic claim.
"""

__version__ = "0.1.0-stage1"
