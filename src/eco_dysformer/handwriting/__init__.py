"""Stage-2 handwriting auxiliary branch (RQ3).

The handwriting signal comes from a DISJOINT cohort (different children than
ETDD70's gaze cohort), so it is treated as an explicitly auxiliary modality. This
subpackage is built inspect-first: before any modeling, `inspect_handwriting`
determines whether the dataset carries subject-level (per-writer) linkage, which
decides the whole RQ3 design:

    linkage present -> a real per-writer reversal-rate feature
    linkage absent  -> an aggregated reversal-rate PROXY, reported as a proxy and
                       NEVER as subject-level diagnosis

Nothing here claims the handwriting cohort is paired with ETDD70's children.
"""
