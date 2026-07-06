"""Feature engineering: gaze (per child per passage) and linguistic (per passage).

Gaze features are RECOMPUTED from the raw fixation/saccade event files (the user
decision), with a cross-check against the dataset-provided ``*_metrics.csv``
trial aggregates so any definitional drift is visible and reported rather than
hidden. Linguistic features are computed per passage from the reconstructed
Czech text with a Czech-appropriate pipeline (UDPipe/Stanza), plus Zipf
frequency; a neural embedding is available as a supplementary signal behind a
config flag.
"""
