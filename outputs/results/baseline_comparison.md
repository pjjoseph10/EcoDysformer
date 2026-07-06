# ETDD70 baseline comparison

> **Cross-paper numbers use different validation protocols and are approximate.** They are NOT directly comparable to this project's subject-level nested-CV results. The only strictly comparable numbers are among *This work* rows, which share one protocol.

| system                                          | accuracy   | accuracy_ci   | validation                              | note                                                       |
|:------------------------------------------------|:-----------|:--------------|:----------------------------------------|:-----------------------------------------------------------|
| ETDD70 dataset paper (Sedmidubsky et al., 2024) | 0.9        |               | cross-paper (protocol differs)          | approximate; different validation protocol                 |
| SwinV2 + SGA                                    | 0.9245     |               | cross-paper (protocol differs)          | approximate; different validation protocol                 |
| INSIGHT                                         | 0.8665     |               | cross-paper (protocol differs)          | approximate; different validation protocol                 |
| CatBoost / XGBoost                              | 0.82       |               | cross-paper (protocol differs)          | approximate range 0.80-0.83; different validation protocol |
| This work — (Performer / quadratic / blind)     | pending    |               | subject-level nested CV (this protocol) | run run_stage1 on Kaggle to populate                       |
