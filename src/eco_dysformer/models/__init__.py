"""Model layer: Performer/quadratic attention, encoders, paired fusion, head.

CORE path (RQ2): Gaze encoder + Linguistic-complexity encoder + learned paired
cross-attention fusion over the child's three passage tokens -> per-child
representation -> LightGBM head.

RQ1 comparison arm: the SAME architecture with quadratic (softmax) attention
swapped in for the Performer (kernelized) attention, parameter-matched by
construction so the comparison isolates attention complexity, not capacity.

All modules require PyTorch and are Kaggle-first; they are written to also run on
CPU for the tiny ETDD70 sizes.
"""
