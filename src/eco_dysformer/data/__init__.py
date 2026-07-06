"""ETDD70 data layer: inspect-first loading, labels, and stimulus text.

The guiding rule here is *inspect before you parse*. The schema encoded in the
config was verified against the real files, and every loader re-asserts it, so a
drift in the on-disk layout fails loudly instead of producing quietly-wrong
features. Where structure is genuinely uncertain, the code carries a clear TODO
rather than a silent assumption.
"""
