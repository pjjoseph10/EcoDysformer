"""Supplementary neural sentence embedding for a passage (OFF by default).

XLM-R (multilingual) or RobeCzech (Czech-specific) mean-pooled embedding of the
passage text, used ONLY as a supplementary semantic-complexity signal behind
``config.features.linguistic.use_embedding``. It never replaces the named,
directly-computable linguistic metrics (dependency depth, lexical density, Zipf).

Kaggle-first: requires ``transformers`` + ``torch`` and Internet for the model
download; raises a clear ImportError otherwise. Like the passage-level linguistic
features, the embedding is identical across children (3 distinct values).
"""
from __future__ import annotations


def embedding_feature_names(dim: int) -> list[str]:
    return [f"ling_emb_{i:03d}" for i in range(dim)]


def extract_embedding(text: str, cfg) -> dict[str, float]:
    """Return a dict of mean-pooled embedding features for ``text``.

    Only called when ``config.features.linguistic.use_embedding`` is true.
    """
    lc = cfg.features.linguistic
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:  # pragma: no cover - env dependent
        raise ImportError(
            "Supplementary embeddings need `transformers` + `torch`. Install them "
            "and enable Internet (Kaggle) to download "
            f"'{lc.embedding_model}', or set features.linguistic.use_embedding=false."
        ) from e

    cache = getattr(extract_embedding, "_cache", {})
    key = lc.embedding_model
    if key not in cache:
        tok = AutoTokenizer.from_pretrained(key)
        mdl = AutoModel.from_pretrained(key).eval()
        cache[key] = (tok, mdl)
        extract_embedding._cache = cache  # type: ignore[attr-defined]
    tok, mdl = cache[key]

    with torch.no_grad():
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        out = mdl(**enc).last_hidden_state          # (1, T, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (1, T, 1)
        if lc.embedding_pooling == "mean":
            pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:  # 'cls'
            pooled = out[:, 0]
        vec = pooled.squeeze(0).cpu().numpy()

    names = embedding_feature_names(len(vec))
    return {n: float(v) for n, v in zip(names, vec)}
