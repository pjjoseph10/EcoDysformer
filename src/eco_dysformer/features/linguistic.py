"""Per-passage linguistic complexity features (Czech-appropriate).

Computed once per passage (there are only 3 distinct stimuli) from the Czech
text reconstructed in :mod:`eco_dysformer.data.stimuli`. Because these values are
identical across children, their role is the RQ2 *conditioning* signal, not
per-child discrimination.

Engine: UDPipe (Charles University, strong Czech treebank support -- the
proposal's safer default) or Stanza, selected via ``config.features.linguistic
.engine``. Both are imported lazily and, if unavailable, raise a clear
ImportError with install/run guidance rather than emitting placeholder numbers.
This module is Kaggle-first: the local bare env cannot run it, and that is by
design (HARD RULE: failing assertion, never fabricated metrics).

Features produced (see :data:`LINGUISTIC_FEATURE_NAMES`):
    - surface (stdlib, always computable): token/sentence counts, mean word
      length, type-token ratio, mean sentence length
    - syntactic: mean parse-tree depth, mean dependency distance
    - lexical:   lexical density (content-POS / tokens)
    - frequency: mean / std / min Zipf frequency (wordfreq, Czech)
Optional supplementary XLM-R/RobeCzech embedding lives in ``embeddings.py`` and
is appended only when ``config.features.linguistic.use_embedding`` is true.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

# UD content parts of speech for lexical density.
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

# Surface features are always available; parser/frequency features require the
# engine + wordfreq. Kept explicit so the feature-table schema is stable.
SURFACE_FEATURE_NAMES = [
    "ling_n_tokens",
    "ling_n_sentences",
    "ling_mean_word_len",
    "ling_type_token_ratio",
    "ling_mean_sentence_len",
]
PARSER_FEATURE_NAMES = [
    "ling_mean_tree_depth",
    "ling_mean_dependency_distance",
    "ling_lexical_density",
]
FREQUENCY_FEATURE_NAMES = [
    "ling_mean_zipf",
    "ling_std_zipf",
    "ling_min_zipf",
]
LINGUISTIC_FEATURE_NAMES = (
    SURFACE_FEATURE_NAMES + PARSER_FEATURE_NAMES + FREQUENCY_FEATURE_NAMES
)


@dataclass
class Token:
    idx: int          # 1-based id within sentence
    head: int         # id of syntactic head (0 = root)
    upos: str         # universal POS tag
    form: str         # surface form


# --------------------------------------------------------------------------- #
# Engine loading (lazy)
# --------------------------------------------------------------------------- #
class _EngineUnavailable(ImportError):
    pass


def _parse_with_udpipe(text: str, model_path: str) -> list[list[Token]]:
    try:
        from ufal.udpipe import Model, Pipeline
    except ImportError as e:  # pragma: no cover - env dependent
        raise _EngineUnavailable(
            "UDPipe (ufal.udpipe) is not installed. `pip install ufal.udpipe` and "
            "provide a Czech model (config.features.linguistic.udpipe_model). "
            "Run on Kaggle with Internet enabled."
        ) from e

    import os
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"UDPipe model not found at {model_path!r}. Download the Czech model "
            "(e.g. czech-pdt-ud-2.5) from the UDPipe/LINDAT model repository and "
            "point config.features.linguistic.udpipe_model at it."
        )
    model = Model.load(model_path)
    if model is None:
        raise RuntimeError(f"UDPipe failed to load model {model_path!r}")
    pipe = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
    conllu = pipe.process(text)
    return _parse_conllu(conllu)


def _parse_conllu(conllu: str) -> list[list[Token]]:
    sentences: list[list[Token]] = []
    cur: list[Token] = []
    for line in conllu.splitlines():
        if not line.strip():
            if cur:
                sentences.append(cur)
                cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 8 or "-" in cols[0] or "." in cols[0]:
            continue  # skip multiword-token ranges / empty nodes
        cur.append(Token(idx=int(cols[0]), head=int(cols[6]),
                         upos=cols[3], form=cols[1]))
    if cur:
        sentences.append(cur)
    return sentences


def _parse_with_stanza(text: str, lang: str) -> list[list[Token]]:
    try:
        import stanza
    except ImportError as e:  # pragma: no cover - env dependent
        raise _EngineUnavailable(
            "Stanza is not installed. `pip install stanza` and it will download "
            "the Czech model on first use (Internet required). Run on Kaggle."
        ) from e

    # Cache the pipeline on the function to avoid reloading per passage.
    nlp = getattr(_parse_with_stanza, "_nlp", None)
    if nlp is None:
        try:
            nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse",
                                  verbose=False)
        except Exception:
            stanza.download(lang, verbose=False)
            nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse",
                                  verbose=False)
        _parse_with_stanza._nlp = nlp  # type: ignore[attr-defined]

    doc = nlp(text)
    sentences: list[list[Token]] = []
    for sent in doc.sentences:
        toks = [Token(idx=int(w.id), head=int(w.head), upos=w.upos, form=w.text)
                for w in sent.words]
        sentences.append(toks)
    return sentences


# --------------------------------------------------------------------------- #
# Feature computation
# --------------------------------------------------------------------------- #
def _token_depth(tok: Token, by_id: dict[int, Token], _cache: dict[int, int]) -> int:
    """Depth of a token in its dependency tree (root depth = 0)."""
    if tok.head == 0:
        return 0
    if tok.idx in _cache:
        return _cache[tok.idx]
    depth, cur, seen = 0, tok, set()
    while cur.head != 0:
        if cur.idx in seen:      # cycle guard (malformed parse)
            break
        seen.add(cur.idx)
        depth += 1
        parent = by_id.get(cur.head)
        if parent is None:
            break
        cur = parent
    _cache[tok.idx] = depth
    return depth


def _surface_features(text: str) -> dict[str, float]:
    tokens = re.findall(r"\w+", text, flags=re.UNICODE)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    n_tok = len(tokens)
    n_sent = max(len(sentences), 1)
    types = {t.lower() for t in tokens}
    return {
        "ling_n_tokens": float(n_tok),
        "ling_n_sentences": float(n_sent),
        "ling_mean_word_len": float(statistics.mean(len(t) for t in tokens)) if n_tok else 0.0,
        "ling_type_token_ratio": float(len(types) / n_tok) if n_tok else 0.0,
        "ling_mean_sentence_len": float(n_tok / n_sent),
    }


def _parser_features(sentences: list[list[Token]]) -> dict[str, float]:
    tree_depths, dep_dists, content, total = [], [], 0, 0
    for sent in sentences:
        by_id = {t.idx: t for t in sent}
        cache: dict[int, int] = {}
        depths = [_token_depth(t, by_id, cache) for t in sent]
        if depths:
            tree_depths.append(max(depths))
        for t in sent:
            if t.head != 0:
                dep_dists.append(abs(t.head - t.idx))
            if t.upos != "PUNCT":
                total += 1
                if t.upos in _CONTENT_POS:
                    content += 1
    return {
        "ling_mean_tree_depth": float(statistics.mean(tree_depths)) if tree_depths else 0.0,
        "ling_mean_dependency_distance": float(statistics.mean(dep_dists)) if dep_dists else 0.0,
        "ling_lexical_density": float(content / total) if total else 0.0,
    }


def _frequency_features(text: str, lang: str) -> dict[str, float]:
    try:
        from wordfreq import zipf_frequency
    except ImportError as e:  # pragma: no cover - env dependent
        raise _EngineUnavailable(
            "wordfreq is not installed. `pip install wordfreq` (Czech Zipf table)."
        ) from e
    tokens = [t.lower() for t in re.findall(r"\w+", text, flags=re.UNICODE)]
    zipfs = [zipf_frequency(t, lang) for t in tokens] if tokens else []
    if not zipfs:
        return {"ling_mean_zipf": 0.0, "ling_std_zipf": 0.0, "ling_min_zipf": 0.0}
    return {
        "ling_mean_zipf": float(statistics.mean(zipfs)),
        "ling_std_zipf": float(statistics.pstdev(zipfs)) if len(zipfs) > 1 else 0.0,
        "ling_min_zipf": float(min(zipfs)),
    }


def extract_linguistic_features(text: str, cfg) -> dict[str, float]:
    """Compute the full per-passage linguistic feature dict for one text.

    Raises ``_EngineUnavailable`` (an ImportError) if the parser/frequency
    dependencies are missing -- this is the intended Kaggle-first behavior.
    """
    lc = cfg.features.linguistic
    engine = lc.engine
    if engine == "udpipe":
        sentences = _parse_with_udpipe(text, lc.udpipe_model)
    elif engine == "stanza":
        sentences = _parse_with_stanza(text, lc.stanza_lang)
    else:
        raise ValueError(f"unknown linguistic engine {engine!r}; use udpipe|stanza")

    feats: dict[str, float] = {}
    feats.update(_surface_features(text))
    feats.update(_parser_features(sentences))
    feats.update(_frequency_features(text, lc.zipf_lang))
    # Guarantee full ordered schema.
    return {k: feats.get(k, float("nan")) for k in LINGUISTIC_FEATURE_NAMES}
