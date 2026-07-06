"""Configuration loading, validation, and path resolution.

The YAML at ``configs/stage1.yaml`` is the single source of truth for a run.
This module loads it, resolves every path in the ``paths`` block relative to the
repo root (so the project is portable between local dev and Kaggle), validates a
handful of invariants we depend on downstream, and exposes the result as a small
dotted-access wrapper.

Deliberately dependency-light: only PyYAML is required, so this imports cleanly
in the bare local environment.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import yaml


def repo_root() -> Path:
    """Return the repo root (the directory that contains ``configs/``).

    Resolved from this file's location: ``src/eco_dysformer/config.py`` -> up 3.
    """
    return Path(__file__).resolve().parents[2]


class Config:
    """Thin dotted-access wrapper around the parsed YAML dict.

    ``cfg.paths.data_dir``, ``cfg.model.gaze_encoder.d_model``, etc. Unknown keys
    raise ``AttributeError`` rather than returning ``None`` silently, so typos in
    code surface immediately. Use :meth:`get` for optional keys with a default.
    """

    def __init__(self, data: dict[str, Any]):
        self._data = data

    def __getattr__(self, key: str) -> Any:
        # __getattr__ only fires for names not found normally, so no recursion
        # on self._data (which is set via __dict__ in __init__).
        try:
            value = self._data[key]
        except KeyError as exc:
            raise AttributeError(
                f"config has no key '{key}' (available: {sorted(self._data)})"
            ) from exc
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value) if isinstance(value, dict) else value

    def get(self, key: str, default: Any = None) -> Any:
        value = self._data.get(key, default)
        return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def keys(self):
        return self._data.keys()

    def to_dict(self) -> dict[str, Any]:
        """Return a deep copy as plain dicts/lists (safe to serialize)."""
        def _plain(v: Any) -> Any:
            if isinstance(v, Config):
                return {k: _plain(vv) for k, vv in v._data.items()}
            if isinstance(v, dict):
                return {k: _plain(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_plain(vv) for vv in v]
            return v
        return _plain(self)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def _resolve_paths(data: dict[str, Any], root: Path) -> None:
    """In-place: make every value under ``paths`` an absolute, root-relative path."""
    paths = data.get("paths", {})
    for key, value in paths.items():
        p = Path(value)
        paths[key] = str(p if p.is_absolute() else (root / p))


def _validate(data: dict[str, Any]) -> None:
    """Assert invariants the rest of the pipeline relies on. Fail loud, fail early."""
    assert "seed" in data, "config must define a global `seed`"

    ds = data.get("dataset", {})
    passages = ds.get("passages", [])
    assert len(passages) == 3, (
        f"expected exactly 3 passages (ETDD70 syllable/narrative/pseudo), "
        f"got {len(passages)}"
    )
    ranks = sorted(p["complexity_rank"] for p in passages)
    assert ranks == [0, 1, 2], f"passage complexity_ranks must be 0,1,2; got {ranks}"

    cv = data.get("eval", {}).get("cv", {})
    assert cv.get("group_key") == "subject_id", (
        "eval.cv.group_key MUST be 'subject_id' -- subject-level folds are the "
        "single most important correctness requirement; a child's passages must "
        "never split across train/test."
    )
    assert cv.get("outer_folds", 0) >= 2 and cv.get("inner_folds", 0) >= 2, (
        "nested CV needs >=2 outer and >=2 inner folds"
    )

    assert data.get("explain", {}).get("on_features") == "original", (
        "explain.on_features MUST be 'original' -- LIME/attention run on "
        "interpretable features, NEVER on PCA components."
    )


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load, validate, and return the Stage-1 config.

    Parameters
    ----------
    path
        Path to the YAML. Defaults to ``<repo_root>/configs/stage1.yaml``.
    """
    root = repo_root()
    if path is None:
        path = root / "configs" / "stage1.yaml"
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data)}")

    _resolve_paths(data, root)
    _validate(data)
    return Config(data)


if __name__ == "__main__":
    # Quick smoke test: load and print resolved paths + dataset schema summary.
    cfg = load_config()
    print("repo root:", repo_root())
    print("seed:", cfg.seed)
    print("paths:")
    for k in cfg.paths.keys():
        print(f"  {k:14s} -> {cfg.paths[k]}")
    print("passages:", [p["name"] for p in cfg.dataset.passages])
