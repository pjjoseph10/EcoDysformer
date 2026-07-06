"""ETDD70 file discovery and defensive loading.

Builds a manifest of every ``Subject_<id>_<task>_<type>.csv`` on disk, maps each
task token to a passage name via the config, and asserts the grid is complete
(70 subjects x 3 passages x 4 file types = 840 files). Loaders return raw pandas
DataFrames; feature code lives in ``features/``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Subject_1235_T5_Pseudo_Text_fixations.csv
#         ^id  ^task (non-greedy) ......  ^file type
_FNAME_RE = re.compile(
    r"^Subject_(?P<subject>\d+)_(?P<task>T\d+_.+?)_(?P<ftype>raw|fixations|saccades|metrics)\.csv$"
)


class ETDD70:
    """Handle to the ETDD70 dataset on disk.

    Parameters
    ----------
    cfg
        A loaded :class:`~eco_dysformer.config.Config`.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = Path(cfg.paths.data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        self.file_types = list(cfg.dataset.file_types)
        # task token (e.g. 'T5_Pseudo_Text') -> passage name (e.g. 'pseudo')
        self._task_to_name = {p["task"]: p["name"] for p in cfg.dataset.passages}
        self._name_to_task = {p["name"]: p["task"] for p in cfg.dataset.passages}
        self.passage_names = [p["name"] for p in cfg.dataset.passages]

        self.manifest = self._build_manifest()
        self._validate_grid()

    # ------------------------------------------------------------------ build
    def _build_manifest(self) -> pd.DataFrame:
        rows = []
        unknown_tasks: set[str] = set()
        for path in sorted(self.data_dir.glob("Subject_*.csv")):
            m = _FNAME_RE.match(path.name)
            if not m:
                # A file we did not anticipate -- record and skip, do not crash,
                # but the grid validation below will notice any real shortfall.
                continue
            task = m.group("task")
            name = self._task_to_name.get(task)
            if name is None:
                unknown_tasks.add(task)
                continue
            rows.append({
                "subject_id": int(m.group("subject")),
                "task": task,
                "passage_name": name,
                "file_type": m.group("ftype"),
                "path": str(path),
            })

        assert not unknown_tasks, (
            f"found task tokens on disk not present in config.dataset.passages: "
            f"{sorted(unknown_tasks)} -- update the config, do not ignore."
        )
        assert rows, f"no ETDD70 csv files matched under {self.data_dir}"
        return pd.DataFrame(rows)

    def _validate_grid(self) -> None:
        cfg = self.cfg
        subjects = sorted(self.manifest["subject_id"].unique())
        n_exp = cfg.dataset.n_subjects_expected
        assert len(subjects) == n_exp, (
            f"expected {n_exp} subjects on disk, found {len(subjects)}"
        )

        # Every (subject, passage) must have all file types present.
        grid = (self.manifest
                .groupby(["subject_id", "passage_name"])["file_type"]
                .apply(lambda s: set(s)))
        needed = set(self.file_types)
        problems = []
        for (sid, passage), have in grid.items():
            miss = needed - have
            if miss:
                problems.append((sid, passage, sorted(miss)))
        assert not problems, (
            f"incomplete file grid for {len(problems)} (subject,passage) cells, "
            f"e.g. {problems[:5]}"
        )

        n_cells = len(grid)
        expected_cells = n_exp * len(self.passage_names)
        assert n_cells == expected_cells, (
            f"expected {expected_cells} (subject,passage) cells, got {n_cells}"
        )

    # ------------------------------------------------------------------ access
    def subjects(self) -> list[int]:
        return sorted(self.manifest["subject_id"].unique().tolist())

    def path(self, subject_id: int, passage_name: str, file_type: str) -> Path:
        if file_type not in self.file_types:
            raise ValueError(f"unknown file_type {file_type!r}; use {self.file_types}")
        if passage_name not in self._name_to_task:
            raise ValueError(
                f"unknown passage {passage_name!r}; use {self.passage_names}"
            )
        sel = self.manifest[
            (self.manifest["subject_id"] == subject_id)
            & (self.manifest["passage_name"] == passage_name)
            & (self.manifest["file_type"] == file_type)
        ]
        assert len(sel) == 1, (
            f"expected exactly 1 file for "
            f"({subject_id}, {passage_name}, {file_type}); got {len(sel)}"
        )
        return Path(sel.iloc[0]["path"])

    def load(self, subject_id: int, passage_name: str, file_type: str) -> pd.DataFrame:
        """Load one CSV as a DataFrame (no feature processing)."""
        return pd.read_csv(self.path(subject_id, passage_name, file_type))

    def iter_cells(self):
        """Yield ``(subject_id, passage_name)`` for all 210 cells, sorted."""
        for sid in self.subjects():
            for name in self.passage_names:
                yield sid, name
