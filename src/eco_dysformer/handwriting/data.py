"""Handwriting image dataset (RQ3 auxiliary branch).

Verified layout (from inspect_handwriting): ``<root>/{Train,Test}/{Normal,
Reversal,Corrected}/*.png``, 28x28 RGB character images, ~208k total, NO writer
linkage. This loader scans that structure defensively (asserts the expected
label folders exist), maps folders to labels per config, and yields tensors.

Primary signal is BINARY reversal-vs-normal (``classes: {Normal:0, Reversal:1}``);
``Corrected`` is excluded unless ``include_corrected`` adds it as a third class.
Because there are no writers, the Train/Test split carries no leakage risk within
the handwriting cohort (there is nothing to leak across).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def discover_split(root: Path, split: str, class_map: dict[str, int]) -> list[tuple[Path, int]]:
    """Return ``[(image_path, label), ...]`` for one split, asserting structure."""
    split_dir = root / split
    assert split_dir.is_dir(), (
        f"expected split folder {split_dir} (run the extract step; root should "
        f"contain Train/ and Test/)"
    )
    items: list[tuple[Path, int]] = []
    for folder, label in class_map.items():
        cdir = split_dir / folder
        assert cdir.is_dir(), f"expected class folder {cdir} not found"
        n = 0
        for p in cdir.iterdir():
            if p.suffix.lower() in _IMG_EXTS:
                items.append((p, label))
                n += 1
        assert n > 0, f"no images under {cdir}"
    return items


class HandwritingDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]], image_size: int = 28,
                 grayscale: bool = True):
        self.items = items
        self.image_size = image_size
        self.grayscale = grayscale

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        from PIL import Image
        path, label = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("L" if self.grayscale else "RGB")
            if im.size != (self.image_size, self.image_size):
                im = im.resize((self.image_size, self.image_size))
            arr = np.asarray(im, dtype=np.float32) / 255.0
        if self.grayscale:
            arr = arr[None, :, :]                 # (1, H, W)
        else:
            arr = arr.transpose(2, 0, 1)          # (3, H, W)
        return torch.from_numpy(arr), int(label)


def _subsample(items, max_per_class, rng):
    if not max_per_class:
        return items
    by_label: dict[int, list] = {}
    for it in items:
        by_label.setdefault(it[1], []).append(it)
    out = []
    for label, lst in by_label.items():
        idx = rng.permutation(len(lst))[:max_per_class]
        out.extend(lst[i] for i in idx)
    return out


def build_loaders(cfg, seed: int):
    """Return (train_loader, test_loader, meta) from the handwriting config."""
    hw = cfg.rq3.handwriting
    root = Path(hw.data_root)
    class_map = dict(hw.classes.to_dict() if hasattr(hw.classes, "to_dict") else hw.classes)
    if hw.include_corrected and "Corrected" not in class_map:
        class_map["Corrected"] = max(class_map.values()) + 1

    rng = np.random.default_rng(seed)
    train_items = _subsample(discover_split(root, "Train", class_map),
                             hw.get("max_images_per_class"), rng)
    test_items = discover_split(root, "Test", class_map)

    n_classes = len(set(class_map.values()))
    ds_kw = dict(image_size=hw.image_size, grayscale=hw.grayscale)
    train_loader = DataLoader(
        HandwritingDataset(train_items, **ds_kw), batch_size=hw.train.batch_size,
        shuffle=True, num_workers=hw.train.num_workers,
        generator=torch.Generator().manual_seed(seed), drop_last=False)
    test_loader = DataLoader(
        HandwritingDataset(test_items, **ds_kw), batch_size=hw.train.batch_size,
        shuffle=False, num_workers=hw.train.num_workers)

    meta = {
        "class_map": class_map,
        "n_classes": n_classes,
        "n_train": len(train_items),
        "n_test": len(test_items),
        "in_channels": 1 if hw.grayscale else 3,
    }
    return train_loader, test_loader, meta
