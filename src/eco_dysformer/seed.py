"""Global seeding and determinism.

One call -- :func:`set_global_seed` -- seeds Python ``random``, NumPy, and (if
installed) PyTorch, and flips the deterministic flags. LightGBM is seeded via its
``random_state`` param at construction time; :func:`lightgbm_seed_params` returns
the dict to splat into the LightGBM constructor.

torch is imported lazily and guarded, so this module imports fine in the bare
local environment (no torch) used for the data/features/stats steps.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_global_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed every RNG we use and (optionally) request deterministic kernels.

    Parameters
    ----------
    seed
        The global seed from ``config.seed``.
    deterministic
        If True, set the env vars / torch flags that make CUDA and cuDNN behave
        deterministically. Costs some speed; correct for a reproducibility study.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---- PyTorch (optional; absent in the bare local env) ----
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS reproducibility for matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch may not support warn_only; best-effort.
            pass


def lightgbm_seed_params(seed: int) -> dict[str, Any]:
    """Return LightGBM kwargs that fully pin its randomness to ``seed``."""
    return {
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_row_wise": True,  # avoids a nondeterministic threading path
    }


def torch_generator(seed: int):
    """Return a seeded ``torch.Generator`` (for DataLoader shuffling etc.).

    Raises ImportError if torch is unavailable -- callers in the neural path
    already require torch, so this is an intentional hard dependency there.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
