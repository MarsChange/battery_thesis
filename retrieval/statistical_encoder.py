"""Deterministic numerical window encoder for legacy FAISS utilities.

The main forecasting pipeline uses the case-bank retriever, not this helper.
This encoder exists only so older FAISS demos can still build an index without
external sequence foundation models. It summarizes each window with named,
bounded statistics over the raw numerical channels.
"""

from __future__ import annotations

import numpy as np


class StatisticalWindowEncoder:
    """Encode a raw sliding window with simple numerical statistics.

    Input shape is ``[B, L]`` or ``[B, L, C]``. For each channel the encoder
    returns mean, standard deviation, minimum, maximum, last value, and a linear
    slope proxy. Larger downstream distances still come from FAISS metric
    choice; this encoder does not introduce black-box learned features.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    @property
    def embedding_dim(self) -> int | None:
        """Return ``None`` because the dimension depends on input channels."""

        return None

    def encode(self, windows: np.ndarray) -> np.ndarray:
        """Return fixed statistics for each input window."""

        arr = np.asarray(windows, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError("windows must have shape [B, L] or [B, L, C]")

        valid = np.isfinite(arr)
        safe = np.where(valid, arr, np.nan)
        mean = np.nanmean(safe, axis=1)
        std = np.nanstd(safe, axis=1)
        vmin = np.nanmin(safe, axis=1)
        vmax = np.nanmax(safe, axis=1)
        last = safe[:, -1, :]

        x = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
        x_centered = x - x.mean()
        y_centered = safe - np.nanmean(safe, axis=1, keepdims=True)
        slope = np.nansum(y_centered * x_centered[None, :, None], axis=1) / (
            np.sum(x_centered**2) + self.eps
        )

        emb = np.concatenate([mean, std, vmin, vmax, last, slope], axis=1)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return emb
