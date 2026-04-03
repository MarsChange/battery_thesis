"""
encoder_chronos2.py — Chronos-2 embedding encoder for time series retrieval.

Uses ``chronos.Chronos2Pipeline`` (shipped with chronos-forecasting ≥ 2.0,
bundled as a dependency of autogluon.timeseries ≥ 1.5).

Why Chronos2Pipeline directly?
  AutoGluon 1.5's high-level ``TimeSeriesPredictor`` API focuses on forecasting
  and does **not** expose a retrieval-embedding endpoint.  ``Chronos2Pipeline``
  provides a first-class ``.embed()`` method that returns encoder hidden states.
  We wrap it here with a thin class that:
    • accepts numpy / torch input in ``[B, L]`` or ``[B, L, C]`` layout,
    • pools the patch-level hidden states into a fixed ``[B, D]`` vector,
    • keeps all processing deterministic and batch-independent.

Pooling strategies
  * ``mean``  — mean over the patch dimension (default, more stable).
  * ``last``  — last token embedding (mirrors the old TS-RAG EOS approach).

Cross-learning / batch coupling
  ``Chronos2Pipeline.embed()`` processes each series independently (no
  cross-series attention), so there is no cross-learning effect between
  samples in a batch.  The ``batch_size`` parameter controls only GPU
  parallelism, not model semantics.
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import torch


class Chronos2RetrieverEncoder:
    """Extract fixed-dim embeddings from Chronos-2 for retrieval.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path, e.g. ``"autogluon/chronos-2"``.
    pooling : ``"mean"`` | ``"last"``
        How to reduce ``(num_patches+2, D)`` → ``(D,)``.
    device : str
        ``"cpu"``, ``"cuda"``, ``"cuda:0"``, etc.
    context_length : int | None
        Maximum context fed to the model (default: let model decide).
    batch_size : int
        Internal batch size for ``pipeline.embed()``.
    torch_dtype : str
        ``"float32"`` or ``"bfloat16"`` (bfloat16 is faster on modern GPUs).
    """

    VALID_POOLING = ("mean", "last")

    def __init__(
        self,
        model_path: str = "autogluon/chronos-2",
        pooling: Literal["mean", "last"] = "mean",
        device: str = "cpu",
        context_length: Optional[int] = None,
        batch_size: int = 256,
        torch_dtype: str = "float32",
    ):
        if pooling not in self.VALID_POOLING:
            raise ValueError(f"pooling must be one of {self.VALID_POOLING}, got {pooling!r}")

        self.model_path = model_path
        self.pooling = pooling
        self.device = device
        self.context_length = context_length
        self.batch_size = batch_size
        self.torch_dtype_str = torch_dtype

        # Lazy-load to avoid import overhead when just configuring
        self._pipeline = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        from chronos import Chronos2Pipeline

        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float32)

        self._pipeline = Chronos2Pipeline.from_pretrained(
            self.model_path,
            device_map=self.device,
            dtype=torch_dtype,
        )

    @property
    def pipeline(self):
        if self._pipeline is None:
            self._load_pipeline()
        return self._pipeline

    @property
    def embedding_dim(self) -> int:
        """Return the model's hidden size (typically 768)."""
        return self.pipeline.model.config.d_model

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """Encode a batch of time series windows into fixed-dim embeddings.

        Parameters
        ----------
        x : array-like, shape ``[B, L]`` or ``[B, L, C]``
            Batch of lookback windows.
            * ``[B, L]`` — univariate (most common).
            * ``[B, L, C]`` — multivariate; each channel is embedded
              independently, then averaged across channels.

        Returns
        -------
        embeddings : np.ndarray, shape ``[B, D]``
            Fixed-dimensional embedding vectors (float32).
        """
        # --- Normalise input to numpy [B, L] or [B, L, C] ----------------
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x[np.newaxis, :]  # [L] -> [1, L]

        multivariate = x.ndim == 3
        if multivariate:
            B, L, C = x.shape
            # Chronos2 expects (batch, n_variates, length)
            x_input = x.transpose(0, 2, 1)  # [B, C, L]
        else:
            B, L = x.shape
            C = 1
            x_input = x[:, np.newaxis, :]  # [B, 1, L]

        # --- Embed via Chronos2Pipeline -----------------------------------
        embeddings_list, _ = self.pipeline.embed(
            x_input,
            batch_size=self.batch_size,
            context_length=self.context_length,
        )
        # embeddings_list: list of B tensors, each (n_variates, num_patches+2, D)

        pooled = []
        for emb_tensor in embeddings_list:
            # emb_tensor: (n_variates, S, D)  where S = num_patches + 2
            emb = emb_tensor.float()  # ensure float32

            if self.pooling == "mean":
                # mean over the patch/sequence dimension
                vec = emb.mean(dim=1)  # (n_variates, D)
            elif self.pooling == "last":
                vec = emb[:, -1, :]  # (n_variates, D)
            else:
                raise ValueError(f"Unknown pooling: {self.pooling}")

            # Average across variates for multivariate case
            vec = vec.mean(dim=0)  # (D,)
            pooled.append(vec)

        result = torch.stack(pooled, dim=0).numpy()  # (B, D)
        return result
