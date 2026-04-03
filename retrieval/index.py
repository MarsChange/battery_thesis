"""
index.py — FAISS index wrapper with L2 and cosine similarity support.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import faiss
import numpy as np


class FAISSIndex:
    """Thin wrapper around a FAISS flat index.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    metric : ``"cosine"`` | ``"l2"``
        Similarity metric.  Cosine is implemented via L2-normalised vectors
        + inner-product index.
    """

    def __init__(self, dim: int, metric: Literal["cosine", "l2"] = "cosine"):
        self.dim = dim
        self.metric = metric
        self._normalize = metric == "cosine"

        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dim)
        elif metric == "l2":
            self._index = faiss.IndexFlatL2(dim)
        else:
            raise ValueError(f"Unsupported metric: {metric!r}")

    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray, shape ``(N, dim)``, dtype float32.
        """
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if self._normalize:
            faiss.normalize_L2(vectors)
        self._index.add(vectors)

    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search the index.

        Returns
        -------
        distances : np.ndarray, shape ``(n_queries, top_k)``
            For cosine: higher is more similar (inner product of unit vecs).
            For L2: lower is more similar.
        indices : np.ndarray, shape ``(n_queries, top_k)``
        """
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[np.newaxis, :]
        if self._normalize:
            faiss.normalize_L2(query)
        distances, indices = self._index.search(query, top_k)
        return distances, indices

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        faiss.write_index(self._index, path)

    @classmethod
    def load(cls, path: str, metric: Literal["cosine", "l2"] = "cosine") -> "FAISSIndex":
        raw = faiss.read_index(path)
        obj = cls.__new__(cls)
        obj._index = raw
        obj.dim = raw.d
        obj.metric = metric
        obj._normalize = metric == "cosine"
        return obj
