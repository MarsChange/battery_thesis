"""Query a FAISS retrieval database and return structured numerical results."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd

from .index import FAISSIndex
from .io import load_sidecar
from .schema import SearchResult


class RetrieverSearcher:
    """Load a persisted database and answer top-k queries.

    Parameters
    ----------
    db_dir : str | Path
        Directory containing ``<name>.faiss``, ``<name>_meta.parquet``,
        and optionally ``<name>_futures.npy``.
    name : str
        Prefix used when saving (default ``"db"``).
    metric : ``"cosine"`` | ``"l2"``
        Must match what was used at build time.
    encoder : object | None
        Optional numerical encoder with an ``encode(windows)`` method. The
        current project does not attach external sequence foundation encoders.
    """

    def __init__(
        self,
        db_dir: str | Path,
        name: str = "db",
        metric: Literal["cosine", "l2"] = "cosine",
        encoder: Optional[Any] = None,
    ):
        self.db_dir = Path(db_dir)
        self.name = name
        self.metric = metric
        self.encoder = encoder

        # Load FAISS index
        self.index = FAISSIndex.load(str(self.db_dir / f"{name}.faiss"), metric=metric)

        # Load sidecar
        self.meta_df, self.futures = load_sidecar(self.db_dir, name)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def search_by_embedding(
        self,
        query_emb: np.ndarray,
        top_k: int = 5,
        drop_self_ids: Optional[set] = None,
        time_boundary: Optional[int] = None,
    ) -> List[SearchResult]:
        """Search using pre-computed query embeddings.

        Parameters
        ----------
        query_emb : np.ndarray, shape ``(B, D)`` or ``(D,)``
        top_k : int
        drop_self_ids : set of int | None
            Global DB row IDs to exclude (e.g. the query's own row).
        time_boundary : int | None
            If set, exclude neighbours whose ``window_end > time_boundary``
            to prevent future leakage.

        Returns
        -------
        results : list[SearchResult]
        """
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]

        # Over-fetch to account for filtering
        fetch_k = top_k + (10 if drop_self_ids or time_boundary else 0)
        fetch_k = min(fetch_k, self.index.ntotal)

        distances, indices = self.index.search(query_emb, fetch_k)

        results: List[SearchResult] = []
        for q_idx in range(len(query_emb)):
            dists_q = distances[q_idx]
            ids_q = indices[q_idx]

            # Filter
            mask = ids_q >= 0  # FAISS returns -1 for missing
            if drop_self_ids:
                mask &= np.array([int(i) not in drop_self_ids for i in ids_q])
            if time_boundary is not None:
                w_ends = self.meta_df["window_end"].values[ids_q[mask]]
                mask_tb = w_ends <= time_boundary
                valid_pos = np.where(mask)[0]
                mask[valid_pos[~mask_tb]] = False

            valid = np.where(mask)[0][:top_k]
            sel_ids = ids_q[valid]
            sel_dists = dists_q[valid]

            rows = self.meta_df.iloc[sel_ids]
            future_vals = (
                [self.futures[i] for i in sel_ids] if self.futures is not None else None
            )

            meta_list = []
            for _, row in rows.iterrows():
                try:
                    meta_list.append(json.loads(row["metadata_json"]))
                except (TypeError, json.JSONDecodeError):
                    try:
                        meta_list.append(ast.literal_eval(row["metadata_json"]))
                    except (ValueError, SyntaxError):
                        meta_list.append({})

            results.append(
                SearchResult(
                    query_id=q_idx,
                    distances=sel_dists,
                    neighbor_ids=sel_ids,
                    neighbor_series_ids=rows["series_id"].tolist(),
                    neighbor_window_starts=rows["window_start"].values,
                    neighbor_window_ends=rows["window_end"].values,
                    neighbor_future_values=future_vals,
                    neighbor_metadata=meta_list,
                )
            )
        return results

    def search_by_window(
        self,
        query_windows: np.ndarray,
        top_k: int = 5,
        drop_self_ids: Optional[set] = None,
        time_boundary: Optional[int] = None,
    ) -> List[SearchResult]:
        """Encode raw windows, then search.

        Parameters
        ----------
        query_windows : np.ndarray, shape ``(B, L)`` or ``(B, L, C)``
        """
        if self.encoder is None:
            raise RuntimeError("No encoder attached; use search_by_embedding() or provide encoder.")
        emb = self.encoder.encode(query_windows)
        return self.search_by_embedding(emb, top_k, drop_self_ids, time_boundary)
