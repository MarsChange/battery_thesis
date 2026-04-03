"""
schema.py — Data structures for the retrieval framework.

Defines WindowSample (what goes *into* the database) and
SearchResult (what comes *out* of a query).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class WindowSample:
    """A single windowed sample stored in the retrieval database.

    Attributes
    ----------
    series_id : str
        Identifier for the source time series.
    window_start : int
        Start index of the lookback window in the original series.
    window_end : int
        End index (exclusive) of the lookback window.
    target_start : int
        Start index of the prediction / future horizon.
    target_end : int
        End index (exclusive) of the prediction horizon.
    window_values : np.ndarray
        Shape ``(lookback_length,)`` or ``(lookback_length, C)``.
    future_values : np.ndarray | None
        Shape ``(prediction_length,)`` or ``(prediction_length, C)``.
        May be ``None`` when building query-only samples.
    metadata : dict
        Arbitrary key-value pairs (dataset name, frequency, etc.).
    """

    series_id: str
    window_start: int
    window_end: int
    target_start: int
    target_end: int
    window_values: np.ndarray
    future_values: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result returned by a top-k search.

    Attributes
    ----------
    query_id : int
        Index of the query in the batch.
    distances : np.ndarray
        Shape ``(top_k,)`` — distance to each neighbor.
    neighbor_ids : np.ndarray
        Shape ``(top_k,)`` — global integer IDs in the database.
    neighbor_series_ids : list[str]
        Series ID of each neighbor.
    neighbor_window_starts : np.ndarray
        Shape ``(top_k,)`` — window_start of each neighbor.
    neighbor_window_ends : np.ndarray
        Shape ``(top_k,)`` — window_end of each neighbor.
    neighbor_future_values : list[np.ndarray] | None
        Future horizon values of each neighbor (if stored).
    neighbor_metadata : list[dict]
        Metadata dicts of each neighbor.
    """

    query_id: int
    distances: np.ndarray
    neighbor_ids: np.ndarray
    neighbor_series_ids: List[str]
    neighbor_window_starts: np.ndarray
    neighbor_window_ends: np.ndarray
    neighbor_future_values: Optional[List[np.ndarray]] = None
    neighbor_metadata: List[Dict[str, Any]] = field(default_factory=list)
