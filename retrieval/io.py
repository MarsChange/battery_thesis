"""
io.py — Side-car storage for metadata and future values.

The FAISS index stores only embedding vectors.  Everything else
(series_id, window boundaries, future_values, metadata) is persisted
in a pair of files:

  * ``<name>_meta.parquet`` — one row per sample (scalar fields).
  * ``<name>_futures.npy``  — stacked future_values array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import WindowSample


def save_sidecar(
    samples: List[WindowSample],
    directory: str | Path,
    name: str = "db",
) -> None:
    """Persist sample metadata and future values alongside a FAISS index."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    records = []
    futures = []
    for s in samples:
        records.append(
            {
                "series_id": s.series_id,
                "window_start": s.window_start,
                "window_end": s.window_end,
                "target_start": s.target_start,
                "target_end": s.target_end,
                "metadata_json": str(s.metadata),
            }
        )
        if s.future_values is not None:
            futures.append(s.future_values)

    df = pd.DataFrame(records)
    df.to_parquet(directory / f"{name}_meta.parquet", index=False)

    if futures:
        np.save(directory / f"{name}_futures.npy", np.stack(futures))


def load_sidecar(
    directory: str | Path,
    name: str = "db",
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Load metadata parquet and optional futures array."""
    directory = Path(directory)
    df = pd.read_parquet(directory / f"{name}_meta.parquet")

    futures_path = directory / f"{name}_futures.npy"
    futures = np.load(futures_path) if futures_path.exists() else None

    return df, futures
