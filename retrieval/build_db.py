"""Build a FAISS retrieval database from numerically encoded time series windows.

This module is kept for legacy vector-index utilities. The main battery SOH
pipeline uses the case-bank retriever and interpretable distance components.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from .index import FAISSIndex
from .io import save_sidecar
from .schema import WindowSample


def slice_windows(
    values: np.ndarray,
    lookback_length: int,
    prediction_length: int,
    stride: int = 1,
    series_id: str = "series_0",
    metadata: Optional[Dict] = None,
) -> List[WindowSample]:
    """Slice a univariate series into overlapping WindowSamples.

    Parameters
    ----------
    values : np.ndarray, shape ``(T,)`` or ``(T, C)``
    lookback_length : int
    prediction_length : int
    stride : int
    series_id : str
    metadata : dict | None
    """
    metadata = metadata or {}
    T = len(values)
    samples: List[WindowSample] = []
    for start in range(0, T - lookback_length - prediction_length + 1, stride):
        w_end = start + lookback_length
        t_end = w_end + prediction_length
        samples.append(
            WindowSample(
                series_id=series_id,
                window_start=start,
                window_end=w_end,
                target_start=w_end,
                target_end=t_end,
                window_values=values[start:w_end],
                future_values=values[w_end:t_end],
                metadata=metadata,
            )
        )
    return samples


def build_database(
    samples: List[WindowSample],
    encoder: Any,
    metric: str = "cosine",
    encode_batch_size: int = 256,
) -> tuple[FAISSIndex, np.ndarray]:
    """Encode all samples and build a FAISS index.

    Returns
    -------
    index : FAISSIndex
    embeddings : np.ndarray, shape ``(N, D)``
    """
    if not samples:
        raise ValueError("Cannot build a database from an empty sample list.")

    print(
        f"[Retrieval] Building FAISS database for {len(samples)} samples "
        f"(metric={metric}, batch_size={encode_batch_size})",
        flush=True,
    )

    if hasattr(encoder, "embedding_dim"):
        try:
            dim = encoder.embedding_dim
            print(f"[Retrieval] Encoder ready (embedding_dim={dim}).", flush=True)
        except Exception:
            print("[Retrieval] Encoder initialization failed while probing embedding_dim.", flush=True)
            raise

    # Batch-encode without stacking the full dataset in memory.
    all_embeddings: List[np.ndarray] = []
    N = len(samples)
    for start in tqdm(range(0, N, encode_batch_size), desc="Encoding"):
        batch_samples = samples[start : start + encode_batch_size]
        batch = np.stack([s.window_values for s in batch_samples])
        emb = encoder.encode(batch)  # [B, D]
        all_embeddings.append(emb)
    embeddings = np.concatenate(all_embeddings, axis=0)  # [N, D]

    # Build FAISS index
    index = FAISSIndex(dim=embeddings.shape[1], metric=metric)
    index.add(embeddings)
    return index, embeddings


def build_and_save(
    samples: List[WindowSample],
    encoder: Any,
    output_dir: str | Path,
    name: str = "db",
    metric: str = "cosine",
    encode_batch_size: int = 256,
) -> None:
    """Full pipeline: encode → FAISS index → sidecar files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    index, embeddings = build_database(samples, encoder, metric, encode_batch_size)

    # Save FAISS index
    index.save(str(output_dir / f"{name}.faiss"))

    # Save raw embeddings (useful for debugging / re-indexing)
    np.save(output_dir / f"{name}_embeddings.npy", embeddings)

    # Save sidecar metadata + futures
    save_sidecar(samples, output_dir, name)

    print(f"Database saved to {output_dir}/  ({index.ntotal} vectors, dim={index.dim})")
