"""
cli_build.py — CLI entry point: build a numerical FAISS retrieval database.

Usage:
    python -m retrieval.cli_build --config configs/retrieval/demo.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from .build_db import build_and_save, slice_windows
from .statistical_encoder import StatisticalWindowEncoder


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_data(cfg: dict) -> dict[str, np.ndarray]:
    """Load time series data.  Supports CSV or 'synthetic' mode."""
    data_cfg = cfg["data"]

    if data_cfg.get("source") == "synthetic":
        np.random.seed(data_cfg.get("seed", 42))
        n_series = data_cfg.get("n_series", 3)
        length = data_cfg.get("length", 1000)
        series = {}
        for i in range(n_series):
            t = np.arange(length, dtype=np.float32)
            freq = 0.01 + 0.005 * i
            series[f"synthetic_{i}"] = np.sin(freq * t) + 0.1 * np.random.randn(length).astype(np.float32)
        return series

    # CSV mode
    import pandas as pd

    csv_path = data_cfg["csv_path"]
    df = pd.read_csv(csv_path)
    series = {}
    columns = data_cfg.get("columns", None)
    if columns is None:
        # Use all numeric columns except the first (often a date column)
        columns = [c for c in df.columns[1:] if df[c].dtype in ("float64", "float32", "int64")]
    for col in columns:
        series[col] = df[col].values.astype(np.float32)
    return series


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build FAISS retrieval database")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)

    enc_cfg = cfg.get("encoder", {})
    encoder = StatisticalWindowEncoder(eps=float(enc_cfg.get("eps", 1e-8)))

    # Load data
    series_dict = _load_data(cfg)

    # Slice windows
    data_cfg = cfg["data"]
    lookback = data_cfg["lookback_length"]
    pred_len = data_cfg["prediction_length"]
    stride = data_cfg.get("stride", 1)

    all_samples = []
    for sid, values in series_dict.items():
        samples = slice_windows(
            values,
            lookback_length=lookback,
            prediction_length=pred_len,
            stride=stride,
            series_id=sid,
            metadata={"source": data_cfg.get("source", "csv")},
        )
        all_samples.extend(samples)
        print(f"  {sid}: {len(samples)} windows")

    print(f"Total windows: {len(all_samples)}")

    # Build and save
    ret_cfg = cfg.get("retrieval", {})
    output_dir = cfg.get("output_dir", "output/retrieval_db")

    build_and_save(
        samples=all_samples,
        encoder=encoder,
        output_dir=output_dir,
        name=cfg.get("db_name", "db"),
        metric=ret_cfg.get("metric", "cosine"),
        encode_batch_size=enc_cfg.get("batch_size", 256),
    )


if __name__ == "__main__":
    main()
