"""
cli_search.py — CLI entry point: query a FAISS retrieval database.

Usage:
    python -m retrieval.cli_search --config configs/retrieval/demo.yaml --top_k 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from .encoder_chronos2 import Chronos2RetrieverEncoder
from .search import RetrieverSearcher


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Search FAISS retrieval database")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--top_k", type=int, default=None, help="Override top_k from config")
    parser.add_argument("--query_idx", type=int, default=None, help="Index of window to use as query (random if unset)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    top_k = args.top_k or cfg.get("retrieval", {}).get("top_k", 5)

    # Build encoder
    enc_cfg = cfg["encoder"]
    encoder = Chronos2RetrieverEncoder(
        model_path=enc_cfg.get("model_path", "autogluon/chronos-2"),
        pooling=enc_cfg.get("pooling", "mean"),
        device=enc_cfg.get("device", "auto"),
        context_length=enc_cfg.get("context_length"),
        batch_size=enc_cfg.get("batch_size", 256),
        torch_dtype=enc_cfg.get("torch_dtype", "float32"),
    )

    # Load searcher
    output_dir = cfg.get("output_dir", "output/retrieval_db")
    db_name = cfg.get("db_name", "db")
    metric = cfg.get("retrieval", {}).get("metric", "cosine")

    searcher = RetrieverSearcher(
        db_dir=output_dir,
        name=db_name,
        metric=metric,
        encoder=encoder,
    )
    print(f"Database loaded: {searcher.size} vectors")

    # Pick a query window
    embeddings_path = Path(output_dir) / f"{db_name}_embeddings.npy"
    all_emb = np.load(embeddings_path)

    if args.query_idx is not None:
        q_idx = args.query_idx
    else:
        np.random.seed(42)
        q_idx = np.random.randint(0, len(all_emb))

    print(f"\nQuery: window index {q_idx}")
    print(f"  series_id  = {searcher.meta_df.iloc[q_idx]['series_id']}")
    print(f"  window     = [{searcher.meta_df.iloc[q_idx]['window_start']}, {searcher.meta_df.iloc[q_idx]['window_end']})")

    # Search (exclude self)
    results = searcher.search_by_embedding(
        all_emb[q_idx],
        top_k=top_k + 1,
        drop_self_ids={q_idx},
    )
    result = results[0]

    print(f"\nTop-{top_k} neighbors:")
    print(f"{'Rank':<6}{'DB_ID':<10}{'Series':<20}{'Window':<20}{'Distance':<12}")
    print("-" * 68)
    for rank in range(min(top_k, len(result.distances))):
        sid = result.neighbor_series_ids[rank]
        ws = result.neighbor_window_starts[rank]
        we = result.neighbor_window_ends[rank]
        dist = result.distances[rank]
        db_id = result.neighbor_ids[rank]
        print(f"{rank+1:<6}{db_id:<10}{sid:<20}[{ws}, {we}){'':<8}{dist:<12.6f}")

    if result.neighbor_future_values is not None and len(result.neighbor_future_values) > 0:
        fv = result.neighbor_future_values[0]
        print(f"\nTop-1 neighbor future horizon (first 10 values): {fv[:10]}")


if __name__ == "__main__":
    main()
