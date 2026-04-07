from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .retrieval_eval import choose_db_spec, run_retrieval_visual_evaluation


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_encoder_from_config(cfg: dict):
    from retrieval.encoder_chronos2 import Chronos2RetrieverEncoder

    enc_cfg = cfg["encoder"]
    return Chronos2RetrieverEncoder(
        model_path=enc_cfg.get("model_path", "autogluon/chronos-2"),
        pooling=enc_cfg.get("pooling", "mean"),
        device=enc_cfg.get("device", "auto"),
        context_length=enc_cfg.get("context_length"),
        batch_size=enc_cfg.get("batch_size", 256),
        torch_dtype=enc_cfg.get("torch_dtype", "float32"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate battery retrieval and plot top-3 neighbors")
    parser.add_argument("--config", type=str, required=True, help="Path to battery YAML config")
    parser.add_argument("--db_name", type=str, default=None, help="Vector DB name from config")
    parser.add_argument("--query_split", type=str, default="target_query", help="Preferred query split")
    parser.add_argument("--eval_split", type=str, default=None, help="Preferred split for hit-rate evaluation")
    parser.add_argument("--query_index", type=int, default=-1, help="Index within the chosen query split; -1 = auto-select")
    parser.add_argument("--top_k", type=int, default=3, help="Number of neighbors to retrieve")
    parser.add_argument(
        "--series_columns",
        type=str,
        default="soh,voltage_mean,temp_mean,current_mean",
        help="Comma-separated canonical series columns to compare and plot",
    )
    parser.add_argument(
        "--series_column",
        type=str,
        default=None,
        help="Backward-compatible alias for a single canonical series column",
    )
    parser.add_argument(
        "--max_eval_queries",
        type=int,
        default=0,
        help="Max number of queries for hit-rate evaluation; 0 skips the extra metric pass",
    )
    parser.add_argument("--plot_context_length", type=int, default=64, help="How many historical cycles to show in each subplot")
    parser.add_argument("--min_query_cycle_idx", type=int, default=50, help="Prefer query windows ending after this cycle index")
    parser.add_argument(
        "--display_mode",
        type=str,
        default="zscore",
        choices=["raw", "delta", "relative", "zscore"],
        help="Plot transformation mode",
    )
    parser.add_argument("--allow_same_series_neighbors", action="store_true", help="Allow multiple top-k neighbors from the same cell")
    args = parser.parse_args(argv)

    if args.series_column:
        series_columns = [args.series_column]
    else:
        series_columns = [item.strip() for item in args.series_columns.split(",") if item.strip()]

    cfg = load_config(args.config)
    db_spec = choose_db_spec(cfg, db_name=args.db_name)
    output_dir = Path(cfg.get("output_dir", "output/battery_memory_bank"))
    all_memory_embeddings_path = output_dir / "all_memory_embeddings.npy"
    offline_ready = all(
        path.exists()
        for path in [
            output_dir / "canonical_cycles.parquet",
            output_dir / "split_manifest.csv",
            output_dir / f"{db_spec['name']}.faiss",
            output_dir / f"{db_spec['name']}_meta.parquet",
            output_dir / f"{db_spec['name']}_futures.npy",
            output_dir / f"{db_spec['name']}_embeddings.npy",
        ]
    )

    encoder = None
    # Even when the FAISS DB already exists, we still need an encoder for
    # out-of-index queries (e.g. target_query) if all_memory_embeddings.npy
    # is unavailable.
    need_encoder = (not offline_ready) or (not all_memory_embeddings_path.exists()) or (args.max_eval_queries > 0)
    if need_encoder:
        reason = []
        if not offline_ready:
            reason.append("retrieval artifacts are incomplete")
        if not all_memory_embeddings_path.exists():
            reason.append("all_memory_embeddings.npy is missing")
        if args.max_eval_queries > 0:
            reason.append("evaluation queries require on-the-fly encoding")
        print(f"[EvalCLI] Loading encoder because {', '.join(reason)}.", flush=True)
        encoder = build_encoder_from_config(cfg)

    result = run_retrieval_visual_evaluation(
        cfg=cfg,
        encoder=encoder,
        db_name=args.db_name,
        query_split=args.query_split,
        query_index=args.query_index,
        top_k=args.top_k,
        series_columns=series_columns,
        eval_split=args.eval_split,
        max_eval_queries=args.max_eval_queries,
        plot_context_length=args.plot_context_length,
        min_query_cycle_idx=args.min_query_cycle_idx,
        unique_neighbor_series=not args.allow_same_series_neighbors,
        display_mode=args.display_mode,
    )

    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
