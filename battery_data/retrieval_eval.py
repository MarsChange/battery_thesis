from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_cache_root = Path(tempfile.gettempdir()) / "battery_ts_rag_plot_cache"
_cache_root.mkdir(parents=True, exist_ok=True)
(_cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retrieval.search import RetrieverSearcher

from .build_memory_bank import build_battery_memory_bank
from .schema import BatteryMemorySample
from .windowing import build_memory_samples

DEFAULT_SERIES_COLUMNS = ["soh", "voltage_mean", "temp_mean", "current_mean"]


def ensure_memory_bank_ready(
    cfg: Dict[str, object],
    encoder,
    db_name: str,
) -> Path:
    output_dir = Path(cfg.get("output_dir", "output/battery_memory_bank"))
    required = [
        output_dir / "canonical_cycles.parquet",
        output_dir / "split_manifest.csv",
        output_dir / f"{db_name}.faiss",
        output_dir / f"{db_name}_meta.parquet",
        output_dir / f"{db_name}_futures.npy",
        output_dir / f"{db_name}_embeddings.npy",
    ]
    if all(path.exists() for path in required):
        print(f"[Eval] Reusing existing retrieval artifacts for '{db_name}' in {output_dir}", flush=True)
        return output_dir
    if encoder is None:
        raise ValueError(
            "Memory-bank artifacts are incomplete and no encoder was provided. "
            "Build the bank first or rerun with an available Chronos-2 encoder."
        )
    print(
        f"[Eval] Retrieval artifacts for '{db_name}' are incomplete. "
        "Building memory bank first...",
        flush=True,
    )
    eval_cfg = dict(cfg)
    eval_cfg["save_all_memory_embeddings"] = False
    build_battery_memory_bank(cfg=eval_cfg, encoder=encoder, run_search_validation=False)
    return output_dir


def load_memory_samples_from_artifacts(
    cfg: Dict[str, object],
    output_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, List[BatteryMemorySample], pd.DataFrame]:
    output_dir = Path(output_dir)
    canonical_cycles = pd.read_parquet(output_dir / "canonical_cycles.parquet")
    split_manifest = pd.read_csv(output_dir / "split_manifest.csv")
    memory_samples, memory_df = build_memory_samples(
        canonical_cycles=canonical_cycles,
        split_manifest=split_manifest,
        memory_cfg=cfg.get("memory", {}),
        domain_rules=cfg.get("domain_labeling"),
    )
    return canonical_cycles, split_manifest, memory_samples, memory_df


def choose_db_spec(cfg: Dict[str, object], db_name: Optional[str] = None) -> Dict[str, object]:
    db_specs = list(cfg.get("vector_dbs", []))
    if not db_specs:
        db_specs = [{"name": cfg.get("db_name", "source_bank"), "include_splits": ["source_train"]}]
    if db_name is None:
        return db_specs[0]
    for spec in db_specs:
        if spec["name"] == db_name:
            return spec
    raise KeyError(f"Database spec {db_name!r} not found in config.")


def make_sample_key(sample: BatteryMemorySample) -> tuple[str, int, int, int, int]:
    return (
        sample.cell_uid,
        sample.window_start,
        sample.window_end,
        sample.target_start,
        sample.target_end,
    )


def select_query_samples(
    memory_samples: Sequence[BatteryMemorySample],
    preferred_split: str,
) -> tuple[str, List[BatteryMemorySample]]:
    fallback_order = [preferred_split, "target_query", "source_val", "source_train", "target_support"]
    seen = set()
    for split in fallback_order:
        if split in seen:
            continue
        seen.add(split)
        samples = [sample for sample in memory_samples if sample.split == split]
        if samples:
            return split, samples
    raise ValueError("No queryable memory samples found in any split.")


def select_representative_query_sample(
    query_candidates: Sequence[BatteryMemorySample],
    canonical_cycles: pd.DataFrame,
    series_columns: Sequence[str],
    min_cycle_idx: int = 50,
) -> int:
    best_idx = 0
    best_score = float("-inf")

    for idx, sample in enumerate(query_candidates):
        cell_df = canonical_cycles.loc[canonical_cycles["cell_uid"] == sample.cell_uid].sort_values("cycle_idx").reset_index(drop=True)
        segment = cell_df.iloc[sample.window_start : sample.window_end]
        if segment.empty:
            continue
        end_cycle = int(segment["cycle_idx"].iloc[-1])
        if end_cycle < min_cycle_idx:
            continue
        score = 0.0
        for series_column in series_columns:
            values = pd.to_numeric(segment[series_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if values.size == 0:
                continue
            variation = float(np.nanstd(values))
            span = float(np.nanmax(values) - np.nanmin(values)) if values.size else 0.0
            slope = float(abs(values[-1] - values[0])) if values.size else 0.0
            score += variation + span + slope
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def distance_to_similarity(metric: str, distances: np.ndarray) -> np.ndarray:
    if metric == "l2":
        return 1.0 / (1.0 + np.asarray(distances, dtype=np.float32))
    return np.asarray(distances, dtype=np.float32)


def get_series_from_sample(
    sample: BatteryMemorySample,
    canonical_cycles: pd.DataFrame,
    series_column: str,
    context_length: Optional[int] = None,
    mode: str = "raw",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    cell_df = canonical_cycles.loc[canonical_cycles["cell_uid"] == sample.cell_uid].sort_values("cycle_idx").reset_index(drop=True)
    context_start = sample.window_start
    if context_length is not None and context_length > 0:
        context_start = max(0, sample.window_end - context_length)
    segment = cell_df.iloc[context_start : sample.window_end]
    x = segment["cycle_idx"].to_numpy(dtype=np.int32)
    y = pd.to_numeric(segment[series_column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if y.size:
        base = float(y[0])
        if mode == "delta":
            y = y - base
        elif mode == "relative":
            denom = base if abs(base) > 1e-8 else 1.0
            y = y / denom - 1.0
    retrieval_start_pos = sample.window_start - context_start
    retrieval_end_pos = sample.window_end - context_start
    return x, y, (retrieval_start_pos, retrieval_end_pos)


def get_multivariate_series_from_sample(
    sample: BatteryMemorySample,
    canonical_cycles: pd.DataFrame,
    series_columns: Sequence[str],
    context_length: Optional[int] = None,
    mode: str = "raw",
) -> tuple[np.ndarray, Dict[str, np.ndarray], tuple[int, int]]:
    x = None
    series_map: Dict[str, np.ndarray] = {}
    highlight = (0, 0)
    for column in series_columns:
        sx, sy, sh = get_series_from_sample(
            sample,
            canonical_cycles,
            column,
            context_length=context_length,
            mode=mode,
        )
        if x is None:
            x = sx
            highlight = sh
        series_map[column] = sy
    if x is None:
        x = np.zeros((0,), dtype=np.int32)
    return x, series_map, highlight


def compute_curve_shape_similarity(
    query_sample: BatteryMemorySample,
    neighbor_sample: BatteryMemorySample,
    canonical_cycles: pd.DataFrame,
    series_column: str,
) -> float:
    _, q_y, _ = get_series_from_sample(query_sample, canonical_cycles, series_column, context_length=query_sample.window_end - query_sample.window_start, mode="raw")
    _, n_y, _ = get_series_from_sample(neighbor_sample, canonical_cycles, series_column, context_length=neighbor_sample.window_end - neighbor_sample.window_start, mode="raw")
    if q_y.size == 0 or n_y.size == 0:
        return float("nan")
    length = min(len(q_y), len(n_y))
    q = q_y[-length:].astype(np.float32)
    n = n_y[-length:].astype(np.float32)
    q = q - q.mean()
    n = n - n.mean()
    q_norm = float(np.linalg.norm(q))
    n_norm = float(np.linalg.norm(n))
    if q_norm < 1e-8 or n_norm < 1e-8:
        return float("nan")
    return float(np.dot(q, n) / (q_norm * n_norm))


def compute_multivariate_shape_similarity(
    query_sample: BatteryMemorySample,
    neighbor_sample: BatteryMemorySample,
    canonical_cycles: pd.DataFrame,
    series_columns: Sequence[str],
) -> tuple[float, Dict[str, float]]:
    feature_scores: Dict[str, float] = {}
    for column in series_columns:
        feature_scores[column] = compute_curve_shape_similarity(
            query_sample=query_sample,
            neighbor_sample=neighbor_sample,
            canonical_cycles=canonical_cycles,
            series_column=column,
        )
    valid_scores = [score for score in feature_scores.values() if not np.isnan(score)]
    if not valid_scores:
        return 0.0, feature_scores
    return float(np.mean(valid_scores)), feature_scores


def compute_tsrag_style_l2_distance(
    query_embedding: np.ndarray,
    neighbor_embedding: np.ndarray,
) -> float:
    q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    n = np.asarray(neighbor_embedding, dtype=np.float32).reshape(-1)
    return float(np.linalg.norm(q - n, ord=2))


def filter_neighbor_rows(
    result,
    top_k: int,
    unique_series: bool,
) -> tuple[np.ndarray, np.ndarray, List[dict]]:
    keep_indices = []
    seen_series = set()
    candidates = list(zip(range(len(result.neighbor_ids)), result.neighbor_ids, result.neighbor_series_ids))

    for idx, db_id, sid in candidates:
        if db_id < 0:
            continue
        if unique_series and sid in seen_series:
            continue
        seen_series.add(sid)
        keep_indices.append(idx)
        if len(keep_indices) >= top_k:
            break

    if len(keep_indices) < top_k:
        for idx, db_id, _sid in candidates:
            if db_id < 0 or idx in keep_indices:
                continue
            keep_indices.append(idx)
            if len(keep_indices) >= top_k:
                break

    keep_indices = np.asarray(keep_indices, dtype=np.int32)
    return keep_indices, result.distances[keep_indices], [result.neighbor_metadata[i] for i in keep_indices]


def evaluate_retrieval_hits(
    searcher: RetrieverSearcher,
    query_samples: Sequence[BatteryMemorySample],
    db_id_by_key: Dict[tuple[str, int, int, int, int], int],
    memory_index_by_key: Dict[tuple[str, int, int, int, int], int],
    all_memory_embeddings: Optional[np.ndarray],
    encoder,
    top_k: int,
    metric: str,
    max_queries: Optional[int] = None,
) -> Dict[str, object]:
    if max_queries is not None and max_queries <= 0:
        return {
            "evaluated_queries": 0,
            "top1_domain_hit_rate": 0.0,
            f"top{top_k}_domain_hit_rate": 0.0,
            "mean_reciprocal_rank_domain": 0.0,
        }

    top1_hits = 0
    topk_hits = 0
    reciprocal_ranks = []
    total = 0

    eval_samples = list(query_samples)
    if max_queries is not None:
        eval_samples = eval_samples[: max_queries]

    for sample in eval_samples:
        drop_self_ids = None
        key = make_sample_key(sample)
        if key in db_id_by_key:
            drop_self_ids = {db_id_by_key[key]}
        if all_memory_embeddings is not None and key in memory_index_by_key:
            query_emb = all_memory_embeddings[memory_index_by_key[key]]
            result = searcher.search_by_embedding(
                np.asarray(query_emb, dtype=np.float32),
                top_k=top_k,
                drop_self_ids=drop_self_ids,
            )[0]
        else:
            if encoder is None:
                raise ValueError("Query embeddings are unavailable and no encoder was provided for evaluation.")
            result = searcher.search_by_window(
                np.asarray([sample.window_tokens], dtype=np.float32),
                top_k=top_k,
                drop_self_ids=drop_self_ids,
            )[0]
        neighbor_labels = [meta.get("d_i") for meta in result.neighbor_metadata]
        total += 1
        if neighbor_labels and neighbor_labels[0] == sample.domain_label:
            top1_hits += 1
        if sample.domain_label in neighbor_labels:
            topk_hits += 1
            rank = neighbor_labels.index(sample.domain_label) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        "evaluated_queries": total,
        "top1_domain_hit_rate": float(top1_hits / total) if total else 0.0,
        f"top{top_k}_domain_hit_rate": float(topk_hits / total) if total else 0.0,
        "mean_reciprocal_rank_domain": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def plot_query_and_neighbors(
    canonical_cycles: pd.DataFrame,
    query_sample: BatteryMemorySample,
    neighbor_samples: Sequence[BatteryMemorySample],
    index_scores: Sequence[float],
    tsrag_l2_scores: Sequence[float],
    multivariate_shape_scores: Sequence[float],
    output_path: str | Path,
    series_columns: Sequence[str],
    plot_context_length: int = 64,
    display_mode: str = "zscore",
) -> str:
    output_path = str(output_path)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.flatten()

    color_map = {
        "soh": "#1f77b4",
        "voltage_mean": "#ff7f0e",
        "temp_mean": "#2ca02c",
        "current_mean": "#9467bd",
        "capacity": "#8c564b",
    }

    query_x, query_series_map, query_highlight = get_multivariate_series_from_sample(
        query_sample,
        canonical_cycles,
        series_columns,
        context_length=plot_context_length,
        mode="raw",
    )
    neighbor_plot_data = []
    all_y = []
    for series in query_series_map.values():
        if len(series):
            all_y.append(_normalize_for_plot(series, display_mode))
    for neighbor_sample in neighbor_samples:
        x, series_map, highlight = get_multivariate_series_from_sample(
            neighbor_sample,
            canonical_cycles,
            series_columns,
            context_length=plot_context_length,
            mode="raw",
        )
        neighbor_plot_data.append((x, series_map, highlight))
        for series in series_map.values():
            if len(series):
                all_y.append(_normalize_for_plot(series, display_mode))

    if all_y:
        y_min = min(float(np.min(y)) for y in all_y if len(y))
        y_max = max(float(np.max(y)) for y in all_y if len(y))
        y_pad = 0.08 * max(y_max - y_min, 1.0)
    else:
        y_min, y_max, y_pad = -1.0, 1.0, 0.2

    _plot_multivariate_subplot(
        ax=axes[0],
        x=query_x,
        series_map=query_series_map,
        highlight=query_highlight,
        title=f"Query | {query_sample.cell_uid} | {query_sample.domain_label}",
        series_columns=series_columns,
        color_map=color_map,
        display_mode=display_mode,
        y_lim=(y_min - y_pad, y_max + y_pad),
        show_legend=True,
    )

    for idx, (neighbor_sample, index_score, tsrag_l2_score, multivariate_shape_score, plot_data) in enumerate(
        zip(neighbor_samples, index_scores, tsrag_l2_scores, multivariate_shape_scores, neighbor_plot_data),
        start=1,
    ):
        ax = axes[idx]
        x, series_map, highlight = plot_data
        _plot_multivariate_subplot(
            ax=ax,
            x=x,
            series_map=series_map,
            highlight=highlight,
            title="Top-{} | {} | {}\nIndex={:.4f} | TSRAG-L2={:.4f} | MultiShape={:.4f}".format(
                idx,
                neighbor_sample.cell_uid,
                neighbor_sample.domain_label,
                index_score,
                tsrag_l2_score,
                multivariate_shape_score,
            ),
            series_columns=series_columns,
            color_map=color_map,
            display_mode=display_mode,
            y_lim=(y_min - y_pad, y_max + y_pad),
            show_legend=False,
        )

    fig.suptitle(
        f"Battery Retrieval Top-{len(neighbor_samples)} (features={','.join(series_columns)}, context={plot_context_length}, mode={display_mode})",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _normalize_for_plot(values: np.ndarray, display_mode: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).copy()
    if arr.size == 0:
        return arr
    if display_mode == "raw":
        return arr
    if display_mode == "delta":
        return arr - arr[0]
    if display_mode == "relative":
        denom = arr[0] if abs(arr[0]) > 1e-8 else 1.0
        return arr / denom - 1.0
    if display_mode == "zscore":
        mean = float(arr.mean())
        std = float(arr.std())
        if std < 1e-8:
            return arr - mean
        return (arr - mean) / std
    raise ValueError(f"Unsupported display_mode: {display_mode}")


def _plot_multivariate_subplot(
    ax,
    x: np.ndarray,
    series_map: Dict[str, np.ndarray],
    highlight: tuple[int, int],
    title: str,
    series_columns: Sequence[str],
    color_map: Dict[str, str],
    display_mode: str,
    y_lim: tuple[float, float],
    show_legend: bool,
) -> None:
    for column in series_columns:
        if column not in series_map:
            continue
        y = _normalize_for_plot(series_map[column], display_mode)
        if len(y) == 0:
            continue
        ax.plot(x, y, color=color_map.get(column), linewidth=2, label=column)
    if len(x):
        h_start = x[highlight[0]]
        h_end = x[min(highlight[1] - 1, len(x) - 1)]
        ax.axvspan(h_start, h_end, color="#9ecae1", alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel("Cycle Index")
    ax.set_ylabel(f"Normalized Value ({display_mode})")
    ax.grid(alpha=0.3)
    ax.set_ylim(*y_lim)
    if show_legend:
        ax.legend(loc="best", fontsize=8)


def run_retrieval_visual_evaluation(
    cfg: Dict[str, object],
    encoder,
    db_name: Optional[str] = None,
    query_split: str = "target_query",
    query_index: int = -1,
    top_k: int = 3,
    series_columns: Optional[Sequence[str]] = None,
    eval_split: Optional[str] = None,
    max_eval_queries: Optional[int] = 100,
    plot_context_length: int = 64,
    min_query_cycle_idx: int = 50,
    unique_neighbor_series: bool = True,
    display_mode: str = "zscore",
) -> Dict[str, object]:
    series_columns = list(series_columns or DEFAULT_SERIES_COLUMNS)
    print(
        f"[Eval] Starting retrieval evaluation with db={db_name or 'default'} "
        f"query_split={query_split} top_k={top_k} series={series_columns}",
        flush=True,
    )
    db_spec = choose_db_spec(cfg, db_name=db_name)
    db_name = str(db_spec["name"])
    include_splits = set(db_spec.get("include_splits", ["source_train"]))
    metric = cfg.get("retrieval", {}).get("metric", "cosine")

    output_dir = ensure_memory_bank_ready(cfg, encoder, db_name)
    canonical_cycles, split_manifest, memory_samples, memory_df = load_memory_samples_from_artifacts(cfg, output_dir)

    db_samples = [sample for sample in memory_samples if sample.split in include_splits]
    db_id_by_key = {make_sample_key(sample): idx for idx, sample in enumerate(db_samples)}
    memory_index_by_key = {make_sample_key(sample): idx for idx, sample in enumerate(memory_samples)}
    sample_by_db_id = {idx: sample for idx, sample in enumerate(db_samples)}
    embeddings_path = Path(output_dir) / "all_memory_embeddings.npy"
    all_memory_embeddings = np.load(embeddings_path) if embeddings_path.exists() else None
    db_embeddings_path = Path(output_dir) / f"{db_name}_embeddings.npy"
    db_embeddings = np.load(db_embeddings_path) if db_embeddings_path.exists() else None

    actual_query_split, query_candidates = select_query_samples(memory_samples, query_split)
    if query_index < 0:
        query_index = select_representative_query_sample(
            query_candidates,
            canonical_cycles,
            series_columns=series_columns,
            min_cycle_idx=min_query_cycle_idx,
        )
    query_index = max(0, min(query_index, len(query_candidates) - 1))
    query_sample = query_candidates[query_index]
    print(
        f"[Eval] Using query sample: split={actual_query_split} "
        f"cell={query_sample.cell_uid} window=({query_sample.window_start}, {query_sample.window_end})",
        flush=True,
    )

    searcher = RetrieverSearcher(
        db_dir=output_dir,
        name=db_name,
        metric=metric,
        encoder=encoder,
    )

    drop_self_ids = None
    query_key = make_sample_key(query_sample)
    if query_key in db_id_by_key:
        drop_self_ids = {db_id_by_key[query_key]}

    query_fetch_k = searcher.size if unique_neighbor_series else max(top_k * 8, top_k + 10)
    query_embedding = None
    if all_memory_embeddings is not None and query_key in memory_index_by_key:
        query_embedding = all_memory_embeddings[memory_index_by_key[query_key]]
    elif encoder is not None:
        query_embedding = encoder.encode(np.asarray([query_sample.window_tokens], dtype=np.float32))[0]

    if query_embedding is not None:
        result = searcher.search_by_embedding(
            np.asarray(query_embedding, dtype=np.float32),
            top_k=query_fetch_k,
            drop_self_ids=drop_self_ids,
        )[0]
    else:
        if encoder is None:
            raise ValueError("Query embeddings are unavailable and no encoder was provided for retrieval.")
        result = searcher.search_by_window(
            np.asarray([query_sample.window_tokens], dtype=np.float32),
            top_k=query_fetch_k,
            drop_self_ids=drop_self_ids,
        )[0]
    keep_indices, filtered_distances, filtered_meta = filter_neighbor_rows(
        result,
        top_k=top_k,
        unique_series=unique_neighbor_series,
    )
    neighbor_ids = result.neighbor_ids[keep_indices]
    retrieval_scores = distance_to_similarity(metric, filtered_distances)
    neighbor_samples = [sample_by_db_id[int(db_id)] for db_id in neighbor_ids]
    multivariate_shape_results = [
        compute_multivariate_shape_similarity(query_sample, neighbor_sample, canonical_cycles, series_columns)
        for neighbor_sample in neighbor_samples
    ]
    multivariate_shape_scores = [item[0] for item in multivariate_shape_results]
    per_feature_shape_scores = [item[1] for item in multivariate_shape_results]
    tsrag_l2_scores = []
    for db_id, neighbor_sample in zip(neighbor_ids, neighbor_samples):
        neighbor_key = make_sample_key(neighbor_sample)
        if query_embedding is not None and db_embeddings is not None and 0 <= int(db_id) < len(db_embeddings):
            neighbor_embedding = db_embeddings[int(db_id)]
            tsrag_l2_scores.append(compute_tsrag_style_l2_distance(query_embedding, neighbor_embedding))
        elif query_embedding is not None and all_memory_embeddings is not None and neighbor_key in memory_index_by_key:
            neighbor_embedding = all_memory_embeddings[memory_index_by_key[neighbor_key]]
            tsrag_l2_scores.append(compute_tsrag_style_l2_distance(query_embedding, neighbor_embedding))
        else:
            tsrag_l2_scores.append(float("nan"))

    figure_path = Path(output_dir) / f"{db_name}_{actual_query_split}_query_{query_index}_top{top_k}_{'_'.join(series_columns)}.png"
    plot_query_and_neighbors(
        canonical_cycles=canonical_cycles,
        query_sample=query_sample,
        neighbor_samples=neighbor_samples,
        index_scores=retrieval_scores,
        tsrag_l2_scores=tsrag_l2_scores,
        multivariate_shape_scores=multivariate_shape_scores,
        output_path=figure_path,
        series_columns=series_columns,
        plot_context_length=plot_context_length,
        display_mode=display_mode,
    )
    print(f"[Eval] Saved retrieval figure: {figure_path}", flush=True)

    actual_eval_split, eval_candidates = select_query_samples(memory_samples, eval_split or query_split)
    evaluation_metrics = evaluate_retrieval_hits(
        searcher=searcher,
        query_samples=eval_candidates,
        db_id_by_key=db_id_by_key,
        memory_index_by_key=memory_index_by_key,
        all_memory_embeddings=all_memory_embeddings,
        encoder=encoder,
        top_k=top_k,
        metric=metric,
        max_queries=max_eval_queries,
    )

    rows = []
    for rank, (db_id, dist, retrieval_score, tsrag_l2_score, multivariate_shape_score, feature_shape_score_map, meta, sample) in enumerate(
        zip(
            neighbor_ids,
            filtered_distances,
            retrieval_scores,
            tsrag_l2_scores,
            multivariate_shape_scores,
            per_feature_shape_scores,
            filtered_meta,
            neighbor_samples,
        ),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "db_row_id": int(db_id),
                "cell_uid": sample.cell_uid,
                "domain_label": sample.domain_label,
                "raw_distance": float(dist),
                "index_score": float(retrieval_score),
                "retrieval_similarity_score": float(retrieval_score),
                "tsrag_l2_distance": float(tsrag_l2_score) if not np.isnan(tsrag_l2_score) else None,
                "multivariate_shape_similarity": float(multivariate_shape_score),
                "shape_score": float(multivariate_shape_score),
                "per_feature_shape_similarity": {
                    key: (float(value) if not np.isnan(value) else None)
                    for key, value in feature_shape_score_map.items()
                },
                "metadata": meta.get("m_i"),
            }
        )

    summary = {
        "output_dir": str(output_dir),
        "db_name": db_name,
        "db_include_splits": sorted(include_splits),
        "retrieval_metric": metric,
        "query_split_requested": query_split,
        "query_split_used": actual_query_split,
        "eval_split_used": actual_eval_split,
        "query_index": query_index,
        "query_cell_uid": query_sample.cell_uid,
        "query_domain_label": query_sample.domain_label,
        "query_window": {
            "window_start": query_sample.window_start,
            "window_end": query_sample.window_end,
            "plot_context_length": plot_context_length,
            "min_query_cycle_idx": min_query_cycle_idx,
            "display_mode": display_mode,
            "series_columns": list(series_columns),
        },
        "figure_path": str(figure_path),
        "neighbors": rows,
        "evaluation_metrics": evaluation_metrics,
    }

    summary_path = Path(output_dir) / f"{db_name}_{actual_query_split}_query_{query_index}_retrieval_eval.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"[Eval] Saved retrieval summary: {summary_path}", flush=True)
    return summary
