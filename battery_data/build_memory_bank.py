from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from retrieval.build_db import build_and_save
from retrieval.search import RetrieverSearcher

from .canonicalize import assign_cell_uids, combine_canonical_cycles, load_enabled_cells
from .features import serialize_json
from .splits import assert_no_split_leakage, build_split_manifest
from .windowing import build_memory_samples


def _default_db_specs(cfg: Dict[str, object]) -> List[Dict[str, object]]:
    db_specs = cfg.get("vector_dbs")
    if db_specs:
        return list(db_specs)
    return [{"name": cfg.get("db_name", "source_bank"), "include_splits": ["source_train"]}]


def build_dataset_summary(
    canonical_cycles: pd.DataFrame,
    split_manifest: pd.DataFrame,
    memory_df: pd.DataFrame,
) -> Dict[str, object]:
    missing_ratios = {}
    for column in canonical_cycles.columns:
        if column in {"cell_uid", "cycle_idx", "timestamp"}:
            continue
        series = canonical_cycles[column]
        if series.dtype == object:
            missing = series.isna() | (series.astype(str) == "")
        else:
            missing = series.isna()
        missing_ratios[column] = float(missing.mean())

    window_counts = memory_df.groupby("split").size().to_dict() if not memory_df.empty else {}
    cell_counts = split_manifest.groupby("split").size().to_dict()
    domain_counts = split_manifest.groupby("domain_label").size().to_dict()
    return {
        "cells_per_domain": domain_counts,
        "cells_per_split": cell_counts,
        "windows_per_split": window_counts,
        "field_missing_ratio": missing_ratios,
    }


def run_validation_search(
    output_dir: str | Path,
    db_name: str,
    encoder,
    memory_samples,
    top_k: int,
    metric: str = "cosine",
) -> Optional[Dict[str, object]]:
    target_queries = [sample for sample in memory_samples if sample.split == "target_query"]
    if not target_queries:
        return None

    query_sample = target_queries[0]
    searcher = RetrieverSearcher(
        db_dir=output_dir,
        name=db_name,
        metric=metric,
        encoder=encoder,
    )
    results = searcher.search_by_window(
        np.asarray([query_sample.window_tokens], dtype=np.float32),
        top_k=top_k,
    )
    result = results[0]
    neighbor_metadata = result.neighbor_metadata
    return {
        "query_cell_uid": query_sample.cell_uid,
        "top_k_distance": result.distances.tolist(),
        "neighbor_d_i": [meta.get("d_i") for meta in neighbor_metadata],
        "neighbor_m_i": [meta.get("m_i") for meta in neighbor_metadata],
        "neighbor_delta_y": [arr.tolist() for arr in result.neighbor_future_values] if result.neighbor_future_values is not None else [],
        "target_query_in_index": query_sample.cell_uid in set(searcher.meta_df["series_id"]),
    }


def build_battery_memory_bank(
    cfg: Dict[str, object],
    encoder,
    run_search_validation: bool = True,
) -> Dict[str, object]:
    output_dir = Path(cfg.get("output_dir", "output/battery_memory_bank"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = assign_cell_uids(
        load_enabled_cells(cfg),
        prefix=str(cfg.get("cell_uid_prefix", "cell")),
    )
    if not cells:
        raise ValueError("No cells were loaded; check dataset roots and config.")

    split_manifest = build_split_manifest(
        cells,
        cfg.get("split", {}),
        cfg.get("domain_labeling"),
    )
    assert_no_split_leakage(split_manifest)

    canonical_cycles = combine_canonical_cycles(cells)
    canonical_path = output_dir / "canonical_cycles.parquet"
    canonical_cycles.to_parquet(canonical_path, index=False)

    memory_samples, memory_df = build_memory_samples(
        canonical_cycles,
        split_manifest,
        cfg.get("memory", {}),
        cfg.get("domain_labeling"),
    )
    memory_path = output_dir / "memory_samples.parquet"
    memory_df.to_parquet(memory_path, index=False)

    split_path = output_dir / "split_manifest.csv"
    split_manifest.to_csv(split_path, index=False)

    db_outputs = []
    metric = cfg.get("retrieval", {}).get("metric", "cosine")
    batch_size = int(cfg.get("encoder", {}).get("batch_size", 256))
    for db_spec in _default_db_specs(cfg):
        include_splits = set(db_spec.get("include_splits", ["source_train"]))
        selected_samples = [sample for sample in memory_samples if sample.split in include_splits]
        if not selected_samples:
            continue
        db_name = str(db_spec["name"])
        build_and_save(
            samples=[sample.to_window_sample() for sample in selected_samples],
            encoder=encoder,
            output_dir=output_dir,
            name=db_name,
            metric=metric,
            encode_batch_size=batch_size,
        )
        db_rows = memory_df[memory_df["split"].isin(include_splits)].reset_index(drop=True).copy()
        db_rows.insert(0, "db_row_id", np.arange(len(db_rows)))
        db_rows.to_parquet(output_dir / f"{db_name}_memory_rows.parquet", index=False)
        db_outputs.append({"name": db_name, "include_splits": sorted(include_splits), "rows": len(db_rows)})

    summary = build_dataset_summary(canonical_cycles, split_manifest, memory_df)
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    validation = None
    if run_search_validation and db_outputs:
        validation = run_validation_search(
            output_dir=output_dir,
            db_name=db_outputs[0]["name"],
            encoder=encoder,
            memory_samples=memory_samples,
            top_k=int(cfg.get("retrieval", {}).get("top_k", 3)),
            metric=metric,
        )
        if validation is not None:
            validation_path = output_dir / "validation_search.json"
            validation_path.write_text(json.dumps(validation, indent=2, ensure_ascii=True))

    return {
        "output_dir": str(output_dir),
        "canonical_cycles_path": str(canonical_path),
        "memory_samples_path": str(memory_path),
        "split_manifest_path": str(split_path),
        "dataset_summary_path": str(summary_path),
        "vector_dbs": db_outputs,
        "validation": validation,
        "checks": {
            "target_query_cells_in_index": bool(validation and validation["target_query_in_index"]),
            "source_target_overlap": False,
        },
    }
