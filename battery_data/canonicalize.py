from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .features import (
    build_missing_mask,
    bucket_charge_rate,
    bucket_nominal_capacity,
    bucket_temperature,
    bucket_voltage_window,
    serialize_json,
    stable_nominal_capacity,
)
from .schema import CANONICAL_COLUMNS, CANONICAL_NUMERIC_COLUMNS, CanonicalCell

# ---------------------------------------------------------------------------
# Adapter-level parquet cache
# ---------------------------------------------------------------------------
# Each adapter reads GB-scale raw CSVs and aggregates them into cycle-level
# DataFrames.  This is the dominant cost when running cli_retrieval_eval.
# We cache the result per-dataset so subsequent runs skip the raw parsing.
#
# Cache location: <dataset_root>/.canonical_cache/<fingerprint>.parquet
#   + a companion .json with per-cell source_info.
# The fingerprint is derived from the sorted list of raw file paths + their
# mtime, so the cache auto-invalidates when files change.
# ---------------------------------------------------------------------------

_CACHE_DIR_NAME = ".canonical_cache"


def _cache_fingerprint(root: Path, glob_pattern: str) -> str:
    """Deterministic hash of (sorted file paths + mtimes) under *root*."""
    entries = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith("."):
            entries.append(f"{p.relative_to(root)}:{p.stat().st_mtime_ns}")
    digest = hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]
    return digest


def _try_load_adapter_cache(
    root: Path,
    dataset_name: str,
) -> Optional[List[CanonicalCell]]:
    """Return cached CanonicalCells if a valid cache exists, else None."""
    cache_dir = root / _CACHE_DIR_NAME
    if not cache_dir.is_dir():
        return None
    # Find the most recent cache file for this dataset
    parquet_files = sorted(cache_dir.glob(f"{dataset_name}_*.parquet"))
    if not parquet_files:
        return None
    # Use the latest one
    parquet_path = parquet_files[-1]
    meta_path = parquet_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        df = pd.read_parquet(parquet_path)
        with open(meta_path) as f:
            cell_metas = json.load(f)
    except Exception:
        return None

    # Reconstruct CanonicalCell objects
    cells: List[CanonicalCell] = []
    for meta in cell_metas:
        cell_uid_or_id = meta["raw_cell_id"]
        cell_df = df[df["_raw_cell_id"] == cell_uid_or_id].drop(columns=["_raw_cell_id"]).reset_index(drop=True)
        cells.append(
            CanonicalCell(
                source_dataset=meta["source_dataset"],
                raw_cell_id=meta["raw_cell_id"],
                file_path=meta["file_path"],
                cycles=cell_df,
                source_info=meta.get("source_info", {}),
            )
        )
    return cells


def _save_adapter_cache(
    root: Path,
    dataset_name: str,
    cells: List[CanonicalCell],
) -> None:
    """Persist adapter output so next run can skip raw CSV parsing."""
    cache_dir = root / _CACHE_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Use timestamp as simple versioning
    tag = str(int(time.time()))
    parquet_path = cache_dir / f"{dataset_name}_{tag}.parquet"
    meta_path = parquet_path.with_suffix(".json")

    frames = []
    cell_metas = []
    for cell in cells:
        frame = cell.cycles.copy()
        frame["_raw_cell_id"] = cell.raw_cell_id
        frames.append(frame)
        cell_metas.append({
            "source_dataset": cell.source_dataset,
            "raw_cell_id": cell.raw_cell_id,
            "file_path": cell.file_path,
            "source_info": cell.source_info,
        })

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_parquet(parquet_path, index=False)
    else:
        pd.DataFrame().to_parquet(parquet_path, index=False)

    with open(meta_path, "w") as f:
        json.dump(cell_metas, f, ensure_ascii=True)

    # Clean up older cache files for this dataset (keep only latest)
    for old in sorted(cache_dir.glob(f"{dataset_name}_*.parquet")):
        if old != parquet_path:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)


def finalize_canonical_cell_frame(
    base_df: pd.DataFrame,
    static_metadata: Dict[str, object],
    cfg: Dict[str, object] | None = None,
) -> pd.DataFrame:
    cfg = cfg or {}
    df = base_df.copy()
    if "cycle_idx" not in df.columns:
        raise ValueError("Each adapter must provide a cycle_idx column.")

    df = df.sort_values("cycle_idx").reset_index(drop=True)

    for col in CANONICAL_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    if "timestamp" not in df.columns:
        df["timestamp"] = None

    nominal_capacity = stable_nominal_capacity(
        df["capacity"].values,
        skip_initial_cycles=int(cfg.get("soh_skip_initial_cycles", 0)),
        stable_window=int(cfg.get("nominal_capacity_window", 5)),
    )
    nominal_hint = static_metadata.get("nominal_capacity_hint")
    if not np.isfinite(nominal_capacity) and nominal_hint is not None:
        nominal_capacity = float(nominal_hint)

    if "soh" not in df.columns or df["soh"].isna().all():
        if np.isfinite(nominal_capacity) and nominal_capacity > 0:
            df["soh"] = df["capacity"] / nominal_capacity
        else:
            df["soh"] = np.nan

    if df["charge_throughput"].isna().all():
        df["charge_throughput"] = df["capacity"].fillna(0).cumsum()
    if df["discharge_throughput"].isna().all():
        df["discharge_throughput"] = df["capacity"].fillna(0).cumsum()

    charge_rate_value = static_metadata.get("charge_rate_c")
    if charge_rate_value is None and np.isfinite(nominal_capacity) and nominal_capacity > 0:
        charge_current = df["current_max"].dropna()
        if not charge_current.empty:
            charge_rate_value = float(charge_current.abs().median() / nominal_capacity)

    temp_bucket = static_metadata.get("temperature_bucket")
    if temp_bucket is None:
        temp_bucket = bucket_temperature(df["temp_mean"].dropna().median() if df["temp_mean"].notna().any() else None)

    voltage_min_hint = static_metadata.get("voltage_min_hint")
    voltage_max_hint = static_metadata.get("voltage_max_hint")
    v_min = voltage_min_hint
    if v_min is None and df["voltage_min"].notna().any():
        v_min = float(df["voltage_min"].min())
    v_max = voltage_max_hint
    if v_max is None and df["voltage_max"].notna().any():
        v_max = float(df["voltage_max"].max())

    repeated_meta = {
        "chemistry_family": static_metadata.get("chemistry_family"),
        "temperature_bucket": temp_bucket,
        "charge_rate_bucket": static_metadata.get("charge_rate_bucket") or bucket_charge_rate(charge_rate_value),
        "discharge_policy_family": static_metadata.get("discharge_policy_family"),
        "full_or_partial": static_metadata.get("full_or_partial"),
        "nominal_capacity_bucket": static_metadata.get("nominal_capacity_bucket") or bucket_nominal_capacity(nominal_capacity),
        "voltage_window_bucket": static_metadata.get("voltage_window_bucket") or bucket_voltage_window(v_min, v_max),
    }
    for key, value in repeated_meta.items():
        df[key] = value

    df["cell_uid"] = ""
    df["missing_mask"] = df.apply(lambda row: serialize_json(build_missing_mask(row)), axis=1)
    return df[CANONICAL_COLUMNS]


def load_enabled_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    from .adapters import ADAPTER_REGISTRY

    datasets_cfg = cfg.get("datasets", {})
    use_cache = cfg.get("adapter_cache", True)  # enabled by default
    all_cells: List[CanonicalCell] = []
    for dataset_name, dataset_cfg in datasets_cfg.items():
        if not dataset_cfg or not dataset_cfg.get("enabled", True):
            continue
        if dataset_name not in ADAPTER_REGISTRY:
            raise KeyError(f"Unsupported dataset adapter: {dataset_name}")

        root = Path(dataset_cfg.get("root", ""))

        # Try loading from parquet cache first
        if use_cache and root.is_dir():
            cached = _try_load_adapter_cache(root, dataset_name)
            if cached is not None:
                print(f"  [{dataset_name}] Loaded {len(cached)} cells from cache (skipped raw CSV parsing)", flush=True)
                all_cells.extend(cached)
                continue

        # Fall back to full adapter (slow path: reads raw CSVs)
        t0 = time.time()
        cells = ADAPTER_REGISTRY[dataset_name](dataset_cfg)
        elapsed = time.time() - t0
        print(f"  [{dataset_name}] Parsed {len(cells)} cells from raw data in {elapsed:.1f}s", flush=True)

        # Save cache for next time
        if use_cache and root.is_dir() and cells:
            _save_adapter_cache(root, dataset_name, cells)
            print(f"  [{dataset_name}] Saved adapter cache to {root / _CACHE_DIR_NAME}/", flush=True)

        all_cells.extend(cells)
    return all_cells


def assign_cell_uids(cells: Iterable[CanonicalCell], prefix: str = "cell") -> List[CanonicalCell]:
    sorted_cells = sorted(cells, key=lambda cell: (cell.source_dataset, cell.raw_cell_id))
    for idx, cell in enumerate(sorted_cells, start=1):
        cell_uid = f"{prefix}_{idx:05d}"
        cell.cycles = cell.cycles.copy()
        cell.cycles["cell_uid"] = cell_uid
        cell.source_info = dict(cell.source_info)
        cell.source_info["cell_uid"] = cell_uid
    return sorted_cells


def combine_canonical_cycles(cells: Iterable[CanonicalCell]) -> pd.DataFrame:
    frames = []
    for cell in cells:
        frame = cell.cycles.copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    return combined[CANONICAL_COLUMNS]
