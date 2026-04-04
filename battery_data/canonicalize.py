from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

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
    all_cells: List[CanonicalCell] = []
    for dataset_name, dataset_cfg in datasets_cfg.items():
        if not dataset_cfg or not dataset_cfg.get("enabled", True):
            continue
        if dataset_name not in ADAPTER_REGISTRY:
            raise KeyError(f"Unsupported dataset adapter: {dataset_name}")
        all_cells.extend(ADAPTER_REGISTRY[dataset_name](dataset_cfg))
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
