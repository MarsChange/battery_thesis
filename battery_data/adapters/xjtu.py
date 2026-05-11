from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.canonicalize import finalize_canonical_cell_frame
from battery_data.features import bucket_temperature
from battery_data.schema import CanonicalCell

XJTU_CHEMISTRY_FAMILY = "NCM"


def _should_exclude_xjtu_file(path: Path, cfg: Dict[str, object]) -> bool:
    """Return whether an XJTU cell should be skipped by user-configured rules.

    The XJTU dataset contains several protocol families.  Some downstream SOH
    forecasting experiments intentionally exclude protocol families whose
    per-cycle summary capacity is not directly comparable to ordinary full
    cycling samples.  The rule is explicit in config so the excluded cells are
    auditable instead of silently disappearing.
    """

    raw_cell_id = path.stem.replace("_summary", "")
    path_text = str(path)
    for token in cfg.get("exclude_file_path_contains", []) or []:
        if str(token) and str(token) in path_text:
            return True
    for prefix in cfg.get("exclude_raw_cell_id_prefixes", []) or []:
        if raw_cell_id.startswith(str(prefix)):
            return True
    for pattern in cfg.get("exclude_raw_cell_id_patterns", []) or []:
        if fnmatch(raw_cell_id, str(pattern)):
            return True
    return False


def _infer_xjtu_metadata(path: Path, cfg: Dict[str, object]) -> Dict[str, object]:
    stem = path.stem.replace("_summary", "")
    condition = stem.split("_battery-")[0]
    charge_rate = None
    discharge_policy = "regular"
    full_or_partial = "full"

    if condition.endswith("C"):
        try:
            charge_rate = float(condition[:-1])
        except ValueError:
            charge_rate = None
    elif condition.startswith("R") and condition[1:].replace(".", "", 1).isdigit():
        charge_rate = float(condition[1:])
    elif condition == "RW":
        discharge_policy = "irregular"
        full_or_partial = "partial"
    elif condition == "Sim_satellite":
        discharge_policy = "satellite"
        full_or_partial = "partial"

    return {
        "chemistry_family": cfg.get("chemistry_family") or XJTU_CHEMISTRY_FAMILY,
        "temperature_bucket": cfg.get("temperature_bucket"),
        "charge_rate_c": charge_rate,
        "discharge_policy_family": discharge_policy,
        "full_or_partial": full_or_partial,
    }


def _load_xjtu_operation_frame(summary_path: Path) -> pd.DataFrame | None:
    data_path = summary_path.with_name(summary_path.name.replace("_summary.csv", "_data.csv"))
    if not data_path.exists():
        return None

    df = pd.read_csv(data_path, usecols=["cycle_index", "voltage_V", "current_A", "temperature_C"])
    if df.empty:
        return None
    for column in ["voltage_V", "current_A", "temperature_C"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    grouped = (
        df.groupby("cycle_index", sort=True)
        .agg(
            voltage_max=("voltage_V", "max"),
            voltage_min=("voltage_V", "min"),
            current_mean=("current_A", "mean"),
            current_abs_mean=("current_A", lambda values: values.abs().mean()),
            current_max=("current_A", "max"),
            current_min=("current_A", "min"),
            temp_mean=("temperature_C", "mean"),
            temp_max=("temperature_C", "max"),
            temp_min=("temperature_C", "min"),
        )
        .reset_index()
        .rename(columns={"cycle_index": "cycle_idx"})
    )
    return grouped


def load_xjtu_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    root = Path(cfg["root"])
    paths = sorted(root.glob("Batch-*/*_summary.csv"))
    paths = [path for path in paths if not _should_exclude_xjtu_file(path, cfg)]
    max_cells = cfg.get("max_cells")
    if max_cells:
        paths = paths[: int(max_cells)]

    cells: List[CanonicalCell] = []
    for path in paths:
        df = pd.read_csv(path)
        operation_frame = _load_xjtu_operation_frame(path)
        charge_capacity = df["charge_capacity_Ah"].astype(float)
        discharge_capacity = df["discharge_capacity_Ah"].astype(float)
        finite_initial = discharge_capacity.replace([np.inf, -np.inf], np.nan).dropna()
        initial_capacity = float(finite_initial.iloc[0]) if not finite_initial.empty else float("nan")
        if not np.isfinite(initial_capacity) or initial_capacity <= 0:
            initial_capacity = float("nan")
        # XJTU's first cycle is an initial-capacity measurement. In the summary
        # files, the first-cycle charge capacity can be a short pre-test charge
        # segment, while the first-cycle discharge capacity is the measured
        # initial capacity. For subsequent full-charge cycles, use charge
        # capacity as the SOH numerator as requested by the dataset protocol.
        soh_capacity = charge_capacity.copy()
        if np.isfinite(initial_capacity) and initial_capacity > 0 and len(soh_capacity) > 0:
            soh_capacity.iloc[0] = initial_capacity
        soh = soh_capacity / initial_capacity if np.isfinite(initial_capacity) and initial_capacity > 0 else np.nan
        base = pd.DataFrame(
            {
                "cycle_idx": df["cycle_index"].astype(int),
                "timestamp": None,
                "capacity": soh_capacity,
                "soh": soh,
                "voltage_mean": df[["charge_mean_voltage", "discharge_mean_voltage"]].mean(axis=1),
                "voltage_max": np.nan,
                "voltage_min": np.nan,
                "current_mean": np.nan,
                "current_max": np.nan,
                "current_min": np.nan,
                "temp_mean": np.nan,
                "temp_max": np.nan,
                "temp_min": np.nan,
                "cc_time": np.nan,
                "cv_time": np.nan,
                "charge_throughput": df["charge_capacity_Ah"].fillna(0).cumsum(),
                "discharge_throughput": df["discharge_capacity_Ah"].fillna(0).cumsum(),
                "energy_charge": df["charge_power_Wh"].astype(float),
                "energy_discharge": df["discharge_power_Wh"].astype(float),
            }
        )
        if operation_frame is not None:
            base = base.merge(operation_frame, on="cycle_idx", how="left", suffixes=("", "_from_data"))
            for column in [
                "voltage_max",
                "voltage_min",
                "current_mean",
                "current_abs_mean",
                "current_max",
                "current_min",
                "temp_mean",
                "temp_max",
                "temp_min",
            ]:
                merged_col = f"{column}_from_data"
                if merged_col in base.columns:
                    base[column] = base[merged_col]
                    base = base.drop(columns=[merged_col])

        metadata = _infer_xjtu_metadata(path, cfg)
        if np.isfinite(initial_capacity) and initial_capacity > 0:
            # XJTU documentation states that the first cycle measures the
            # initial capacity.  Use it as the SOH denominator instead of the
            # generic first-N stable-window median, which is invalid for
            # partial/satellite protocols.
            metadata["nominal_capacity_hint"] = initial_capacity
            metadata["prefer_nominal_capacity_hint"] = True
        if metadata.get("temperature_bucket") is None and base["temp_mean"].notna().any():
            metadata["temperature_bucket"] = bucket_temperature(base["temp_mean"].median())

        canonical = finalize_canonical_cell_frame(base, metadata, cfg)
        cells.append(
            CanonicalCell(
                source_dataset="xjtu",
                raw_cell_id=path.stem.replace("_summary", ""),
                file_path=str(path),
                cycles=canonical,
                source_info={
                    "batch": path.parent.name,
                    "chemistry_family": XJTU_CHEMISTRY_FAMILY,
                },
            )
        )
    return cells
