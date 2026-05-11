from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.canonicalize import finalize_canonical_cell_frame
from battery_data.features import bucket_temperature
from battery_data.schema import CanonicalCell

CHEMISTRY_BY_FOLDER = {
    "Dataset_1_NCA_battery": "NCA",
    "Dataset_2_NCM_battery": "NCM",
    "Dataset_3_NCM_NCA_battery": "Blend",
}


def _repair_isolated_capacity_outliers(base: pd.DataFrame, cfg: Dict[str, object]) -> pd.DataFrame:
    """Interpolate isolated gross capacity dropouts in TJU cycle summaries.

    Some TJU raw cycles contain an almost-zero recorded discharge capacity while
    the adjacent cycles remain near the normal degradation trajectory. Those
    points are measurement/segmentation dropouts, not physical SOH changes. If
    left untreated, they become target SOH spikes and are copied by RAG
    references. This repair only targets gross local outliers and leaves smooth
    degradation trends unchanged.
    """

    enabled = bool(cfg.get("tju_repair_soh_outliers", cfg.get("repair_soh_outliers", True)))
    if not enabled or base.empty or "capacity" not in base.columns:
        return base

    repaired = base.copy()
    capacity = pd.to_numeric(repaired["capacity"], errors="coerce").astype(float)
    if capacity.notna().sum() < 5:
        return repaired

    window = int(cfg.get("soh_outlier_local_window", cfg.get("capacity_outlier_local_window", 7)))
    if window < 3:
        window = 3
    if window % 2 == 0:
        window += 1
    relative_floor = float(cfg.get("soh_outlier_relative_floor", cfg.get("capacity_outlier_relative_floor", 0.60)))
    min_capacity = float(cfg.get("soh_outlier_min_capacity_ah", cfg.get("capacity_outlier_min_capacity_ah", 1e-4)))

    local_median = capacity.rolling(window=window, center=True, min_periods=max(3, window // 2)).median()
    global_median = float(capacity[capacity > min_capacity].median()) if (capacity > min_capacity).any() else float("nan")
    if np.isfinite(global_median):
        local_median = local_median.fillna(global_median)

    gross_dropout = capacity.le(min_capacity)
    local_dropout = capacity.lt(local_median * relative_floor) & local_median.gt(min_capacity)
    outlier_mask = (gross_dropout | local_dropout) & capacity.notna()
    if not bool(outlier_mask.any()):
        return repaired

    repaired_capacity = capacity.mask(outlier_mask, np.nan)
    repaired_capacity = repaired_capacity.interpolate(method="linear", limit_direction="both")
    repaired["capacity"] = repaired_capacity.astype(float)

    # Keep per-cycle discharge throughput consistent before the adapter turns it
    # into cumulative throughput below.
    if "discharge_throughput" in repaired.columns:
        repaired["discharge_throughput"] = repaired_capacity.astype(float)
    if "energy_discharge" in repaired.columns:
        old_capacity = capacity.replace(0.0, np.nan)
        voltage_proxy = pd.to_numeric(repaired["energy_discharge"], errors="coerce") / old_capacity
        repaired["energy_discharge"] = repaired_capacity * voltage_proxy.interpolate(method="linear", limit_direction="both")

    return repaired


def _parse_rate_token(token: str) -> float | None:
    mapping = {
        "025": 0.25,
        "05": 0.5,
        "1": 1.0,
    }
    if token in mapping:
        return mapping[token]
    try:
        return float(token)
    except ValueError:
        return None


def _infer_tju_metadata(path: Path, cfg: Dict[str, object]) -> Dict[str, object]:
    folder_name = path.parent.name
    chemistry = cfg.get("chemistry_family") or CHEMISTRY_BY_FOLDER.get(folder_name)
    match = re.match(r"CY(?P<temp>\d+)-(?P<rate>[0-9]+)_(?P<group>\d+)-#(?P<cell>\d+)", path.stem)
    charge_rate = None
    temperature_bucket = cfg.get("temperature_bucket")
    temperature_c = None
    if match:
        charge_rate = _parse_rate_token(match.group("rate"))
        temperature_c = float(match.group("temp"))
        if temperature_bucket is None:
            temperature_bucket = bucket_temperature(temperature_c)
    return {
        "chemistry_family": chemistry,
        "temperature_bucket": temperature_bucket,
        "temperature_c": temperature_c,
        "charge_rate_c": charge_rate,
        "discharge_policy_family": "regular",
        "full_or_partial": "full",
        "voltage_min_hint": 2.65,
        "voltage_max_hint": 4.2,
    }


def load_tju_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    root = Path(cfg["root"])
    paths = sorted(root.glob("Dataset_*/*.csv"))
    exclude_folders = {str(value) for value in cfg.get("exclude_folders", [])}
    if exclude_folders:
        paths = [path for path in paths if path.parent.name not in exclude_folders]
    max_cells = cfg.get("max_cells")
    if max_cells:
        paths = paths[: int(max_cells)]

    cells: List[CanonicalCell] = []
    usecols = [
        "time/s",
        "control/V",
        "Ecell/V",
        "<I>/mA",
        "Q discharge/mA.h",
        "Q charge/mA.h",
        "cycle number",
    ]
    for path in paths:
        metadata = _infer_tju_metadata(path, cfg)
        temperature_c = metadata.get("temperature_c")
        df = pd.read_csv(path, usecols=usecols)
        grouped = df.groupby("cycle number", sort=True)
        rows = []
        for cycle_idx, grp in grouped:
            current_a = grp["<I>/mA"] / 1000.0
            voltage = grp["Ecell/V"]
            charge_capacity = grp["Q charge/mA.h"].max() / 1000.0
            discharge_capacity = grp["Q discharge/mA.h"].max() / 1000.0
            charge_rows = current_a > 0
            cc_rows = charge_rows & grp["control/V"].fillna(0).abs().lt(1e-6)
            cv_rows = charge_rows & grp["control/V"].fillna(0).gt(0)

            def _duration(mask: pd.Series) -> float | None:
                if not mask.any():
                    return None
                times = grp.loc[mask, "time/s"]
                return float(times.max() - times.min())

            charge_v = voltage[charge_rows]
            discharge_v = voltage[current_a < 0]
            rows.append(
                {
                    "cycle_idx": int(cycle_idx),
                    "timestamp": None,
                    "capacity": float(discharge_capacity),
                    "voltage_mean": float(voltage.mean()),
                    "voltage_max": float(voltage.max()),
                    "voltage_min": float(voltage.min()),
                    "current_mean": float(current_a.mean()),
                    "current_abs_mean": float(current_a.abs().mean()),
                    "current_max": float(current_a.max()),
                    "current_min": float(current_a.min()),
                    "temp_mean": temperature_c,
                    "temp_max": temperature_c,
                    "temp_min": temperature_c,
                    "cc_time": _duration(cc_rows),
                    "cv_time": _duration(cv_rows),
                    "charge_throughput": float(charge_capacity),
                    "discharge_throughput": float(discharge_capacity),
                    "energy_charge": float(charge_capacity * charge_v.mean()) if not charge_v.empty else np.nan,
                    "energy_discharge": float(discharge_capacity * discharge_v.mean()) if not discharge_v.empty else np.nan,
                }
            )
        base = pd.DataFrame(rows).sort_values("cycle_idx").reset_index(drop=True)
        base = _repair_isolated_capacity_outliers(base, cfg)
        base["charge_throughput"] = base["charge_throughput"].fillna(0).cumsum()
        base["discharge_throughput"] = base["discharge_throughput"].fillna(0).cumsum()
        metadata.pop("temperature_c", None)
        canonical = finalize_canonical_cell_frame(base, metadata, cfg)
        cells.append(
            CanonicalCell(
                source_dataset="tju",
                raw_cell_id=f"{path.parent.name}/{path.stem}",
                file_path=str(path),
                cycles=canonical,
                source_info={"folder": path.parent.name, "condition_group": path.stem.split("-#")[0]},
            )
        )
    return cells
