from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.canonicalize import finalize_canonical_cell_frame
from battery_data.schema import CanonicalCell

HUST_CHEMISTRY_FAMILY = "LFP"
HUST_TEMPERATURE_C = 30.0
HUST_NOMINAL_CAPACITY_AH = 1.1
HUST_NOMINAL_VOLTAGE_V = 3.3


def _infer_hust_metadata(df: pd.DataFrame, cfg: Dict[str, object]) -> Dict[str, object]:
    # HUST contains LFP/graphite cells cycled under multi-stage discharge
    # protocols at a constant 30 C environment.
    return {
        "chemistry_family": cfg.get("chemistry_family") or HUST_CHEMISTRY_FAMILY,
        "temperature_bucket": cfg.get("temperature_bucket") or "room",
        "discharge_policy_family": "multistage",
        "full_or_partial": "full",
        "nominal_capacity_hint": cfg.get("nominal_capacity_hint", HUST_NOMINAL_CAPACITY_AH),
        "nominal_voltage_hint": cfg.get("nominal_voltage_hint", HUST_NOMINAL_VOLTAGE_V),
        "voltage_min_hint": 2.0,
        "voltage_max_hint": 4.5,
    }


def load_hust_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    root = Path(cfg["root"])
    paths = sorted(root.glob("*.csv"))
    max_cells = cfg.get("max_cells")
    if max_cells:
        paths = paths[: int(max_cells)]

    cells: List[CanonicalCell] = []
    usecols = [
        "battery_id",
        "cycle_index",
        "Status",
        "Current (mA)",
        "Voltage (V)",
        "Capacity (mAh)",
        "Time (s)",
        "dq",
    ]
    for path in paths:
        df = pd.read_csv(path, usecols=usecols)
        grouped = df.groupby("cycle_index", sort=True)
        rows = []
        for cycle_idx, grp in grouped:
            current_a = grp["Current (mA)"] / 1000.0
            voltage = grp["Voltage (V)"]
            status = grp["Status"].fillna("")
            charge_mask = status.str.contains("charge", case=False, regex=False) & current_a.gt(0)
            cv_mask = status.str.contains("constant voltage", case=False, regex=False)
            cc_mask = charge_mask & ~cv_mask
            dq = grp["dq"].dropna()
            discharge_capacity = float(dq.iloc[0] / 1000.0) if not dq.empty else np.nan
            charge_capacity = float(grp.loc[charge_mask, "Capacity (mAh)"].max() / 1000.0) if charge_mask.any() else np.nan
            charge_voltage = voltage[charge_mask]
            discharge_voltage = voltage[current_a.lt(0)]

            def _duration(mask: pd.Series) -> float | None:
                if not mask.any():
                    return None
                times = grp.loc[mask, "Time (s)"]
                return float(times.max() - times.min())

            rows.append(
                {
                    "cycle_idx": int(cycle_idx),
                    "timestamp": None,
                    "capacity": discharge_capacity,
                    "voltage_mean": float(voltage.mean()),
                    "voltage_max": float(voltage.max()),
                    "voltage_min": float(voltage.min()),
                    "current_mean": float(current_a.mean()),
                    "current_max": float(current_a.max()),
                    "current_min": float(current_a.min()),
                    "temp_mean": HUST_TEMPERATURE_C,
                    "temp_max": HUST_TEMPERATURE_C,
                    "temp_min": HUST_TEMPERATURE_C,
                    "cc_time": _duration(cc_mask),
                    "cv_time": _duration(cv_mask),
                    "charge_throughput": charge_capacity,
                    "discharge_throughput": discharge_capacity,
                    "energy_charge": float(charge_capacity * charge_voltage.mean()) if not charge_voltage.empty and not np.isnan(charge_capacity) else np.nan,
                    "energy_discharge": float(discharge_capacity * discharge_voltage.mean()) if not discharge_voltage.empty and not np.isnan(discharge_capacity) else np.nan,
                }
            )
        base = pd.DataFrame(rows).sort_values("cycle_idx").reset_index(drop=True)
        base["charge_throughput"] = base["charge_throughput"].fillna(0).cumsum()
        base["discharge_throughput"] = base["discharge_throughput"].fillna(0).cumsum()
        canonical = finalize_canonical_cell_frame(base, _infer_hust_metadata(df, cfg), cfg)
        cells.append(
            CanonicalCell(
                source_dataset="hust",
                raw_cell_id=path.stem,
                file_path=str(path),
                cycles=canonical,
                source_info={
                    "battery_id": df["battery_id"].iloc[0] if "battery_id" in df.columns and not df.empty else path.stem,
                    "chemistry_family": HUST_CHEMISTRY_FAMILY,
                    "nominal_capacity_ah": HUST_NOMINAL_CAPACITY_AH,
                    "nominal_voltage_v": HUST_NOMINAL_VOLTAGE_V,
                    "temperature_c": HUST_TEMPERATURE_C,
                },
            )
        )
    return cells
