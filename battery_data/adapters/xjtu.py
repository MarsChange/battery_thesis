from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.canonicalize import finalize_canonical_cell_frame
from battery_data.schema import CanonicalCell


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
        "chemistry_family": cfg.get("chemistry_family"),
        "temperature_bucket": cfg.get("temperature_bucket"),
        "charge_rate_c": charge_rate,
        "discharge_policy_family": discharge_policy,
        "full_or_partial": full_or_partial,
    }


def load_xjtu_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    root = Path(cfg["root"])
    paths = sorted(root.glob("Batch-*/*_summary.csv"))
    max_cells = cfg.get("max_cells")
    if max_cells:
        paths = paths[: int(max_cells)]

    cells: List[CanonicalCell] = []
    for path in paths:
        df = pd.read_csv(path)
        base = pd.DataFrame(
            {
                "cycle_idx": df["cycle_index"].astype(int),
                "timestamp": None,
                "capacity": df["discharge_capacity_Ah"].astype(float),
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
        canonical = finalize_canonical_cell_frame(base, _infer_xjtu_metadata(path, cfg), cfg)
        cells.append(
            CanonicalCell(
                source_dataset="xjtu",
                raw_cell_id=path.stem.replace("_summary", ""),
                file_path=str(path),
                cycles=canonical,
                source_info={"batch": path.parent.name},
            )
        )
    return cells
