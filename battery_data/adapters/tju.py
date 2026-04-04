from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.canonicalize import finalize_canonical_cell_frame
from battery_data.schema import CanonicalCell

CHEMISTRY_BY_FOLDER = {
    "Dataset_1_NCA_battery": "NCA",
    "Dataset_2_NCM_battery": "NCM",
    "Dataset_3_NCM_NCA_battery": "Blend",
}


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
    if match:
        charge_rate = _parse_rate_token(match.group("rate"))
        if temperature_bucket is None:
            temperature = float(match.group("temp"))
            if temperature < 30:
                temperature_bucket = "room"
            elif temperature < 40:
                temperature_bucket = "warm"
            else:
                temperature_bucket = "hot"
    return {
        "chemistry_family": chemistry,
        "temperature_bucket": temperature_bucket,
        "charge_rate_c": charge_rate,
        "discharge_policy_family": "regular",
        "full_or_partial": "full",
        "voltage_min_hint": 2.65,
        "voltage_max_hint": 4.2,
    }


def load_tju_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    root = Path(cfg["root"])
    paths = sorted(root.glob("Dataset_*/*.csv"))
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
                    "current_max": float(current_a.max()),
                    "current_min": float(current_a.min()),
                    "temp_mean": np.nan,
                    "temp_max": np.nan,
                    "temp_min": np.nan,
                    "cc_time": _duration(cc_rows),
                    "cv_time": _duration(cv_rows),
                    "charge_throughput": float(charge_capacity),
                    "discharge_throughput": float(discharge_capacity),
                    "energy_charge": float(charge_capacity * charge_v.mean()) if not charge_v.empty else np.nan,
                    "energy_discharge": float(discharge_capacity * discharge_v.mean()) if not discharge_v.empty else np.nan,
                }
            )
        base = pd.DataFrame(rows).sort_values("cycle_idx").reset_index(drop=True)
        base["charge_throughput"] = base["charge_throughput"].fillna(0).cumsum()
        base["discharge_throughput"] = base["discharge_throughput"].fillna(0).cumsum()
        canonical = finalize_canonical_cell_frame(base, _infer_tju_metadata(path, cfg), cfg)
        cells.append(
            CanonicalCell(
                source_dataset="tju",
                raw_cell_id=path.stem,
                file_path=str(path),
                cycles=canonical,
                source_info={"folder": path.parent.name},
            )
        )
    return cells
