from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.canonicalize import finalize_canonical_cell_frame
from battery_data.schema import CanonicalCell

MIT_CHEMISTRY_FAMILY = "LFP"


def _infer_mit_metadata(summary_path: Path, meta_path: Path | None, cfg: Dict[str, object]) -> Dict[str, object]:
    protocol = None
    if meta_path and meta_path.exists():
        meta_df = pd.read_csv(meta_path)
        if "protocol" in meta_df.columns and not meta_df.empty:
            protocol = str(meta_df.loc[0, "protocol"])

    charge_rate = None
    if protocol:
        rates = [float(match) for match in re.findall(r"(\d+(?:\.\d+)?)C", protocol)]
        if rates:
            charge_rate = max(rates)

    return {
        "chemistry_family": cfg.get("chemistry_family") or MIT_CHEMISTRY_FAMILY,
        "temperature_bucket": cfg.get("temperature_bucket"),
        "charge_rate_c": charge_rate,
        "discharge_policy_family": "fastcharge",
        "full_or_partial": "full",
    }


def load_mit_cells(cfg: Dict[str, object]) -> List[CanonicalCell]:
    root = Path(cfg["root"])
    paths = sorted(root.glob("*_structure_summary.csv"))
    max_cells = cfg.get("max_cells")
    if max_cells:
        paths = paths[: int(max_cells)]

    cells: List[CanonicalCell] = []
    for summary_path in paths:
        summary = pd.read_csv(summary_path)
        discharge_v = summary["discharge_energy"] / summary["discharge_capacity"].replace(0, np.nan)
        charge_v = summary["charge_energy"] / summary["charge_capacity"].replace(0, np.nan)
        voltage_mean = pd.concat([discharge_v, charge_v], axis=1).mean(axis=1)
        base = pd.DataFrame(
            {
                "cycle_idx": summary["cycle_index"].astype(int),
                "timestamp": summary.get("date_time_iso"),
                "capacity": summary["discharge_capacity"].astype(float),
                "voltage_mean": voltage_mean.astype(float),
                "voltage_max": np.nan,
                "voltage_min": np.nan,
                "current_mean": np.nan,
                "current_max": np.nan,
                "current_min": np.nan,
                "temp_mean": summary["temperature_average"].astype(float),
                "temp_max": summary["temperature_maximum"].astype(float),
                "temp_min": summary["temperature_minimum"].astype(float),
                "cc_time": summary["charge_duration"].astype(float),
                "cv_time": np.nan,
                "charge_throughput": summary["charge_throughput"].astype(float),
                "discharge_throughput": summary["discharge_capacity"].fillna(0).cumsum(),
                "energy_charge": summary["charge_energy"].astype(float),
                "energy_discharge": summary["discharge_energy"].astype(float),
            }
        )
        meta_path = summary_path.with_name(summary_path.name.replace("_summary.csv", "_meta.csv"))
        canonical = finalize_canonical_cell_frame(base, _infer_mit_metadata(summary_path, meta_path, cfg), cfg)
        cells.append(
            CanonicalCell(
                source_dataset="mit",
                raw_cell_id=summary_path.stem.replace("_structure_summary", ""),
                file_path=str(summary_path),
                cycles=canonical,
                source_info={
                    "meta_path": str(meta_path) if meta_path.exists() else None,
                    "chemistry_family": MIT_CHEMISTRY_FAMILY,
                },
            )
        )
    return cells
