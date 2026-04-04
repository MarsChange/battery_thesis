from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from retrieval.schema import WindowSample

CANONICAL_NUMERIC_COLUMNS = [
    "capacity",
    "soh",
    "voltage_mean",
    "voltage_max",
    "voltage_min",
    "current_mean",
    "current_max",
    "current_min",
    "temp_mean",
    "temp_max",
    "temp_min",
    "cc_time",
    "cv_time",
    "charge_throughput",
    "discharge_throughput",
    "energy_charge",
    "energy_discharge",
]

CANONICAL_META_COLUMNS = [
    "chemistry_family",
    "temperature_bucket",
    "charge_rate_bucket",
    "discharge_policy_family",
    "full_or_partial",
    "nominal_capacity_bucket",
    "voltage_window_bucket",
    "missing_mask",
]

CANONICAL_COLUMNS = [
    "cell_uid",
    "cycle_idx",
    "timestamp",
    *CANONICAL_NUMERIC_COLUMNS,
    *CANONICAL_META_COLUMNS,
]

DEFAULT_TOKEN_FEATURES = [
    "soh",
    "capacity",
    "voltage_mean",
    "voltage_max",
    "voltage_min",
    "current_mean",
    "temp_mean",
    "cc_time",
    "cv_time",
    "charge_throughput",
    "discharge_throughput",
    "energy_charge",
    "energy_discharge",
]


@dataclass
class CanonicalCell:
    source_dataset: str
    raw_cell_id: str
    file_path: str
    cycles: pd.DataFrame
    source_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatteryMemorySample:
    cell_uid: str
    split: str
    window_start: int
    window_end: int
    target_start: int
    target_end: int
    window_tokens: np.ndarray
    state: Dict[str, Any]
    delta_soh: np.ndarray
    metadata: Dict[str, Any]
    domain_label: str

    def to_window_sample(self) -> WindowSample:
        return WindowSample(
            series_id=self.cell_uid,
            window_start=self.window_start,
            window_end=self.window_end,
            target_start=self.target_start,
            target_end=self.target_end,
            window_values=np.asarray(self.window_tokens, dtype=np.float32),
            future_values=np.asarray(self.delta_soh, dtype=np.float32),
            metadata={
                "s_i": self.state,
                "m_i": self.metadata,
                "d_i": self.domain_label,
            },
        )
