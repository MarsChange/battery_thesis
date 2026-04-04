from __future__ import annotations

import json
import math
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import CANONICAL_META_COLUMNS, CANONICAL_NUMERIC_COLUMNS


def stable_nominal_capacity(
    capacity: Sequence[float],
    skip_initial_cycles: int = 0,
    stable_window: int = 5,
) -> float:
    series = pd.Series(capacity, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return float("nan")
    start = min(skip_initial_cycles, len(series) - 1)
    window = series.iloc[start : start + max(stable_window, 1)]
    if window.empty:
        window = series.iloc[: max(stable_window, 1)]
    return float(window.median())


def safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def bucket_temperature(value: object) -> str | None:
    temp = safe_float(value)
    if temp is None:
        return None
    if temp < 15:
        return "cold"
    if temp < 30:
        return "room"
    if temp < 40:
        return "warm"
    return "hot"


def bucket_charge_rate(value: object) -> str | None:
    rate = safe_float(value)
    if rate is None:
        return None
    if rate < 0.4:
        return "low"
    if rate < 1.5:
        return "medium"
    if rate < 3.5:
        return "high"
    return "very_high"


def bucket_nominal_capacity(value: object) -> str | None:
    cap = safe_float(value)
    if cap is None:
        return None
    if cap < 1.0:
        return "sub_1ah"
    if cap < 2.0:
        return "1_to_2ah"
    if cap < 3.0:
        return "2_to_3ah"
    return "3ah_plus"


def bucket_voltage_window(v_min: object, v_max: object) -> str | None:
    low = safe_float(v_min)
    high = safe_float(v_max)
    if low is None or high is None:
        return None
    return f"{low:.1f}_{high:.1f}V"


def build_missing_mask(
    row: pd.Series,
    fields: Optional[Iterable[str]] = None,
) -> List[str]:
    if fields is None:
        fields = [*CANONICAL_NUMERIC_COLUMNS, *CANONICAL_META_COLUMNS[:-1]]
    missing = []
    for field in fields:
        value = row.get(field)
        if isinstance(value, str):
            if value == "":
                missing.append(field)
            continue
        if value is None or (isinstance(value, float) and np.isnan(value)):
            missing.append(field)
    return sorted(set(missing))


def serialize_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True)


def parse_json_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value:
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except json.JSONDecodeError:
            pass
    return []


def recent_delta_mean(values: Sequence[float], last_k: int) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 2:
        return 0.0
    tail = arr[max(0, arr.size - last_k - 1) :]
    if tail.size < 2:
        return 0.0
    return float(np.diff(tail).mean())
