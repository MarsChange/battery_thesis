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


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float32")
    return pd.to_numeric(df[column], errors="coerce").astype("float32")


def _rolling_delta(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.diff(periods=periods).astype("float32")


def _rolling_mean_slope(series: pd.Series, window: int) -> pd.Series:
    return series.diff().rolling(window=max(window - 1, 1), min_periods=1).mean().astype("float32")


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=max(window, 2), min_periods=2).std().astype("float32")


def _rolling_spectral_entropy(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(max(window // 2, 4), window)

    def _entropy(chunk: np.ndarray) -> float:
        arr = np.asarray(chunk, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size < 4:
            return np.nan
        arr = arr - arr.mean()
        power = np.abs(np.fft.rfft(arr)) ** 2
        if power.size <= 1:
            return np.nan
        power = power[1:]
        total = float(power.sum())
        if total < 1e-12:
            return np.nan
        probs = power / total
        denom = math.log(len(probs))
        if denom <= 0:
            return np.nan
        return float(-(probs * np.log(probs + 1e-12)).sum() / denom)

    return series.rolling(window=window, min_periods=min_periods).apply(_entropy, raw=True).astype("float32")


def _rolling_low_freq_ratio(series: pd.Series, window: int) -> pd.Series:
    min_periods = min(max(window // 2, 4), window)

    def _ratio(chunk: np.ndarray) -> float:
        arr = np.asarray(chunk, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size < 4:
            return np.nan
        arr = arr - arr.mean()
        power = np.abs(np.fft.rfft(arr)) ** 2
        if power.size <= 2:
            return np.nan
        power = power[1:]
        total = float(power.sum())
        if total < 1e-12:
            return np.nan
        low_bins = max(1, len(power) // 3)
        return float(power[:low_bins].sum() / total)

    return series.rolling(window=window, min_periods=min_periods).apply(_ratio, raw=True).astype("float32")


def augment_cycle_feature_frame(
    cell_cycles: pd.DataFrame,
    rolling_window: int = 5,
    spectral_window: int = 16,
    spectral_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    df = cell_cycles.copy()
    spectral_columns = list(spectral_columns or ["voltage_mean", "temp_mean", "current_mean"])

    df["soh_pct"] = _numeric_series(df, "soh") * 100.0
    df["capacity_ratio"] = _numeric_series(df, "soh")
    df["voltage_range"] = _numeric_series(df, "voltage_max") - _numeric_series(df, "voltage_min")
    df["temp_range"] = _numeric_series(df, "temp_max") - _numeric_series(df, "temp_min")
    df["current_abs_mean"] = _numeric_series(df, "current_mean").abs()

    for column in ["soh", "voltage_mean", "temp_mean", "current_mean"]:
        series = _numeric_series(df, column)
        df[f"{column}_diff_1"] = _rolling_delta(series, periods=1)
        df[f"{column}_slope_{rolling_window}"] = _rolling_mean_slope(series, window=rolling_window)
        df[f"{column}_std_{rolling_window}"] = _rolling_std(series, window=rolling_window)

    for column in ["charge_throughput", "discharge_throughput", "energy_charge", "energy_discharge"]:
        series = _numeric_series(df, column)
        df[f"{column}_delta_1"] = _rolling_delta(series, periods=1)

    for column in spectral_columns:
        series = _numeric_series(df, column)
        df[f"{column}_fft_entropy_{spectral_window}"] = _rolling_spectral_entropy(series, window=spectral_window)
        df[f"{column}_fft_low_ratio_{spectral_window}"] = _rolling_low_freq_ratio(series, window=spectral_window)

    return df
