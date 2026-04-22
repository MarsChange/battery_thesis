from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QV_CHANNEL_NAMES = ["Vc", "Vd", "Ic", "Id", "DeltaV", "R"]


def _find_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return str(lowered[candidate.lower()])
    return None


def _rolling_median(series: pd.Series, window: int) -> pd.Series:
    window = max(int(window), 1)
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).median()


def _safe_numeric(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _segment_masks(df: pd.DataFrame, current: pd.Series, current_eps: float) -> Tuple[pd.Series, pd.Series]:
    step_col = _find_column(df, ("step", "status", "step_type", "description"))
    if step_col is not None:
        step_text = df[step_col].fillna("").astype(str).str.lower()
        charge_mask = step_text.str.contains("charge") | step_text.str.contains("cv") | step_text.str.contains("cc")
        discharge_mask = step_text.str.contains("discharge")
        # Prefer explicit status, but fall back to current sign where status is ambiguous.
        charge_mask = charge_mask | (current > current_eps)
        discharge_mask = discharge_mask | (current < -current_eps)
    else:
        charge_mask = current > current_eps
        discharge_mask = current < -current_eps
    return charge_mask.fillna(False), discharge_mask.fillna(False)


def _normalized_capacity(capacity: pd.Series) -> pd.Series:
    valid = capacity.replace([np.inf, -np.inf], np.nan)
    if valid.notna().sum() < 2:
        return pd.Series(np.nan, index=capacity.index, dtype="float64")
    cap_min = float(valid.min())
    cap_max = float(valid.max())
    if not np.isfinite(cap_min) or not np.isfinite(cap_max) or abs(cap_max - cap_min) < 1e-8:
        return pd.Series(np.nan, index=capacity.index, dtype="float64")
    return (capacity - cap_min) / (cap_max - cap_min)


def _interpolate_to_q_grid(
    q: pd.Series,
    value: pd.Series,
    q_grid: np.ndarray,
) -> np.ndarray | None:
    arr_q = np.asarray(q, dtype=np.float64)
    arr_v = np.asarray(value, dtype=np.float64)
    valid = np.isfinite(arr_q) & np.isfinite(arr_v)
    if valid.sum() < 2:
        return None
    arr_q = arr_q[valid]
    arr_v = arr_v[valid]
    order = np.argsort(arr_q)
    arr_q = arr_q[order]
    arr_v = arr_v[order]
    unique_q, unique_idx = np.unique(arr_q, return_index=True)
    arr_v = arr_v[unique_idx]
    if unique_q.size < 2:
        return None
    if unique_q[-1] - unique_q[0] < 1e-6:
        return None
    return np.interp(q_grid, unique_q, arr_v, left=arr_v[0], right=arr_v[-1]).astype(np.float32)


def _nan_to_zero(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def extract_q_indexed_feature_map(
    cycle_raw_df: pd.DataFrame,
    q_grid_size: int = 100,
    current_eps: float = 1e-4,
    rolling_median_window: int = 5,
    r_clip_quantile: float = 0.99,
) -> Dict[str, object]:
    df = cycle_raw_df.copy()
    voltage_col = _find_column(df, ("voltage", "voltage_v", "ecell/v"))
    current_col = _find_column(df, ("current", "current_a", "<i>/ma", "current (ma)"))
    capacity_col = _find_column(
        df,
        ("capacity", "capacity_ah", "capacity (mah)", "charge_capacity", "discharge_capacity", "q charge/ma.h", "q discharge/ma.h"),
    )

    voltage = _rolling_median(_safe_numeric(df, voltage_col), rolling_median_window)
    current = _rolling_median(_safe_numeric(df, current_col), rolling_median_window)

    if capacity_col is not None:
        capacity = _rolling_median(_safe_numeric(df, capacity_col), rolling_median_window)
    else:
        # Fall back to cumulative |I| when explicit capacity is missing.
        dt = np.ones(len(df), dtype=np.float64)
        capacity = pd.Series(np.cumsum(np.abs(current.fillna(0.0).to_numpy()) * dt), index=df.index, dtype="float64")

    charge_mask, discharge_mask = _segment_masks(df, current, current_eps=current_eps)
    q_grid = np.linspace(0.0, 1.0, int(q_grid_size), dtype=np.float32)

    qv_map = np.zeros((6, len(q_grid)), dtype=np.float32)
    qv_mask = np.zeros(6, dtype=np.float32)

    def _segment_values(mask: pd.Series) -> Tuple[np.ndarray | None, np.ndarray | None]:
        if not mask.any():
            return None, None
        seg_cap = _normalized_capacity(capacity[mask])
        seg_v = _interpolate_to_q_grid(seg_cap, voltage[mask], q_grid)
        seg_i = _interpolate_to_q_grid(seg_cap, current[mask], q_grid)
        return seg_v, seg_i

    v_charge, i_charge = _segment_values(charge_mask)
    v_discharge, i_discharge = _segment_values(discharge_mask)

    if v_charge is not None:
        qv_map[0] = _nan_to_zero(v_charge)
        qv_mask[0] = 1.0
    if v_discharge is not None:
        qv_map[1] = _nan_to_zero(v_discharge)
        qv_mask[1] = 1.0
    if i_charge is not None:
        qv_map[2] = _nan_to_zero(i_charge)
        qv_mask[2] = 1.0
    if i_discharge is not None:
        qv_map[3] = _nan_to_zero(i_discharge)
        qv_mask[3] = 1.0

    if v_charge is not None and v_discharge is not None:
        delta_v = np.asarray(v_charge - v_discharge, dtype=np.float32)
        qv_map[4] = _nan_to_zero(delta_v)
        qv_mask[4] = 1.0

    if i_charge is not None and i_discharge is not None and v_charge is not None and v_discharge is not None:
        denom = np.asarray(i_charge - i_discharge, dtype=np.float32)
        denom[np.abs(denom) < current_eps] = float(current_eps)
        resistance = np.asarray((v_charge - v_discharge) / denom, dtype=np.float32)
        finite = np.isfinite(resistance)
        if finite.any():
            clip_value = float(np.quantile(np.abs(resistance[finite]), float(r_clip_quantile)))
            if clip_value > 0:
                resistance = np.clip(resistance, -clip_value, clip_value)
        qv_map[5] = _nan_to_zero(resistance)
        qv_mask[5] = 1.0

    vc_slope = np.gradient(qv_map[0], q_grid) if qv_mask[0] else np.zeros_like(q_grid)
    vd_slope = np.gradient(qv_map[1], q_grid) if qv_mask[1] else np.zeros_like(q_grid)
    delta_v = qv_map[4] if qv_mask[4] else np.zeros_like(q_grid)
    resistance = qv_map[5] if qv_mask[5] else np.zeros_like(q_grid)

    def _masked_mean(values: np.ndarray, enabled: bool) -> float:
        return float(np.mean(values)) if enabled else 0.0

    def _masked_std(values: np.ndarray, enabled: bool) -> float:
        return float(np.std(values)) if enabled else 0.0

    curve_stats = {
        "delta_v_mean": _masked_mean(delta_v, bool(qv_mask[4])),
        "delta_v_std": _masked_std(delta_v, bool(qv_mask[4])),
        "delta_v_max": float(np.max(delta_v)) if qv_mask[4] else 0.0,
        "r_mean": _masked_mean(resistance, bool(qv_mask[5])),
        "r_std": _masked_std(resistance, bool(qv_mask[5])),
        "r_q95": float(np.quantile(resistance, 0.95)) if qv_mask[5] else 0.0,
        "vc_slope_mean": _masked_mean(vc_slope, bool(qv_mask[0])),
        "vd_slope_mean": _masked_mean(vd_slope, bool(qv_mask[1])),
        "ic_mean": _masked_mean(qv_map[2], bool(qv_mask[2])),
        "id_mean": _masked_mean(qv_map[3], bool(qv_mask[3])),
        "v_charge_mean": _masked_mean(qv_map[0], bool(qv_mask[0])),
        "v_discharge_mean": _masked_mean(qv_map[1], bool(qv_mask[1])),
    }

    return {
        "qv_map": qv_map.astype(np.float32),
        "qv_mask": qv_mask.astype(np.float32),
        "q_grid": q_grid.astype(np.float32),
        "curve_stats": curve_stats,
    }


def plot_qv_feature_map(
    q_grid: np.ndarray,
    qv_map: np.ndarray,
    qv_mask: np.ndarray,
    save_path: str | Path,
    title: str | None = None,
) -> None:
    q_grid = np.asarray(q_grid, dtype=np.float32)
    qv_map = np.asarray(qv_map, dtype=np.float32)
    qv_mask = np.asarray(qv_mask, dtype=np.float32)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180)
    axes = axes.reshape(-1)

    unavailable = [name for name, enabled in zip(QV_CHANNEL_NAMES, qv_mask) if enabled < 0.5]

    axes[0].plot(q_grid, qv_map[0], label="Vc(Q)", linewidth=1.6)
    axes[0].plot(q_grid, qv_map[1], label="Vd(Q)", linewidth=1.6)
    axes[0].set_xlabel("Normalized capacity Q")
    axes[0].set_ylabel("Voltage")
    axes[0].set_title("Voltage curves")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(q_grid, qv_map[2], label="Ic(Q)", linewidth=1.6)
    axes[1].plot(q_grid, qv_map[3], label="Id(Q)", linewidth=1.6)
    axes[1].set_xlabel("Normalized capacity Q")
    axes[1].set_ylabel("Current")
    axes[1].set_title("Current curves")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(q_grid, qv_map[4], label="DeltaV(Q)", color="tab:red", linewidth=1.8)
    axes[2].set_xlabel("Normalized capacity Q")
    axes[2].set_ylabel("Voltage gap")
    axes[2].set_title("Polarization gap")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    axes[3].plot(q_grid, qv_map[5], label="R(Q)", color="tab:purple", linewidth=1.8)
    axes[3].set_xlabel("Normalized capacity Q")
    axes[3].set_ylabel("Resistance proxy")
    axes[3].set_title("Resistance proxy")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend()

    if unavailable:
        figure.text(
            0.5,
            0.01,
            "Unavailable channels: " + ", ".join(unavailable),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    figure.suptitle(title or "Q-indexed curve feature map", fontsize=14)
    figure.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
