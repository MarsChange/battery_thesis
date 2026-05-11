"""battery_data.curve_features

提取跨数据集可用的放电 Q-V 窗口特征。

当前主线不再使用内阻、弛豫或外部时序基础模型特征。这里保留
`extract_q_indexed_feature_map` 的历史函数名和 `[6, W]` 输出形状以兼容
已有 case bank / retriever / model 接口，但语义已经改为 voltage-window
discharge features：

0. `V_window`: 选定放电电压窗口网格，单位 V。
1. `Qd(V)`: 放电段累计容量在电压窗口上的插值，单位 Ah 或原始容量单位。
2. `dQdV(V)`: Qd(V) 对电压的导数绝对值，表示 Q-V 区段峰形。
3. `Id_abs(V)`: 放电电流绝对值在电压窗口上的插值，单位 A。
4. `Temp(V)`: 温度在电压窗口上的插值，单位摄氏度；缺失时 mask=0。
5. `Power_abs(V)`: `|V * I|` 在电压窗口上的插值，单位近似 W。

Dataset-specific voltage windows:
- MIT/HUST LFP: discharge 2.8 V to 3.6 V. MIT first cycle is calibration and
  should be skipped by the caller.
- TJU/XJTU NCM and TJU NCA: discharge 3.6 V to 4.1 V.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


QV_CHANNEL_NAMES = ["V_window", "Qd(V)", "dQdV(V)", "Id_abs(V)", "Temp(V)", "Power_abs(V)"]


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


def infer_discharge_voltage_window(
    source_dataset: str | None = None,
    chemistry_family: str | None = None,
    voltage_min: float | None = None,
    voltage_max: float | None = None,
) -> tuple[float | None, float | None]:
    """Return the dataset/chemistry-specific discharge voltage window.

    中文含义：为 Q-V 窗口特征选择可比的放电电压范围。
    - MIT/HUST 的 LFP 使用 2.8 V 到 3.6 V；
    - TJU/XJTU 的 NCM 和 TJU-NCA 使用 3.6 V 到 4.1 V；
    - 显式传入 `voltage_min/max` 时优先使用显式配置。
    """

    if voltage_min is not None and voltage_max is not None:
        return float(voltage_min), float(voltage_max)
    dataset = str(source_dataset or "").lower()
    chemistry = str(chemistry_family or "").upper()
    if chemistry == "LFP" or dataset in {"mit", "hust"}:
        return 2.8, 3.6
    if chemistry in {"NCM", "NMC", "NCA"} or dataset in {"tju", "xjtu"}:
        return 3.6, 4.1
    return voltage_min, voltage_max


def _infer_time_hours(df: pd.DataFrame) -> np.ndarray:
    time_col = _find_column(df, ("time", "time (s)", "time/s", "relative_time_min"))
    if time_col is None:
        return np.arange(len(df), dtype=np.float64) / 3600.0
    values = pd.to_numeric(df[time_col], errors="coerce").astype("float64").to_numpy()
    finite = np.isfinite(values)
    if finite.any() and not finite.all():
        values = values.copy()
        values[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), values[finite])
    if "min" in time_col.lower():
        return values / 60.0
    if "time/s" in time_col.lower() or "(s)" in time_col.lower():
        return values / 3600.0
    return values / 3600.0 if np.nanmax(values) > 200 else values


def _empty_qv_window_result(q_grid_size: int, voltage_min: float = 0.0, voltage_max: float = 1.0) -> Dict[str, object]:
    voltage_grid = np.linspace(float(voltage_min), float(voltage_max), int(q_grid_size), dtype=np.float32)
    qv_map = np.zeros((6, int(q_grid_size)), dtype=np.float32)
    qv_map[0] = voltage_grid
    curve_stats = {
        "qv_dqdv_peak_value": 0.0,
        "qv_dqdv_peak_voltage": 0.0,
        "qv_dqdv_area": 0.0,
        "qv_capacity_span": 0.0,
        "qv_voltage_min": float(voltage_grid[0]) if voltage_grid.size else 0.0,
        "qv_voltage_max": float(voltage_grid[-1]) if voltage_grid.size else 0.0,
        "qv_window_available": 0.0,
        "qv_power_energy_proxy": 0.0,
    }
    return {
        "qv_map": qv_map,
        "qv_mask": np.zeros(6, dtype=np.float32),
        "q_grid": voltage_grid,
        "curve_stats": curve_stats,
    }


def extract_q_indexed_feature_map(
    cycle_raw_df: pd.DataFrame,
    q_grid_size: int = 100,
    current_eps: float = 1e-4,
    rolling_median_window: int = 5,
    r_clip_quantile: float = 0.99,
    source_dataset: str | None = None,
    chemistry_family: str | None = None,
    voltage_min: float | None = None,
    voltage_max: float | None = None,
    skip_feature_extraction: bool = False,
) -> Dict[str, object]:
    """Extract discharge-window Q-V/dQ-dV features.

    中文含义：在指定放电电压窗口内提取 Qd(V)、dQ/dV(V)、电流、温度和
    功率 proxy。输出仍命名为 `qv_map`，但不再表示旧的 Vc/Vd/DeltaV/R
    六通道图。
    """

    df = cycle_raw_df.copy()
    voltage_col = _find_column(df, ("voltage", "voltage_v", "ecell/v"))
    current_col = _find_column(df, ("current", "current_a", "<i>/ma", "current (ma)"))
    capacity_col = _find_column(
        df,
        ("capacity", "capacity_ah", "capacity (mah)", "charge_capacity", "discharge_capacity", "q charge/ma.h", "q discharge/ma.h"),
    )

    selected_min, selected_max = infer_discharge_voltage_window(
        source_dataset=source_dataset,
        chemistry_family=chemistry_family,
        voltage_min=voltage_min,
        voltage_max=voltage_max,
    )
    if selected_min is None or selected_max is None:
        selected_min, selected_max = 0.0, 1.0
    if skip_feature_extraction or df.empty:
        return _empty_qv_window_result(q_grid_size, selected_min, selected_max)

    voltage = _rolling_median(_safe_numeric(df, voltage_col), rolling_median_window)
    current = _rolling_median(_safe_numeric(df, current_col), rolling_median_window)
    temperature = _rolling_median(_safe_numeric(df, _find_column(df, ("temperature", "temperature_c", "temp", "temperature (c)"))), rolling_median_window)

    if capacity_col is not None:
        capacity = _rolling_median(_safe_numeric(df, capacity_col), rolling_median_window)
    else:
        # Fall back to cumulative |I| when explicit capacity is missing.
        dt = np.ones(len(df), dtype=np.float64)
        capacity = pd.Series(np.cumsum(np.abs(current.fillna(0.0).to_numpy()) * dt), index=df.index, dtype="float64")

    _, discharge_mask = _segment_masks(df, current, current_eps=current_eps)
    window_mask = discharge_mask & voltage.ge(float(selected_min)) & voltage.le(float(selected_max))
    if window_mask.sum() < 4:
        return _empty_qv_window_result(q_grid_size, selected_min, selected_max)

    v = voltage[window_mask].to_numpy(dtype=np.float64)
    i = current[window_mask].to_numpy(dtype=np.float64)
    q = capacity[window_mask].to_numpy(dtype=np.float64)
    temp = temperature[window_mask].to_numpy(dtype=np.float64)
    t_hours = _infer_time_hours(df)[window_mask.to_numpy()]
    valid = np.isfinite(v) & np.isfinite(i) & np.isfinite(q)
    if valid.sum() < 4:
        return _empty_qv_window_result(q_grid_size, selected_min, selected_max)
    v = v[valid]
    i = i[valid]
    q = q[valid]
    temp = temp[valid] if temp.shape[0] == valid.shape[0] else np.full(valid.sum(), np.nan)
    t_hours = t_hours[valid] if t_hours.shape[0] == valid.shape[0] else np.arange(valid.sum(), dtype=np.float64) / 3600.0

    q_rel = np.abs(q - q[0])
    order = np.argsort(v)
    v = v[order]
    q_rel = q_rel[order]
    i = i[order]
    temp = temp[order]
    t_hours = t_hours[order]
    unique_v, unique_idx = np.unique(v, return_index=True)
    if unique_v.size < 4 or unique_v[-1] - unique_v[0] < 1e-3:
        return _empty_qv_window_result(q_grid_size, selected_min, selected_max)

    q_unique = q_rel[unique_idx]
    i_unique = np.abs(i[unique_idx])
    temp_unique = temp[unique_idx]
    power_unique = np.abs(unique_v * i_unique)
    voltage_grid = np.linspace(float(selected_min), float(selected_max), int(q_grid_size), dtype=np.float32)
    q_interp = np.interp(voltage_grid, unique_v, q_unique, left=np.nan, right=np.nan).astype(np.float32)
    i_interp = np.interp(voltage_grid, unique_v, i_unique, left=np.nan, right=np.nan).astype(np.float32)
    temp_interp = np.interp(voltage_grid, unique_v, temp_unique, left=np.nan, right=np.nan).astype(np.float32)
    power_interp = np.interp(voltage_grid, unique_v, power_unique, left=np.nan, right=np.nan).astype(np.float32)
    finite_core = np.isfinite(q_interp)
    if finite_core.sum() < 4:
        return _empty_qv_window_result(q_grid_size, selected_min, selected_max)
    q_filled = pd.Series(q_interp).interpolate(limit_direction="both").to_numpy(dtype=np.float32)
    dqdv = np.abs(np.gradient(q_filled, voltage_grid)).astype(np.float32)
    finite_dqdv = dqdv[np.isfinite(dqdv)]
    if finite_dqdv.size:
        clip_value = float(np.quantile(finite_dqdv, float(r_clip_quantile)))
        if clip_value > 0:
            dqdv = np.clip(dqdv, 0.0, clip_value)
    peak_idx = int(np.nanargmax(dqdv)) if np.isfinite(dqdv).any() else 0
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:  # NumPy < 2.0 compatibility.
        trapezoid = np.trapz
    dqdv_area = float(trapezoid(np.nan_to_num(dqdv, nan=0.0), voltage_grid))
    dt = np.clip(np.diff(t_hours, prepend=t_hours[0]), 0.0, None)
    power_energy_proxy = float(np.nansum(np.abs(v * i) * dt))

    qv_map = np.zeros((6, len(voltage_grid)), dtype=np.float32)
    qv_map[0] = voltage_grid
    qv_map[1] = _nan_to_zero(q_interp)
    qv_map[2] = _nan_to_zero(dqdv)
    qv_map[3] = _nan_to_zero(i_interp)
    qv_map[4] = _nan_to_zero(temp_interp)
    qv_map[5] = _nan_to_zero(power_interp)
    qv_mask = np.asarray(
        [
            1.0,
            float(np.isfinite(q_interp).any()),
            float(np.isfinite(dqdv).any()),
            float(np.isfinite(i_interp).any()),
            float(np.isfinite(temp_interp).any()),
            float(np.isfinite(power_interp).any()),
        ],
        dtype=np.float32,
    )

    curve_stats = dict(_empty_qv_window_result(q_grid_size, selected_min, selected_max)["curve_stats"])
    curve_stats.update(
        {
            "qv_dqdv_peak_value": float(dqdv[peak_idx]) if np.isfinite(dqdv[peak_idx]) else 0.0,
            "qv_dqdv_peak_voltage": float(voltage_grid[peak_idx]),
            "qv_dqdv_area": dqdv_area,
            "qv_capacity_span": float(np.nanmax(q_rel) - np.nanmin(q_rel)) if np.isfinite(q_rel).any() else 0.0,
            "qv_voltage_min": float(selected_min),
            "qv_voltage_max": float(selected_max),
            "qv_window_available": 1.0,
            "qv_power_energy_proxy": power_energy_proxy,
        }
    )

    return {
        "qv_map": qv_map.astype(np.float32),
        "qv_mask": qv_mask.astype(np.float32),
        "q_grid": voltage_grid.astype(np.float32),
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

    voltage_grid = q_grid
    axes[0].plot(voltage_grid, qv_map[1], label="Qd(V)", linewidth=1.8)
    axes[0].set_xlabel("Voltage (V)")
    axes[0].set_ylabel("Cumulative discharge capacity")
    axes[0].set_title("Discharge Q-V window")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(voltage_grid, qv_map[2], label="dQ/dV(V)", color="tab:red", linewidth=1.8)
    axes[1].set_xlabel("Voltage (V)")
    axes[1].set_ylabel("dQ/dV")
    axes[1].set_title("Q-V peak shape")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    axes[2].plot(voltage_grid, qv_map[3], label="|Id|(V)", color="tab:orange", linewidth=1.6)
    if qv_map.shape[0] > 4 and qv_mask.shape[0] > 4 and qv_mask[4] >= 0.5:
        axes[2].plot(voltage_grid, qv_map[4], label="Temp(V)", color="tab:green", linewidth=1.4)
    axes[2].set_xlabel("Voltage (V)")
    axes[2].set_ylabel("Current / temperature")
    axes[2].set_title("Operation traces over voltage window")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend()

    axes[3].plot(voltage_grid, qv_map[5], label="|V*I|(V)", color="tab:purple", linewidth=1.8)
    axes[3].set_xlabel("Voltage (V)")
    axes[3].set_ylabel("Power proxy")
    axes[3].set_title("Power proxy over voltage window")
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

    figure.suptitle(title or "Discharge Q-V window feature map", fontsize=14)
    figure.tight_layout(rect=[0.02, 0.04, 0.98, 0.96])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
