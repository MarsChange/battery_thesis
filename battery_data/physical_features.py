"""Physical proxy features for numerical battery SOH forecasting.

The current public datasets in this project do not expose a reliable
charge-end rest segment. The main physical feature vector combines Q-V polarization proxies from
`DeltaV(Q)` and `R(Q)` with a small set of partial-charging summaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PHYSICS_FEATURE_NAMES = [
    "delta_v_mean",
    "delta_v_std",
    "delta_v_q95",
    "r_mean",
    "r_std",
    "r_q95",
    "vc_curve_slope_mean",
    "vd_curve_slope_mean",
    "q_total",
    "q_mean",
    "q_std",
    "dq_dv_peak_value",
]


def _find_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return str(lowered[candidate.lower()])
    return None


def _safe_numeric(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _infer_time_minutes(df: pd.DataFrame) -> np.ndarray:
    time_col = _find_column(
        df,
        ("time", "time (s)", "time/s", "relative_time_min", "timestamp", "system_time"),
    )
    if time_col is None:
        return np.arange(len(df), dtype=np.float64)
    series = df[time_col]
    if np.issubdtype(series.dtype, np.number):
        values = pd.to_numeric(series, errors="coerce").astype("float64").to_numpy()
        finite = np.isfinite(values)
        if finite.any():
            values = values.copy()
            values[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), values[finite])
            if "min" in time_col.lower():
                return values
            if "time/s" in time_col.lower() or "(s)" in time_col.lower():
                return values / 60.0
            return values / 60.0 if np.nanmax(values) > 200 else values
    try:
        dt = pd.to_datetime(series, errors="coerce")
        if dt.notna().sum() >= 2:
            return (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
    except Exception:
        pass
    return np.arange(len(df), dtype=np.float64)


def _charge_mask(df: pd.DataFrame) -> pd.Series:
    current = _safe_numeric(df, _find_column(df, ("current", "current_a", "<i>/ma", "current (ma)")))
    step_col = _find_column(df, ("step", "status", "step_type", "description"))
    if step_col is not None:
        step_text = df[step_col].fillna("").astype(str).str.lower()
        mask = step_text.str.contains("charge") | step_text.str.contains("cv") | step_text.str.contains("cc")
        mask = mask | current.gt(1e-4)
    else:
        mask = current.gt(1e-4)
    return mask.fillna(False)


def extract_partial_charge_curve(
    cycle_raw_df: pd.DataFrame,
    voltage_grid_size: int = 50,
    voltage_grid_mode: str = "chemistry_window",
    voltage_min: float | None = None,
    voltage_max: float | None = None,
) -> Dict[str, object]:
    """Extract a partial-charge curve `q(V)`.

    中文含义：
    - 只使用充电段，将累计输入电荷映射到电压网格。
    - 该曲线是可选的辅助物理 proxy；缺失时不丢弃样本。
    """

    df = cycle_raw_df.copy()
    charge_mask = _charge_mask(df)
    voltage = _safe_numeric(df, _find_column(df, ("voltage", "voltage_v", "ecell/v")))
    current = _safe_numeric(df, _find_column(df, ("current", "current_a", "<i>/ma", "current (ma)"))).abs()
    time_min = _infer_time_minutes(df)

    output_curve = np.zeros(int(voltage_grid_size), dtype=np.float32)
    output_stats = {"voltage_min": 0.0, "voltage_max": 0.0, "q_total": 0.0, "voltage_span": 0.0}
    if charge_mask.sum() < 3:
        return {"partial_charge_curve": output_curve, "partial_charge_mask": False, "partial_charge_stats": output_stats}

    v = voltage[charge_mask].to_numpy(dtype=np.float64)
    i_abs = current[charge_mask].to_numpy(dtype=np.float64)
    t = time_min[charge_mask.to_numpy()]
    valid = np.isfinite(v) & np.isfinite(i_abs) & np.isfinite(t)
    if valid.sum() < 3:
        return {"partial_charge_curve": output_curve, "partial_charge_mask": False, "partial_charge_stats": output_stats}
    v = v[valid]
    i_abs = i_abs[valid]
    t = t[valid]
    dt = np.clip(np.diff(t, prepend=t[0]), 0.0, None)
    q_cum = np.cumsum(i_abs * dt / 60.0)

    observed_min = float(np.nanmin(v))
    observed_max = float(np.nanmax(v))
    if voltage_min is None or voltage_max is None:
        if voltage_grid_mode == "chemistry_window":
            voltage_min = observed_min if voltage_min is None else voltage_min
            voltage_max = observed_max if voltage_max is None else voltage_max
        else:
            voltage_min = observed_min if voltage_min is None else voltage_min
            voltage_max = observed_max if voltage_max is None else voltage_max

    if not np.isfinite(voltage_min) or not np.isfinite(voltage_max) or (voltage_max - voltage_min) < 1e-3:
        return {"partial_charge_curve": output_curve, "partial_charge_mask": False, "partial_charge_stats": output_stats}

    order = np.argsort(v)
    v = v[order]
    q_cum = q_cum[order]
    unique_v, unique_idx = np.unique(v, return_index=True)
    q_cum = q_cum[unique_idx]
    if unique_v.size < 3 or (unique_v[-1] - unique_v[0]) < 5e-3:
        return {"partial_charge_curve": output_curve, "partial_charge_mask": False, "partial_charge_stats": output_stats}

    voltage_grid = np.linspace(float(voltage_min), float(voltage_max), int(voltage_grid_size), dtype=np.float32)
    curve = np.interp(voltage_grid, unique_v, q_cum, left=q_cum[0], right=q_cum[-1]).astype(np.float32)
    curve -= float(curve[0])
    output_stats = {
        "voltage_min": float(voltage_grid[0]),
        "voltage_max": float(voltage_grid[-1]),
        "q_total": float(curve[-1]),
        "voltage_span": float(voltage_grid[-1] - voltage_grid[0]),
    }
    return {"partial_charge_curve": curve, "partial_charge_mask": True, "partial_charge_stats": output_stats}


def compute_physics_features(
    partial_charge_curve: np.ndarray,
    partial_charge_mask: bool,
    qv_curve_stats: Dict[str, float],
) -> Dict[str, object]:
    """Compute named physical proxy features.

    输出 12 维特征：
    - 8 维 Q-V 极化 proxy：`delta_v_*`、`r_*` 和充放电曲线斜率；
    - 4 维 partial-charge summary：`q_total`、`q_mean`、`q_std`、`dq_dv_peak_value`。
    """

    features = np.zeros(len(PHYSICS_FEATURE_NAMES), dtype=np.float32)
    mask = np.zeros(len(PHYSICS_FEATURE_NAMES), dtype=np.float32)

    aliases = {
        "delta_v_q95": ("delta_v_q95", "delta_v_max"),
        "vc_curve_slope_mean": ("vc_curve_slope_mean", "vc_slope_mean"),
        "vd_curve_slope_mean": ("vd_curve_slope_mean", "vd_slope_mean"),
    }
    for idx, name in enumerate(PHYSICS_FEATURE_NAMES[:8]):
        keys = aliases.get(name, (name,))
        value = 0.0
        present = False
        for key in keys:
            if key in qv_curve_stats:
                value = qv_curve_stats[key]
                present = True
                break
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0.0
        if present and np.isfinite(value):
            features[idx] = np.float32(value)
            mask[idx] = 1.0

    partial_charge_curve = np.asarray(partial_charge_curve, dtype=np.float32)
    if bool(partial_charge_mask) and partial_charge_curve.size >= 4:
        q = partial_charge_curve.astype(np.float32)
        dq = np.gradient(q)
        peak_idx = int(np.argmax(dq))
        features[8:] = np.asarray([float(q[-1]), float(q.mean()), float(q.std()), float(dq[peak_idx])], dtype=np.float32)
        mask[8:] = 1.0

    return {
        "physics_features": features,
        "physics_feature_mask": mask,
        "physics_feature_names": list(PHYSICS_FEATURE_NAMES),
    }


def aggregate_window_physics_features(features_seq: np.ndarray, masks_seq: np.ndarray) -> Dict[str, object]:
    """Aggregate per-cycle physics features over a sliding window."""

    features_seq = np.asarray(features_seq, dtype=np.float32)
    masks_seq = np.asarray(masks_seq, dtype=np.float32)
    if features_seq.ndim != 2 or masks_seq.ndim != 2:
        raise ValueError("features_seq and masks_seq must be 2-D arrays")

    valid_row = masks_seq.sum(axis=1) > 0
    if valid_row.any():
        first_idx = int(np.flatnonzero(valid_row)[0])
        last_idx = int(np.flatnonzero(valid_row)[-1])
        valid_features = np.where(masks_seq > 0, features_seq, np.nan)
        mean_features = np.nanmean(valid_features, axis=0)
        std_features = np.nanstd(valid_features, axis=0)
        anchor_features = features_seq[last_idx]
        delta_features = features_seq[last_idx] - features_seq[first_idx]
    else:
        mean_features = np.zeros(features_seq.shape[1], dtype=np.float32)
        std_features = np.zeros(features_seq.shape[1], dtype=np.float32)
        anchor_features = np.zeros(features_seq.shape[1], dtype=np.float32)
        delta_features = np.zeros(features_seq.shape[1], dtype=np.float32)

    qv_physics_ratio = float(masks_seq[:, :8].mean()) if masks_seq.shape[1] >= 8 else 0.0
    partial_ratio = float(masks_seq[:, 8:].mean()) if masks_seq.shape[1] > 8 else 0.0
    availability_ratio = float((masks_seq > 0).mean()) if masks_seq.size else 0.0

    return {
        "anchor_physics_features": np.nan_to_num(anchor_features, nan=0.0).astype(np.float32),
        "mean_physics_features": np.nan_to_num(mean_features, nan=0.0).astype(np.float32),
        "std_physics_features": np.nan_to_num(std_features, nan=0.0).astype(np.float32),
        "delta_physics_features": np.nan_to_num(delta_features, nan=0.0).astype(np.float32),
        "physics_availability_ratio": availability_ratio,
        "qv_physics_availability_ratio": qv_physics_ratio,
        "partial_charge_availability_ratio": partial_ratio,
    }


def plot_partial_charge_curve(
    partial_charge_curve: np.ndarray,
    partial_charge_mask: bool,
    save_path: str | Path,
    title: str | None = None,
) -> None:
    """Save a matplotlib plot of the partial-charge curve."""

    curve = np.asarray(partial_charge_curve, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(6, 4.5), dpi=180)
    axis.set_title(title or "Partial charge curve")
    axis.set_xlabel("Voltage-grid index")
    axis.set_ylabel("Cumulative charge proxy")
    axis.grid(True, alpha=0.25)
    if partial_charge_mask:
        axis.plot(np.arange(len(curve)), curve, color="tab:blue", linewidth=1.8)
    else:
        axis.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=axis.transAxes)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
