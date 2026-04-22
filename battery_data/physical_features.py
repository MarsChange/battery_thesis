from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
            mins = (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
            return mins
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
    df = cycle_raw_df.copy()
    charge_mask = _charge_mask(df)
    voltage = _safe_numeric(df, _find_column(df, ("voltage", "voltage_v", "ecell/v")))
    current = _safe_numeric(df, _find_column(df, ("current", "current_a", "<i>/ma", "current (ma)"))).abs()
    time_min = _infer_time_minutes(df)

    output_curve = np.zeros(int(voltage_grid_size), dtype=np.float32)
    output_stats = {
        "voltage_min": 0.0,
        "voltage_max": 0.0,
        "q_total": 0.0,
        "voltage_span": 0.0,
    }
    if charge_mask.sum() < 3:
        return {
            "partial_charge_curve": output_curve,
            "partial_charge_mask": False,
            "partial_charge_stats": output_stats,
        }

    v = voltage[charge_mask].to_numpy(dtype=np.float64)
    i_abs = current[charge_mask].to_numpy(dtype=np.float64)
    t = time_min[charge_mask.to_numpy()]
    valid = np.isfinite(v) & np.isfinite(i_abs) & np.isfinite(t)
    if valid.sum() < 3:
        return {
            "partial_charge_curve": output_curve,
            "partial_charge_mask": False,
            "partial_charge_stats": output_stats,
        }
    v = v[valid]
    i_abs = i_abs[valid]
    t = t[valid]
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0.0, None)
    q_cum = np.cumsum(i_abs * dt / 60.0)

    v_observed_min = float(np.nanmin(v))
    v_observed_max = float(np.nanmax(v))
    if voltage_min is None or voltage_max is None:
        if voltage_grid_mode == "chemistry_window":
            voltage_min = v_observed_min if voltage_min is None else voltage_min
            voltage_max = v_observed_max if voltage_max is None else voltage_max
        else:
            voltage_min = v_observed_min if voltage_min is None else voltage_min
            voltage_max = v_observed_max if voltage_max is None else voltage_max

    if not np.isfinite(voltage_min) or not np.isfinite(voltage_max) or (voltage_max - voltage_min) < 1e-3:
        return {
            "partial_charge_curve": output_curve,
            "partial_charge_mask": False,
            "partial_charge_stats": output_stats,
        }

    order = np.argsort(v)
    v = v[order]
    q_cum = q_cum[order]
    unique_v, unique_idx = np.unique(v, return_index=True)
    q_cum = q_cum[unique_idx]
    if unique_v.size < 3 or (unique_v[-1] - unique_v[0]) < 5e-3:
        return {
            "partial_charge_curve": output_curve,
            "partial_charge_mask": False,
            "partial_charge_stats": output_stats,
        }

    voltage_grid = np.linspace(float(voltage_min), float(voltage_max), int(voltage_grid_size), dtype=np.float32)
    curve = np.interp(voltage_grid, unique_v, q_cum, left=q_cum[0], right=q_cum[-1]).astype(np.float32)
    curve -= float(curve[0])
    output_stats = {
        "voltage_min": float(voltage_grid[0]),
        "voltage_max": float(voltage_grid[-1]),
        "q_total": float(curve[-1]),
        "voltage_span": float(voltage_grid[-1] - voltage_grid[0]),
    }
    return {
        "partial_charge_curve": curve,
        "partial_charge_mask": True,
        "partial_charge_stats": output_stats,
    }


def extract_relaxation_curve(
    cycle_raw_df: pd.DataFrame,
    relax_points: int = 30,
    relax_time_max_minutes: float = 30.0,
    current_rest_threshold: float = 0.02,
) -> Dict[str, object]:
    df = cycle_raw_df.copy()
    current = _safe_numeric(df, _find_column(df, ("current", "current_a", "<i>/ma", "current (ma)")))
    voltage = _safe_numeric(df, _find_column(df, ("voltage", "voltage_v", "ecell/v")))
    time_min = _infer_time_minutes(df)
    charge_mask = _charge_mask(df)

    output_curve = np.zeros(int(relax_points), dtype=np.float32)
    output_stats = {
        "relaxation_delta_v": 0.0,
        "relaxation_span_minutes": 0.0,
        "relaxation_area": 0.0,
    }

    if charge_mask.sum() < 2:
        return {
            "relaxation_curve": output_curve,
            "relaxation_mask": False,
            "relaxation_stats": output_stats,
        }

    last_charge_idx = int(np.flatnonzero(charge_mask.to_numpy())[-1])
    post_df = df.iloc[last_charge_idx + 1 :].copy()
    if post_df.empty:
        return {
            "relaxation_curve": output_curve,
            "relaxation_mask": False,
            "relaxation_stats": output_stats,
        }

    post_current = current.iloc[last_charge_idx + 1 :].to_numpy(dtype=np.float64)
    post_voltage = voltage.iloc[last_charge_idx + 1 :].to_numpy(dtype=np.float64)
    post_time = time_min[last_charge_idx + 1 :]
    rest_mask = np.abs(post_current) < float(current_rest_threshold)
    if rest_mask.sum() < 3:
        return {
            "relaxation_curve": output_curve,
            "relaxation_mask": False,
            "relaxation_stats": output_stats,
        }

    rest_start = int(np.flatnonzero(rest_mask)[0])
    rest_voltage = post_voltage[rest_start:][rest_mask[rest_start:]]
    rest_time = post_time[rest_start:][rest_mask[rest_start:]]
    valid = np.isfinite(rest_voltage) & np.isfinite(rest_time)
    if valid.sum() < 3:
        return {
            "relaxation_curve": output_curve,
            "relaxation_mask": False,
            "relaxation_stats": output_stats,
        }
    rest_voltage = rest_voltage[valid]
    rest_time = rest_time[valid]
    rest_time = rest_time - float(rest_time[0])
    keep = rest_time <= float(relax_time_max_minutes)
    if keep.sum() < 3:
        return {
            "relaxation_curve": output_curve,
            "relaxation_mask": False,
            "relaxation_stats": output_stats,
        }
    rest_voltage = rest_voltage[keep]
    rest_time = rest_time[keep]
    if rest_time[-1] <= 0:
        return {
            "relaxation_curve": output_curve,
            "relaxation_mask": False,
            "relaxation_stats": output_stats,
        }
    grid = np.linspace(0.0, float(min(rest_time[-1], relax_time_max_minutes)), int(relax_points), dtype=np.float32)
    curve = np.interp(grid, rest_time, rest_voltage, left=rest_voltage[0], right=rest_voltage[-1]).astype(np.float32)
    output_stats = {
        "relaxation_delta_v": float(curve[-1] - curve[0]),
        "relaxation_span_minutes": float(grid[-1]),
        "relaxation_area": float(np.trapezoid(np.abs(curve - curve[-1]), grid)) if hasattr(np, "trapezoid") else float(np.trapz(np.abs(curve - curve[-1]), grid)),
    }
    return {
        "relaxation_curve": curve,
        "relaxation_mask": True,
        "relaxation_stats": output_stats,
    }


def compute_physics_features(
    partial_charge_curve: np.ndarray,
    partial_charge_mask: bool,
    relaxation_curve: np.ndarray,
    relaxation_mask: bool,
    qv_curve_stats: Dict[str, float],
) -> Dict[str, object]:
    partial_charge_curve = np.asarray(partial_charge_curve, dtype=np.float32)
    relaxation_curve = np.asarray(relaxation_curve, dtype=np.float32)

    feature_names = [
        "q_total",
        "q_mean",
        "q_std",
        "q_low_voltage_slope",
        "q_mid_voltage_slope",
        "q_high_voltage_slope",
        "dq_dv_peak_value",
        "dq_dv_peak_position",
        "relaxation_delta_v",
        "relaxation_initial_slope",
        "relaxation_late_slope",
        "relaxation_area_or_tau_proxy",
    ]
    features = np.zeros(len(feature_names), dtype=np.float32)
    mask = np.zeros(len(feature_names), dtype=np.float32)

    if bool(partial_charge_mask) and partial_charge_curve.size >= 4:
        q = partial_charge_curve.astype(np.float32)
        dq = np.gradient(q)
        segments = np.array_split(q, 3)
        slopes = [float(seg[-1] - seg[0]) / max(len(seg) - 1, 1) for seg in segments]
        peak_idx = int(np.argmax(dq))
        features[:8] = np.array(
            [
                float(q[-1]),
                float(q.mean()),
                float(q.std()),
                slopes[0] if len(slopes) > 0 else 0.0,
                slopes[1] if len(slopes) > 1 else 0.0,
                slopes[2] if len(slopes) > 2 else 0.0,
                float(dq[peak_idx]),
                float(peak_idx / max(len(q) - 1, 1)),
            ],
            dtype=np.float32,
        )
        mask[:8] = 1.0

    if bool(relaxation_mask) and relaxation_curve.size >= 4:
        rv = relaxation_curve.astype(np.float32)
        mid = max(len(rv) // 3, 1)
        late = max(2 * len(rv) // 3, 1)
        area = float(np.trapezoid(np.abs(rv - rv[-1]), dx=1.0)) if hasattr(np, "trapezoid") else float(np.trapz(np.abs(rv - rv[-1]), dx=1.0))
        features[8:] = np.array(
            [
                float(rv[-1] - rv[0]),
                float((rv[mid] - rv[0]) / max(mid, 1)),
                float((rv[-1] - rv[late]) / max(len(rv) - late - 1, 1)),
                area,
            ],
            dtype=np.float32,
        )
        mask[8:] = 1.0

    # If either curve family is missing, qv stats remain available separately through cycle stats.
    return {
        "physics_features": features,
        "physics_feature_mask": mask,
        "physics_feature_names": feature_names,
    }


def aggregate_window_physics_features(
    features_seq: np.ndarray,
    masks_seq: np.ndarray,
) -> Dict[str, object]:
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

    mean_features = np.nan_to_num(mean_features, nan=0.0).astype(np.float32)
    std_features = np.nan_to_num(std_features, nan=0.0).astype(np.float32)
    anchor_features = np.nan_to_num(anchor_features, nan=0.0).astype(np.float32)
    delta_features = np.nan_to_num(delta_features, nan=0.0).astype(np.float32)

    partial_ratio = float(masks_seq[:, :8].mean()) if masks_seq.shape[1] >= 8 else 0.0
    relax_ratio = float(masks_seq[:, 8:].mean()) if masks_seq.shape[1] > 8 else 0.0
    availability_ratio = float((masks_seq > 0).mean()) if masks_seq.size else 0.0

    return {
        "anchor_physics_features": anchor_features,
        "mean_physics_features": mean_features,
        "std_physics_features": std_features,
        "delta_physics_features": delta_features,
        "physics_availability_ratio": availability_ratio,
        "partial_charge_availability_ratio": partial_ratio,
        "relaxation_availability_ratio": relax_ratio,
    }


def plot_partial_charge_and_relaxation(
    partial_charge_curve: np.ndarray,
    partial_charge_mask: bool,
    relaxation_curve: np.ndarray,
    relaxation_mask: bool,
    save_path: str | Path,
    title: str | None = None,
) -> None:
    partial_charge_curve = np.asarray(partial_charge_curve, dtype=np.float32)
    relaxation_curve = np.asarray(relaxation_curve, dtype=np.float32)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)

    axes[0].set_title("Partial charge curve")
    axes[0].set_xlabel("Voltage-grid index")
    axes[0].set_ylabel("Cumulative charge proxy")
    axes[0].grid(True, alpha=0.25)
    if partial_charge_mask:
        axes[0].plot(np.arange(len(partial_charge_curve)), partial_charge_curve, color="tab:blue", linewidth=1.8)
    else:
        axes[0].text(0.5, 0.5, "unavailable", ha="center", va="center", transform=axes[0].transAxes)

    axes[1].set_title("Relaxation voltage curve")
    axes[1].set_xlabel("Relaxation-grid index")
    axes[1].set_ylabel("Voltage")
    axes[1].grid(True, alpha=0.25)
    if relaxation_mask:
        axes[1].plot(np.arange(len(relaxation_curve)), relaxation_curve, color="tab:orange", linewidth=1.8)
    else:
        axes[1].text(0.5, 0.5, "unavailable", ha="center", va="center", transform=axes[1].transAxes)

    figure.suptitle(title or "Partial charge and relaxation curves", fontsize=14)
    figure.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
