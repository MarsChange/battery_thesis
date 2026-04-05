from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .domain_labeling import build_domain_label
from .features import augment_cycle_feature_frame, parse_json_list, recent_delta_mean, serialize_json
from .schema import BatteryMemorySample, DEFAULT_TOKEN_FEATURES


def _robust_center_scale(values: pd.Series) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return values
    median = float(valid.median())
    mad = float((valid - median).abs().median())
    if mad > 1e-8:
        return (values - median) / (1.4826 * mad)
    scale = float(valid.abs().quantile(0.95)) if len(valid) > 1 else float(abs(valid.iloc[0]))
    if scale < 1e-8:
        scale = 1.0
    return (values - median) / scale


def _quantile_abs_scale(values: pd.Series) -> pd.Series:
    valid = values.replace([np.inf, -np.inf], np.nan).abs().dropna()
    if valid.empty:
        return values
    scale = float(valid.quantile(0.95))
    if scale < 1e-8:
        scale = float(valid.max()) if not valid.empty else 1.0
    if scale < 1e-8:
        scale = 1.0
    return values / scale


def _normalize_feature_values(values: pd.Series, feature: str, normalization: str) -> pd.Series:
    if normalization != "cell_relative":
        return values

    if feature in {"soh", "capacity_ratio"}:
        return values
    if feature == "soh_pct":
        return values / 100.0
    if feature == "capacity":
        valid = values.replace([np.inf, -np.inf], np.nan).dropna()
        scale = float(valid.iloc[0]) if not valid.empty and abs(valid.iloc[0]) > 1e-8 else 1.0
        return values / scale

    if (
        feature.startswith("voltage_")
        and "_fft_" not in feature
        and not feature.endswith("_range")
        and "_diff_" not in feature
        and "_slope_" not in feature
        and "_std_" not in feature
    ):
        return _robust_center_scale(values)
    if (
        feature.startswith("temp_")
        and "_fft_" not in feature
        and not feature.endswith("_range")
        and "_diff_" not in feature
        and "_slope_" not in feature
        and "_std_" not in feature
    ):
        return _robust_center_scale(values)
    if feature in {"current_mean", "current_max", "current_min", "current_abs_mean"}:
        return _robust_center_scale(values)

    if (
        feature.endswith("_diff_1")
        or feature.endswith("_delta_1")
        or "_slope_" in feature
        or "_std_" in feature
        or feature.endswith("_range")
    ):
        return _quantile_abs_scale(values)

    if "_fft_entropy_" in feature or "_fft_low_ratio_" in feature:
        return values

    if feature in {
        "charge_throughput",
        "discharge_throughput",
        "energy_charge",
        "energy_discharge",
        "cc_time",
        "cv_time",
    }:
        return _quantile_abs_scale(values)

    return values


def build_cycle_token_matrix(
    cell_cycles: pd.DataFrame,
    feature_names: Sequence[str],
    normalization: str = "cell_relative",
) -> np.ndarray:
    columns = []
    for feature in feature_names:
        if feature not in cell_cycles.columns:
            values = pd.Series([0.0] * len(cell_cycles), dtype="float32")
        else:
            values = pd.to_numeric(cell_cycles[feature], errors="coerce").astype("float32")
        values = _normalize_feature_values(values, feature, normalization)
        values = values.fillna(0.0)
        columns.append(values.to_numpy(dtype=np.float32))
    return np.stack(columns, axis=-1)


def build_state_vector(
    cell_cycles: pd.DataFrame,
    end_idx: int,
    slope_window: int,
) -> Dict[str, float]:
    soh_values = pd.to_numeric(cell_cycles.loc[:end_idx, "soh"], errors="coerce").ffill().fillna(0.0)
    charge_tp = pd.to_numeric(cell_cycles.loc[:end_idx, "charge_throughput"], errors="coerce").ffill().fillna(0.0)
    discharge_tp = pd.to_numeric(cell_cycles.loc[:end_idx, "discharge_throughput"], errors="coerce").ffill().fillna(0.0)

    throughput_series = charge_tp if charge_tp.iloc[-1] != 0 else discharge_tp
    prev_idx = max(0, end_idx - slope_window)
    throughput_recent = float(throughput_series.iloc[-1] - throughput_series.iloc[prev_idx])

    return {
        "soh_now": float(soh_values.iloc[-1]),
        "recent_slope": float(recent_delta_mean(soh_values.values, slope_window)),
        "throughput_recent": throughput_recent,
    }


def build_memory_metadata(row: pd.Series) -> Dict[str, object]:
    return {
        "chemistry_family": row.get("chemistry_family"),
        "temperature_bucket": row.get("temperature_bucket"),
        "charge_rate_bucket": row.get("charge_rate_bucket"),
        "discharge_policy_family": row.get("discharge_policy_family"),
        "full_or_partial": row.get("full_or_partial"),
        "nominal_capacity_bucket": row.get("nominal_capacity_bucket"),
        "voltage_window_bucket": row.get("voltage_window_bucket"),
        "missing_mask": parse_json_list(row.get("missing_mask")),
    }


def build_memory_samples(
    canonical_cycles: pd.DataFrame,
    split_manifest: pd.DataFrame,
    memory_cfg: Dict[str, object],
    domain_rules: Dict[str, object] | None = None,
) -> Tuple[List[BatteryMemorySample], pd.DataFrame]:
    lookback = int(memory_cfg.get("lookback_length", 32))
    horizon = int(memory_cfg.get("prediction_length", 8))
    stride = int(memory_cfg.get("stride", 1))
    slope_window = int(memory_cfg.get("slope_window", 5))
    token_features = list(memory_cfg.get("token_features", DEFAULT_TOKEN_FEATURES))
    normalization = str(memory_cfg.get("token_normalization", "cell_relative"))
    derived_cfg = memory_cfg.get("derived_features", {})
    rolling_window = int(derived_cfg.get("rolling_window", 5))
    spectral_window = int(derived_cfg.get("spectral_window", 16))
    spectral_columns = list(derived_cfg.get("spectral_columns", ["voltage_mean", "temp_mean", "current_mean"]))

    manifest_map = split_manifest.set_index("cell_uid").to_dict(orient="index")
    memory_samples: List[BatteryMemorySample] = []
    records = []

    for cell_uid, cell_cycles in canonical_cycles.groupby("cell_uid", sort=True):
        cell_cycles = cell_cycles.sort_values("cycle_idx").reset_index(drop=True)
        manifest_row = manifest_map[cell_uid]
        split = manifest_row["split"]
        token_frame = augment_cycle_feature_frame(
            cell_cycles,
            rolling_window=rolling_window,
            spectral_window=spectral_window,
            spectral_columns=spectral_columns,
        )
        token_matrix = build_cycle_token_matrix(token_frame, token_features, normalization)
        soh_values = pd.to_numeric(cell_cycles["soh"], errors="coerce").to_numpy(dtype=np.float32)

        max_start = len(cell_cycles) - lookback - horizon + 1
        if max_start <= 0:
            continue

        for start in range(0, max_start, stride):
            window_end_exclusive = start + lookback
            target_end_exclusive = window_end_exclusive + horizon
            anchor_idx = window_end_exclusive - 1
            future_soh = soh_values[window_end_exclusive:target_end_exclusive]
            if np.isnan(soh_values[anchor_idx]) or np.isnan(future_soh).any():
                continue
            delta_soh = future_soh - soh_values[anchor_idx]
            row = cell_cycles.iloc[anchor_idx]
            state = build_state_vector(cell_cycles, anchor_idx, slope_window)
            metadata = build_memory_metadata(row)
            sample_domain_label = build_domain_label(metadata, domain_rules) or manifest_row["domain_label"]
            sample = BatteryMemorySample(
                cell_uid=cell_uid,
                split=split,
                window_start=start,
                window_end=window_end_exclusive,
                target_start=window_end_exclusive,
                target_end=target_end_exclusive,
                window_tokens=token_matrix[start:window_end_exclusive],
                state=state,
                delta_soh=delta_soh.astype(np.float32),
                metadata=metadata,
                domain_label=sample_domain_label,
            )
            memory_samples.append(sample)
            records.append(
                {
                    "cell_uid": cell_uid,
                    "split": split,
                    "window_start": start,
                    "window_end": window_end_exclusive,
                    "target_start": window_end_exclusive,
                    "target_end": target_end_exclusive,
                    "cycle_idx_start": int(cell_cycles.iloc[start]["cycle_idx"]),
                    "cycle_idx_end": int(cell_cycles.iloc[anchor_idx]["cycle_idx"]),
                    "d_i": sample_domain_label,
                    "s_i_json": serialize_json(state),
                    "m_i_json": serialize_json(metadata),
                    "delta_y_json": serialize_json(delta_soh.astype(float).tolist()),
                    "token_feature_names_json": serialize_json(list(token_features)),
                }
            )

    return memory_samples, pd.DataFrame(records)
