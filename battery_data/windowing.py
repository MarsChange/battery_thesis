from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .domain_labeling import build_domain_label
from .features import parse_json_list, recent_delta_mean, serialize_json
from .schema import BatteryMemorySample, DEFAULT_TOKEN_FEATURES


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
        if normalization == "cell_relative":
            if feature == "capacity":
                valid = values.replace([np.inf, -np.inf], np.nan).dropna()
                scale = float(valid.iloc[0]) if not valid.empty and valid.iloc[0] != 0 else 1.0
                values = values / scale
            elif feature in {
                "charge_throughput",
                "discharge_throughput",
                "energy_charge",
                "energy_discharge",
                "current_mean",
                "current_max",
                "current_min",
                "cc_time",
                "cv_time",
            }:
                valid = values.replace([np.inf, -np.inf], np.nan).abs().dropna()
                scale = float(valid.max()) if not valid.empty and valid.max() != 0 else 1.0
                values = values / scale
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

    manifest_map = split_manifest.set_index("cell_uid").to_dict(orient="index")
    memory_samples: List[BatteryMemorySample] = []
    records = []

    for cell_uid, cell_cycles in canonical_cycles.groupby("cell_uid", sort=True):
        cell_cycles = cell_cycles.sort_values("cycle_idx").reset_index(drop=True)
        manifest_row = manifest_map[cell_uid]
        split = manifest_row["split"]
        token_matrix = build_cycle_token_matrix(cell_cycles, token_features, normalization)
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
