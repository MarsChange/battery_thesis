"""Random segment SOH prediction experiment.

This experiment supports the paper narrative of random-fragment multi-step SOH
forecasting rather than complete lifecycle prediction. A single MIT/LFP target
cell is selected, and one segment is sampled from each degradation stage
(`early`, `middle`, `late`). For each segment, the observed context is `N`
historical SOH values and the task is to forecast the next `K` SOH values.
Earlier long-horizon experiments required `K >= 2N`; that constraint is now an
explicit command-line option so equal-length settings such as 64->64 can be
used for controlled comparisons.

The trained forecasting model still consumes the numerical case-bank features
available in the initial window, including Q-V, partial-charge, physics,
operation, metadata and retrieved numerical references. During the fragment
forecast, only `soh_seq` is advanced with model predictions; the other
historical features remain those of the initial observed fragment. This keeps
the experiment leakage-safe and avoids pretending that future Q-V/operation
features are available.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from experiments.predict_cell_lifecycle import _collect_true_lifecycle, _read_case_rows
from forecasting.data import BatterySOHForecastDataset
from forecasting.metrics import regression_metrics
from forecasting.model import BatterySOHForecaster
from forecasting.routers import SEMANTIC_CONCEPT_NAMES
from forecasting.train import apply_model_init_config_overrides, load_config, move_batch_to_device, resolve_device


STAGE_BOUNDS = {
    "early": (0.10, 0.35),
    "middle": (0.35, 0.65),
    "late": (0.65, 0.90),
}


def _batchify(value):
    """Convert one dataset item into a batch with batch size one."""

    if isinstance(value, dict):
        return {key: _batchify(subvalue) for key, subvalue in value.items()}
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, np.generic):
        return torch.as_tensor(value.item()).unsqueeze(0)
    if isinstance(value, (int, float, bool)):
        return torch.as_tensor(value).unsqueeze(0)
    if isinstance(value, str):
        return [value]
    return value


def _clone_tensor_tree(value):
    """Clone tensor trees before per-segment mutation."""

    if isinstance(value, dict):
        return {key: _clone_tensor_tree(subvalue) for key, subvalue in value.items()}
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return list(value)
    return copy.deepcopy(value)


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _safe_slug(value: str) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _fit_initial_slope(history: np.ndarray) -> float:
    history = np.asarray(history, dtype=np.float32).reshape(-1)
    valid = np.isfinite(history)
    if valid.sum() < 3:
        return 0.0
    x = np.arange(history.size, dtype=np.float32)[valid]
    y = history[valid]
    return float(np.polyfit(x, y, deg=1)[0])


def _select_cell(
    rows: pd.DataFrame,
    *,
    source_dataset: str,
    chemistry: str,
    split: str,
    future_length: int,
    rng: np.random.Generator,
    cell_uid: str | None,
    min_stage_count: int = 2,
) -> str:
    subset = rows[
        (rows["split"].astype(str) == str(split))
        & (rows["chemistry_family"].astype(str).str.upper() == chemistry.upper())
    ].copy()
    if str(source_dataset).lower() not in {"any", "all", "*"}:
        subset = subset[subset["source_dataset"].astype(str).str.lower() == source_dataset.lower()].copy()
    if subset.empty:
        raise ValueError(f"No rows found for source_dataset={source_dataset!r}, chemistry={chemistry!r}, split={split!r}.")
    if cell_uid is not None:
        matched = subset[subset["cell_uid"].astype(str) == str(cell_uid)]
        if matched.empty:
            raise ValueError(f"cell_uid={cell_uid!r} is not available in the requested subset.")
        return str(cell_uid)

    candidates = []
    for candidate_cell, group in subset.groupby("cell_uid"):
        group = group.sort_values("cycle_idx_end")
        max_target = int(group["target_cycle_idx_end"].max())
        usable = group[group["cycle_idx_end"].astype(int) + int(future_length) <= max_target]
        stage_count = 0
        if not usable.empty:
            min_anchor = float(group["cycle_idx_end"].min())
            max_anchor = float(group["cycle_idx_end"].max())
            denom = max(max_anchor - min_anchor, 1.0)
            rel_pos = (usable["cycle_idx_end"].astype(float) - min_anchor) / denom
            for lo, hi in STAGE_BOUNDS.values():
                if bool(((rel_pos >= lo) & (rel_pos < hi)).any()):
                    stage_count += 1
        if stage_count >= int(min_stage_count):
            candidates.append((str(candidate_cell), int(stage_count), int(len(usable))))
    if not candidates:
        raise ValueError("No cell has enough rows for random segment prediction with the requested future length.")
    candidates = sorted(candidates, key=lambda item: (-item[1], -item[2], item[0]))
    top_cells = [cell for cell, stage_count, _ in candidates if stage_count == candidates[0][1]]
    return str(rng.choice(top_cells))


def _sample_random_cases(
    cell_rows: pd.DataFrame,
    *,
    future_length: int,
    rng: np.random.Generator,
    num_segments: int,
) -> Dict[str, pd.Series]:
    ordered = cell_rows.sort_values(["cycle_idx_end", "case_id"]).reset_index(drop=True)
    max_target = int(ordered["target_cycle_idx_end"].max())
    usable = ordered[ordered["cycle_idx_end"].astype(int) + int(future_length) <= max_target].copy()
    if usable.empty:
        raise ValueError("Selected cell has no random segment with enough future cycles.")
    count = min(int(num_segments), len(usable))
    picked = rng.choice(np.arange(len(usable)), size=count, replace=False)
    picked = sorted(int(idx) for idx in np.atleast_1d(picked))
    return {f"random_{pos + 1:02d}": usable.iloc[idx] for pos, idx in enumerate(picked)}


def _sample_stage_cases(
    cell_rows: pd.DataFrame,
    *,
    future_length: int,
    rng: np.random.Generator,
) -> Dict[str, pd.Series]:
    ordered = cell_rows.sort_values(["cycle_idx_end", "case_id"]).reset_index(drop=True)
    min_anchor = float(ordered["cycle_idx_end"].min())
    max_anchor = float(ordered["cycle_idx_end"].max())
    denom = max(max_anchor - min_anchor, 1.0)
    max_target = int(ordered["target_cycle_idx_end"].max())
    usable = ordered[ordered["cycle_idx_end"].astype(int) + int(future_length) <= max_target].copy()
    if usable.empty:
        raise ValueError("Selected cell has no segment with enough future cycles.")
    usable["relative_anchor_position"] = (usable["cycle_idx_end"].astype(float) - min_anchor) / denom

    selected: Dict[str, pd.Series] = {}
    for stage, (lo, hi) in STAGE_BOUNDS.items():
        stage_rows = usable[(usable["relative_anchor_position"] >= lo) & (usable["relative_anchor_position"] < hi)]
        if stage_rows.empty:
            continue
        selected[stage] = stage_rows.iloc[int(rng.integers(0, len(stage_rows)))]
    if not selected:
        selected["middle"] = usable.iloc[int(rng.integers(0, len(usable)))]
    return selected


def _semantic_text(concepts: Mapping[str, float]) -> Dict[str, str]:
    text = {}
    if concepts.get("concept_high_temperature", 0.0) >= 0.55:
        text["high_temperature_expert"] = "温度均值或最高温度偏高，支持高温因素专家。"
    if concepts.get("concept_high_current", 0.0) >= 0.55:
        text["high_current_expert"] = "绝对电流均值或峰值偏高，支持高电流因素专家。"
    if concepts.get("concept_high_cycle_aging", 0.0) >= 0.55:
        text["high_cycle_expert"] = "SOH 已进入更深退化区间或循环老化指数较高，支持高循环老化专家。"
    if concepts.get("concept_high_power", 0.0) >= 0.55:
        text["high_power_expert"] = "功率/能量吞吐 proxy 偏高，支持高功率因素专家。"
    if concepts.get("concept_low_retrieval_reliability", 0.0) >= 0.55:
        text["retrieval"] = "RAG 检索置信度偏低，top-k 参考案例应谨慎使用。"
    else:
        text["retrieval"] = "RAG 检索置信度没有被语义层判定为低可靠。"
    return text


def _finite_array(values: object) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return array[np.isfinite(array)]


def _nan_stat(values: object, fn=np.mean, default: float = float("nan")) -> float:
    array = _finite_array(values)
    if array.size == 0:
        return float(default)
    return float(fn(array))


def _fmt_float(value: object, digits: int = 4, unit: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not np.isfinite(number):
        return "unknown"
    suffix = f" {unit}" if unit else ""
    return f"{number:.{digits}f}{suffix}"


def _policy_summary_from_batch(
    base_batch: Dict[str, Dict[str, torch.Tensor]],
    initial_case: pd.Series,
) -> Dict[str, object]:
    """Summarize query-window operating policy using named case-bank features.

    The current case bank stores absolute current statistics rather than a
    signed charge/discharge split. The summary therefore reports an active-step
    current estimate (`max abs(I)`) and a window-average absolute current. This is
    explicit in the generated markdown so charge/discharge-current semantics are
    not overstated.
    """

    query = base_batch["query"]
    operation_seq = _to_numpy(query["operation_seq"][0]) if "operation_seq" in query else np.empty((0, 0))
    cycle_stats = _to_numpy(query["cycle_stats"][0]) if "cycle_stats" in query else np.empty((0, 0))
    physics = _to_numpy(query["anchor_physics_features"][0]) if "anchor_physics_features" in query else np.empty((0,))

    current_mean = _nan_stat(operation_seq[:, 0] if operation_seq.ndim == 2 and operation_seq.shape[1] > 0 else [])
    current_std = _nan_stat(operation_seq[:, 1] if operation_seq.ndim == 2 and operation_seq.shape[1] > 1 else [])
    current_max = _nan_stat(operation_seq[:, 2] if operation_seq.ndim == 2 and operation_seq.shape[1] > 2 else [])
    temp_mean = _nan_stat(operation_seq[:, 3] if operation_seq.ndim == 2 and operation_seq.shape[1] > 3 else [])
    temp_std = _nan_stat(operation_seq[:, 4] if operation_seq.ndim == 2 and operation_seq.shape[1] > 4 else [])
    temp_max = _nan_stat(operation_seq[:, 5] if operation_seq.ndim == 2 and operation_seq.shape[1] > 5 else [])
    energy_charge = _nan_stat(operation_seq[:, 6] if operation_seq.ndim == 2 and operation_seq.shape[1] > 6 else [])
    energy_discharge = _nan_stat(operation_seq[:, 7] if operation_seq.ndim == 2 and operation_seq.shape[1] > 7 else [])

    qv_voltage_min = _nan_stat(cycle_stats[:, 26] if cycle_stats.ndim == 2 and cycle_stats.shape[1] > 26 else [])
    qv_voltage_max = _nan_stat(cycle_stats[:, 27] if cycle_stats.ndim == 2 and cycle_stats.shape[1] > 27 else [])
    qv_window_ratio = _nan_stat(cycle_stats[:, 28] if cycle_stats.ndim == 2 and cycle_stats.shape[1] > 28 else [], np.mean)
    qv_peak_value = _nan_stat(cycle_stats[:, 22] if cycle_stats.ndim == 2 and cycle_stats.shape[1] > 22 else [])
    qv_peak_voltage = _nan_stat(cycle_stats[:, 23] if cycle_stats.ndim == 2 and cycle_stats.shape[1] > 23 else [])
    qv_area = _nan_stat(cycle_stats[:, 24] if cycle_stats.ndim == 2 and cycle_stats.shape[1] > 24 else [])

    full_or_partial = str(initial_case.get("full_or_partial", "unknown"))
    full_metadata = full_or_partial.lower() == "full"
    qv_available = np.isfinite(qv_window_ratio) and qv_window_ratio >= 0.5
    if full_metadata:
        full_text = "full charge/discharge: yes；metadata=full，按 case bank 标记为完整充放电窗口"
    elif qv_available:
        full_text = "full charge/discharge: not confirmed；metadata 非 full，但 Q-V 区段大部分可用"
    else:
        full_text = "full charge/discharge: unknown；Q-V 区段或 full/partial 标记不足"

    active_current = current_max if np.isfinite(current_max) else current_mean
    if np.isfinite(active_current):
        current_text = (
            f"active abs(I) max {_fmt_float(active_current, 3, 'A')}；"
            f"window mean abs(I) {_fmt_float(current_mean, 3, 'A')}，std {_fmt_float(current_std, 3, 'A')}"
        )
    else:
        current_text = "unknown；case bank 未提供可用电流统计"

    voltage_text = (
        f"{_fmt_float(qv_voltage_min, 3, 'V')} to {_fmt_float(qv_voltage_max, 3, 'V')}"
        if np.isfinite(qv_voltage_min) and np.isfinite(qv_voltage_max)
        else str(initial_case.get("voltage_window_bucket", "unknown"))
    )

    return {
        "policy_condition_label": str(initial_case.get("condition_label", "unknown")),
        "policy_raw_cell_id": str(initial_case.get("raw_cell_id", "unknown")),
        "policy_domain_label": str(initial_case.get("domain_label", "unknown")),
        "policy_temperature_bucket": str(initial_case.get("temperature_bucket", "unknown")),
        "policy_charge_rate_bucket": str(initial_case.get("charge_rate_bucket", "unknown")),
        "policy_voltage_window_bucket": str(initial_case.get("voltage_window_bucket", "unknown")),
        "policy_full_or_partial": full_or_partial,
        "policy_current_abs_mean_a": current_mean,
        "policy_current_abs_std_a": current_std,
        "policy_current_abs_max_a": current_max,
        "policy_active_current_estimate_a": active_current,
        "policy_temperature_mean_c": temp_mean,
        "policy_temperature_std_c": temp_std,
        "policy_temperature_max_c": temp_max,
        "policy_energy_charge_delta_1": energy_charge,
        "policy_energy_discharge_delta_1": energy_discharge,
        "policy_qv_voltage_min_v": qv_voltage_min,
        "policy_qv_voltage_max_v": qv_voltage_max,
        "policy_qv_window_available_ratio": qv_window_ratio,
        "policy_qv_dqdv_peak_value": qv_peak_value,
        "policy_qv_dqdv_peak_voltage": qv_peak_voltage,
        "policy_qv_dqdv_area": qv_area,
        "policy_current_text": current_text,
        "policy_temperature_text": (
            f"mean {_fmt_float(temp_mean, 2, 'degC')}；max {_fmt_float(temp_max, 2, 'degC')}；std {_fmt_float(temp_std, 2, 'degC')}"
        ),
        "policy_voltage_text": voltage_text,
        "policy_full_charge_discharge_text": full_text,
        "policy_current_semantics": "case bank stores absolute current statistics; signed charge and discharge currents are not separated in this summary.",
        "policy_anchor_physics_current_abs_max_a": float(physics[9]) if physics.size > 9 and np.isfinite(physics[9]) else float("nan"),
    }


def _predict_segment(
    *,
    model: BatterySOHForecaster,
    base_batch: Dict[str, Dict[str, torch.Tensor]],
    initial_case: pd.Series,
    truth_map: Mapping[int, float],
    future_length: int,
    history_length: int,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, object]]]:
    current_soh_seq = base_batch["query"]["soh_seq"][0].detach().clone().to(torch.float32)
    if current_soh_seq.shape[0] != history_length:
        raise ValueError(f"history_length={history_length} does not match case-bank lookback={current_soh_seq.shape[0]}.")
    current_anchor_soh = torch.tensor([float(initial_case["anchor_soh"])], dtype=torch.float32, device=device)
    current_anchor_cycle = int(initial_case["cycle_idx_end"])
    initial_anchor_cycle = int(initial_case["cycle_idx_end"])
    initial_anchor_soh = float(initial_case["anchor_soh"])
    initial_slope = _fit_initial_slope(_to_numpy(current_soh_seq).reshape(-1))
    horizon = int(base_batch["query"]["target_delta_soh"].shape[-1])
    expert_names = list(model.expert_names)
    query_policy_summary = _policy_summary_from_batch(base_batch, initial_case)

    observed_records = []
    hist_start = int(initial_case["cycle_idx_start"])
    for offset, soh in enumerate(_to_numpy(current_soh_seq).reshape(-1)):
        observed_records.append(
            {
                "cycle_idx": int(hist_start + offset),
                "stage": str(initial_case["sampled_stage"]),
                "segment_part": "history",
                "observed_soh": float(soh),
                "true_soh": float(truth_map.get(int(hist_start + offset), np.nan)),
                "pred_soh": np.nan,
                "persistence_soh": np.nan,
                "linear_slope_soh": np.nan,
            }
        )

    prediction_records = []
    block_records: List[Dict[str, object]] = []
    remaining = int(future_length)
    block_index = 0
    with torch.no_grad():
        while remaining > 0:
            batch = _clone_tensor_tree(base_batch)
            batch["query"]["soh_seq"] = current_soh_seq.unsqueeze(0)
            batch["query"]["anchor_soh"] = current_anchor_soh.clone()
            if "expert_seq" in batch["query"]:
                expert_seq = batch["query"]["expert_seq"].clone()
                if expert_seq.shape[-1] > 0 and expert_seq.shape[1] == current_soh_seq.shape[0]:
                    expert_seq[0, :, 0] = current_soh_seq[:, 0]
                batch["query"]["expert_seq"] = expert_seq

            outputs = model(batch)
            pred_soh = _to_numpy(outputs["pred_soh"][0]).reshape(-1)
            pred_delta = _to_numpy(outputs["pred_delta"][0]).reshape(-1)
            base_delta = _to_numpy(outputs["base_delta"][0]).reshape(-1)
            residual = _to_numpy(outputs["moe_residual"][0]).reshape(-1)
            residual_gate = _to_numpy(outputs.get("residual_gate", torch.ones_like(outputs["retrieval_confidence"]))[0]).reshape(-1)
            residual_retrieval_gate = _to_numpy(outputs.get("residual_retrieval_gate", torch.ones_like(outputs["retrieval_confidence"]))[0]).reshape(-1)
            residual_reference_agreement_gate = _to_numpy(
                outputs.get("residual_reference_agreement_gate", torch.ones_like(outputs["retrieval_confidence"]))[0]
            ).reshape(-1)
            reference_future_delta_dispersion = _to_numpy(
                outputs.get("reference_future_delta_dispersion", torch.zeros_like(outputs["retrieval_confidence"]))[0]
            ).reshape(-1)
            residual_stage_gate = _to_numpy(outputs.get("residual_stage_gate", torch.ones_like(outputs["retrieval_confidence"]))[0]).reshape(-1)
            residual_trend_anchor_cap = _to_numpy(
                outputs.get("residual_trend_anchor_cap", torch.full_like(outputs["moe_residual"], float("nan")))[0]
            ).reshape(-1)
            anchor_value = float(current_anchor_soh.detach().cpu().item())
            base_soh = anchor_value + base_delta
            step_count = min(horizon, remaining)
            expert_weights_by_horizon = _to_numpy(
                outputs.get("expert_weights_by_horizon", outputs["expert_weights"].unsqueeze(1))[0]
            )
            if expert_weights_by_horizon.ndim == 1:
                expert_weights_by_horizon = np.repeat(expert_weights_by_horizon.reshape(1, -1), horizon, axis=0)
            expert_weights = expert_weights_by_horizon[:step_count].mean(axis=0)
            concepts = _to_numpy(outputs["semantic_concepts"][0]).reshape(-1)
            retrieval_confidence = float(_to_numpy(outputs["retrieval_confidence"][0]).reshape(-1)[0])
            composite_distance = _to_numpy(batch["retrieval"]["composite_distance"][0]).reshape(-1)
            valid_dist = composite_distance[np.isfinite(composite_distance)]

            block_record: Dict[str, object] = {
                "stage": str(initial_case["sampled_stage"]),
                "case_id": int(initial_case["case_id"]),
                "block_index": int(block_index),
                "anchor_cycle_idx": int(current_anchor_cycle),
                "anchor_soh": float(current_anchor_soh.detach().cpu().item()),
                "first_pred_cycle_idx": int(current_anchor_cycle + 1),
                "last_pred_cycle_idx": int(current_anchor_cycle + step_count),
                "retrieval_confidence": retrieval_confidence,
                "topk_mean_composite_distance": float(np.mean(valid_dist)) if valid_dist.size else float("nan"),
                "topk_std_composite_distance": float(np.std(valid_dist)) if valid_dist.size else float("nan"),
                "residual_gate": float(residual_gate[0]) if residual_gate.size else float("nan"),
                "residual_retrieval_gate": float(residual_retrieval_gate[0]) if residual_retrieval_gate.size else float("nan"),
                "residual_reference_agreement_gate": float(residual_reference_agreement_gate[0]) if residual_reference_agreement_gate.size else float("nan"),
                "reference_future_delta_dispersion": float(reference_future_delta_dispersion[0]) if reference_future_delta_dispersion.size else float("nan"),
                "residual_stage_gate": float(residual_stage_gate[0]) if residual_stage_gate.size else float("nan"),
                "residual_trend_anchor_cap_mean": float(np.nanmean(residual_trend_anchor_cap[:step_count])),
                "base_delta_mean": float(np.mean(base_delta[:step_count])),
                "residual_mean": float(np.mean(residual[:step_count])),
                "residual_abs_mean": float(np.mean(np.abs(residual[:step_count]))),
                "residual_step_abs_mean": float(np.mean(np.abs(np.diff(residual[:step_count])))) if step_count >= 2 else 0.0,
                "pred_delta_last": float(pred_delta[step_count - 1]),
                "pred_soh_last": float(pred_soh[step_count - 1]),
            }
            block_record.update(query_policy_summary)
            for pos, name in enumerate(expert_names):
                block_record[f"expert_weight_{name}"] = float(expert_weights[pos]) if pos < expert_weights.size else float("nan")
                if pos < expert_weights_by_horizon.shape[-1]:
                    block_record[f"expert_weight_{name}_first"] = float(expert_weights_by_horizon[0, pos])
                    block_record[f"expert_weight_{name}_last"] = float(expert_weights_by_horizon[step_count - 1, pos])
            concept_map = {}
            for pos, name in enumerate(SEMANTIC_CONCEPT_NAMES):
                value = float(concepts[pos]) if pos < concepts.size else float("nan")
                block_record[f"concept_{name}"] = value
                concept_map[name] = value
            block_record["semantic_evidence_json"] = json.dumps(_semantic_text(concept_map), ensure_ascii=False)
            block_records.append(block_record)

            for horizon_idx in range(step_count):
                cycle_idx = current_anchor_cycle + horizon_idx + 1
                true_soh = float(truth_map.get(cycle_idx, np.nan))
                prediction = float(pred_soh[horizon_idx])
                persistence = initial_anchor_soh
                linear = initial_anchor_soh + initial_slope * float(cycle_idx - initial_anchor_cycle)
                step_record = {
                        "cycle_idx": int(cycle_idx),
                        "stage": str(initial_case["sampled_stage"]),
                        "segment_part": "future",
                        "observed_soh": np.nan,
                        "true_soh": true_soh,
                        "pred_soh": prediction,
                        "base_soh": float(base_soh[horizon_idx]),
                        "residual_delta": float(residual[horizon_idx]),
                        "pred_minus_base": float(pred_soh[horizon_idx] - base_soh[horizon_idx]),
                        "persistence_soh": float(persistence),
                        "linear_slope_soh": float(linear),
                        "absolute_error": abs(prediction - true_soh) if np.isfinite(true_soh) else np.nan,
                        "base_absolute_error": abs(float(base_soh[horizon_idx]) - true_soh) if np.isfinite(true_soh) else np.nan,
                        "persistence_absolute_error": abs(persistence - true_soh) if np.isfinite(true_soh) else np.nan,
                        "linear_slope_absolute_error": abs(linear - true_soh) if np.isfinite(true_soh) else np.nan,
                        "horizon_from_segment_start": int(horizon_idx + 1 + block_index * horizon),
                    }
                for pos, name in enumerate(expert_names):
                    if pos < expert_weights_by_horizon.shape[-1]:
                        step_record[f"expert_weight_{name}"] = float(expert_weights_by_horizon[horizon_idx, pos])
                prediction_records.append(step_record)

            appended = torch.as_tensor(pred_soh[:step_count], dtype=torch.float32, device=device).view(-1, 1)
            current_soh_seq = torch.cat([current_soh_seq, appended], dim=0)[-history_length:]
            current_anchor_soh = appended[-1:].reshape(1)
            current_anchor_cycle += step_count
            remaining -= step_count
            block_index += 1

    return pd.DataFrame(observed_records), pd.DataFrame(prediction_records), block_records


def _plot_segments(
    segment_frame: pd.DataFrame,
    output_path: Path,
    *,
    cell_uid: str,
    source_dataset: str,
    chemistry: str,
    history_length: int,
    future_length: int,
) -> None:
    stages = [stage for stage in ["early", "middle", "late"] if stage in set(segment_frame["stage"].astype(str))]
    if not stages:
        stages = sorted(segment_frame["stage"].astype(str).unique().tolist())
    figure, axes = plt.subplots(len(stages), 1, figsize=(12, 4.1 * len(stages)), dpi=180, sharex=False)
    if len(stages) == 1:
        axes = [axes]
    for axis, stage in zip(axes, stages):
        sub = segment_frame[segment_frame["stage"].astype(str) == stage].sort_values("cycle_idx")
        hist = sub[sub["segment_part"] == "history"]
        fut = sub[sub["segment_part"] == "future"]
        axis.plot(hist["cycle_idx"], hist["true_soh"], color="#1f2933", linewidth=2.3, label=f"Observed history SOH (N={history_length})")
        axis.plot(fut["cycle_idx"], fut["true_soh"], color="#111827", linewidth=2.0, alpha=0.75, label="True future SOH")
        if "base_soh" in fut.columns:
            axis.plot(fut["cycle_idx"], fut["base_soh"], color="#2563eb", linewidth=2.0, alpha=0.9, label="Base-only prediction")
        axis.plot(fut["cycle_idx"], fut["pred_soh"], color="#d62728", linewidth=2.0, alpha=0.9, label=f"Final prediction: base + residual (K={future_length})")
        axis.plot(fut["cycle_idx"], fut["persistence_soh"], color="#64748b", linestyle="--", linewidth=1.2, label="Persistence baseline")
        axis.plot(fut["cycle_idx"], fut["linear_slope_soh"], color="#7c3aed", linestyle=":", linewidth=1.4, label="Initial-slope baseline")
        if not hist.empty:
            axis.axvline(int(hist["cycle_idx"].max()), color="#475569", linestyle="--", linewidth=1.0, alpha=0.65)
        axis.set_title(f"{stage} degradation segment")
        axis.set_xlabel("Cycle index")
        axis.set_ylabel("SOH")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle(f"Random segment SOH prediction | {cell_uid} | {source_dataset}/{chemistry}", y=1.01)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_expert_weights(
    blocks: pd.DataFrame,
    output_path: Path,
    expert_names: List[str],
    segment_frame: pd.DataFrame | None = None,
) -> None:
    stages = [stage for stage in ["early", "middle", "late"] if stage in set(blocks["stage"].astype(str))]
    if not stages:
        stages = sorted(blocks["stage"].astype(str).unique().tolist())
    colors = {
        "high_temperature_expert": "#dc2626",
        "high_current_expert": "#2563eb",
        "high_cycle_expert": "#16a34a",
        "high_power_expert": "#f59e0b",
    }
    figure, axes = plt.subplots(len(stages), 1, figsize=(12, 3.6 * len(stages)), dpi=180, sharex=False)
    if len(stages) == 1:
        axes = [axes]
    for axis, stage in zip(axes, stages):
        if segment_frame is not None and not segment_frame.empty:
            sub_steps = segment_frame[
                (segment_frame["stage"].astype(str) == stage)
                & (segment_frame["segment_part"].astype(str) == "future")
            ].sort_values("cycle_idx")
        else:
            sub_steps = pd.DataFrame()
        sub = blocks[blocks["stage"].astype(str) == stage].sort_values("anchor_cycle_idx")
        for name in expert_names:
            col = f"expert_weight_{name}"
            if col in sub_steps.columns:
                axis.plot(
                    sub_steps["cycle_idx"],
                    sub_steps[col],
                    color=colors.get(name),
                    linewidth=1.7,
                    label=name,
                )
            elif col in sub.columns:
                axis.plot(
                    sub["anchor_cycle_idx"],
                    sub[col],
                    color=colors.get(name),
                    linewidth=1.8,
                    marker="o",
                    markersize=4.5,
                    label=name,
                )
        axis.set_title(f"{stage} segment step-wise expert weights")
        axis.set_xlabel("Forecast cycle index")
        axis.set_ylabel("Expert weight")
        axis.set_ylim(-0.02, 1.02)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _collect_retrieval_topk_segments(
    *,
    base_batch: Dict[str, Dict[str, torch.Tensor]],
    initial_case: pd.Series,
    rows: pd.DataFrame,
    history_length: int,
    future_length: int,
) -> pd.DataFrame:
    """Collect query and top-k retrieved 32->64 segments in aligned coordinates.

    The y-axis uses delta SOH relative to each segment anchor, so query and
    retrieved references can be compared in one coordinate system without
    leaking absolute target-cell identity.
    """

    query = base_batch["query"]
    retrieval = base_batch["retrieval"]
    case_id_to_meta = rows.set_index("case_id", drop=False)
    records: List[Dict[str, object]] = []

    stage = str(initial_case["sampled_stage"])
    query_case_id = int(initial_case["case_id"])
    query_anchor_soh = float(_to_numpy(query["anchor_soh"]).reshape(-1)[0])
    query_history = _to_numpy(query["soh_seq"][0]).reshape(-1)[-history_length:]
    query_history_delta = query_history - query_anchor_soh
    query_future_soh = _to_numpy(query["target_soh"][0]).reshape(-1)[:future_length]
    query_label = f"Query | {initial_case['cell_uid']} | {initial_case['source_dataset']}/{initial_case['chemistry_family']}"

    for pos, soh in enumerate(query_history):
        rel_step = int(pos - len(query_history) + 1)
        records.append(
            {
                "stage": stage,
                "query_case_id": query_case_id,
                "series_role": "query_history",
                "series_label": query_label,
                "neighbor_rank": -1,
                "neighbor_case_id": -1,
                "relative_step": rel_step,
                "delta_soh_from_anchor": float(soh - query_anchor_soh),
                "absolute_soh": float(soh),
                "anchor_soh": query_anchor_soh,
                "retrieval_alpha": np.nan,
                "composite_distance": np.nan,
                "d_soh_state": np.nan,
                "d_qv_shape": np.nan,
                "d_physics": np.nan,
                "d_metadata": np.nan,
                "history_shape_rmse": np.nan,
                "source_dataset": str(initial_case["source_dataset"]),
                "chemistry_family": str(initial_case["chemistry_family"]),
                "cell_uid": str(initial_case["cell_uid"]),
            }
        )
    for horizon_idx, soh in enumerate(query_future_soh):
        records.append(
            {
                "stage": stage,
                "query_case_id": query_case_id,
                "series_role": "query_future_true",
                "series_label": query_label,
                "neighbor_rank": -1,
                "neighbor_case_id": -1,
                "relative_step": int(horizon_idx + 1),
                "delta_soh_from_anchor": float(soh - query_anchor_soh),
                "absolute_soh": float(soh),
                "anchor_soh": query_anchor_soh,
                "retrieval_alpha": np.nan,
                "composite_distance": np.nan,
                "d_soh_state": np.nan,
                "d_qv_shape": np.nan,
                "d_physics": np.nan,
                "d_metadata": np.nan,
                "history_shape_rmse": np.nan,
                "source_dataset": str(initial_case["source_dataset"]),
                "chemistry_family": str(initial_case["chemistry_family"]),
                "cell_uid": str(initial_case["cell_uid"]),
            }
        )

    neighbor_case_ids = _to_numpy(retrieval["neighbor_case_ids"][0]).reshape(-1).astype(int)
    retrieval_mask = _to_numpy(retrieval["retrieval_mask"][0]).reshape(-1)
    retrieval_alpha = _to_numpy(retrieval["retrieval_alpha"][0]).reshape(-1)
    composite_distance = _to_numpy(retrieval["composite_distance"][0]).reshape(-1)
    distance_columns = {
        "d_soh_state": _to_numpy(retrieval.get("d_soh_state", torch.full_like(retrieval["composite_distance"], np.nan))[0]).reshape(-1),
        "d_qv_shape": _to_numpy(retrieval.get("d_qv_shape", torch.full_like(retrieval["composite_distance"], np.nan))[0]).reshape(-1),
        "d_physics": _to_numpy(retrieval.get("d_physics", torch.full_like(retrieval["composite_distance"], np.nan))[0]).reshape(-1),
        "d_metadata": _to_numpy(retrieval.get("d_metadata", torch.full_like(retrieval["composite_distance"], np.nan))[0]).reshape(-1),
    }
    ref_anchor_soh = _to_numpy(retrieval["ref_anchor_soh"][0]).reshape(-1)
    ref_history = _to_numpy(retrieval["ref_soh_seq"][0]).reshape(len(neighbor_case_ids), -1)
    ref_future_delta = _to_numpy(retrieval["ref_future_delta_soh"][0]).reshape(len(neighbor_case_ids), -1)

    for rank, neighbor_case_id in enumerate(neighbor_case_ids):
        if neighbor_case_id < 0 or rank >= len(retrieval_mask) or retrieval_mask[rank] <= 0:
            continue
        if neighbor_case_id in case_id_to_meta.index:
            meta = case_id_to_meta.loc[neighbor_case_id]
            neighbor_cell = str(meta.get("cell_uid", "unknown"))
            neighbor_dataset = str(meta.get("source_dataset", "unknown"))
            neighbor_chemistry = str(meta.get("chemistry_family", "unknown"))
        else:
            neighbor_cell = "unknown"
            neighbor_dataset = "unknown"
            neighbor_chemistry = "unknown"
        ref_anchor = float(ref_anchor_soh[rank])
        history = ref_history[rank, -history_length:]
        history_delta = history - ref_anchor
        if history_delta.shape == query_history_delta.shape:
            history_shape_rmse = float(np.sqrt(np.mean((history_delta - query_history_delta) ** 2)))
        else:
            history_shape_rmse = float("nan")
        ref_label = (
            f"Top-{rank + 1} | {neighbor_cell} | {neighbor_dataset}/{neighbor_chemistry} "
            f"| d={float(composite_distance[rank]):.3f} | hRMSE={history_shape_rmse:.4f} "
            f"| alpha={float(retrieval_alpha[rank]):.2f}"
        )
        future_delta = ref_future_delta[rank, :future_length]
        for pos, soh in enumerate(history):
            records.append(
                {
                    "stage": stage,
                    "query_case_id": query_case_id,
                    "series_role": "reference_history",
                    "series_label": ref_label,
                    "neighbor_rank": int(rank + 1),
                    "neighbor_case_id": int(neighbor_case_id),
                    "relative_step": int(pos - len(history) + 1),
                    "delta_soh_from_anchor": float(soh - ref_anchor),
                    "absolute_soh": float(soh),
                    "anchor_soh": ref_anchor,
                    "retrieval_alpha": float(retrieval_alpha[rank]),
                    "composite_distance": float(composite_distance[rank]),
                    **{name: float(values[rank]) for name, values in distance_columns.items()},
                    "history_shape_rmse": history_shape_rmse,
                    "source_dataset": neighbor_dataset,
                    "chemistry_family": neighbor_chemistry,
                    "cell_uid": neighbor_cell,
                }
            )
        for horizon_idx, delta in enumerate(future_delta):
            records.append(
                {
                    "stage": stage,
                    "query_case_id": query_case_id,
                    "series_role": "reference_future_delta",
                    "series_label": ref_label,
                    "neighbor_rank": int(rank + 1),
                    "neighbor_case_id": int(neighbor_case_id),
                    "relative_step": int(horizon_idx + 1),
                    "delta_soh_from_anchor": float(delta),
                    "absolute_soh": float(ref_anchor + delta),
                    "anchor_soh": ref_anchor,
                    "retrieval_alpha": float(retrieval_alpha[rank]),
                    "composite_distance": float(composite_distance[rank]),
                    **{name: float(values[rank]) for name, values in distance_columns.items()},
                    "history_shape_rmse": history_shape_rmse,
                    "source_dataset": neighbor_dataset,
                    "chemistry_family": neighbor_chemistry,
                    "cell_uid": neighbor_cell,
                }
            )
    return pd.DataFrame(records)


def _plot_retrieval_topk_segments(
    retrieval_frame: pd.DataFrame,
    output_path: Path,
    *,
    cell_uid: str,
    source_dataset: str,
    chemistry: str,
    history_length: int,
    future_length: int,
) -> None:
    if retrieval_frame.empty:
        return
    stages = [stage for stage in ["early", "middle", "late"] if stage in set(retrieval_frame["stage"].astype(str))]
    if not stages:
        stages = sorted(retrieval_frame["stage"].astype(str).unique().tolist())
    figure, axes = plt.subplots(len(stages), 1, figsize=(12.5, 4.2 * len(stages)), dpi=180, sharex=True)
    if len(stages) == 1:
        axes = [axes]
    colors = ["#2563eb", "#16a34a", "#9333ea", "#f59e0b", "#0891b2", "#be123c", "#64748b", "#84cc16"]
    for axis, stage in zip(axes, stages):
        sub = retrieval_frame[retrieval_frame["stage"].astype(str) == stage].copy()
        query_hist = sub[sub["series_role"] == "query_history"].sort_values("relative_step")
        query_future = sub[sub["series_role"] == "query_future_true"].sort_values("relative_step")
        axis.plot(
            query_hist["relative_step"],
            query_hist["delta_soh_from_anchor"],
            color="#111827",
            linewidth=2.6,
            label=f"Query history (N={history_length})",
        )
        axis.plot(
            query_future["relative_step"],
            query_future["delta_soh_from_anchor"],
            color="#111827",
            linewidth=2.2,
            alpha=0.75,
            label=f"Query true future (K={future_length})",
        )
        ref_labels = (
            sub[sub["neighbor_rank"] > 0][["neighbor_rank", "series_label"]]
            .drop_duplicates()
            .sort_values("neighbor_rank")
            .to_records(index=False)
        )
        for idx, (rank, label) in enumerate(ref_labels):
            ref = sub[sub["neighbor_rank"] == int(rank)].sort_values("relative_step")
            hist = ref[ref["series_role"] == "reference_history"]
            fut = ref[ref["series_role"] == "reference_future_delta"]
            color = colors[idx % len(colors)]
            alpha = max(0.25, 0.82 - 0.06 * idx)
            axis.plot(hist["relative_step"], hist["delta_soh_from_anchor"], color=color, linestyle="--", linewidth=1.2, alpha=alpha)
            axis.plot(fut["relative_step"], fut["delta_soh_from_anchor"], color=color, linewidth=1.6, alpha=alpha, label=str(label))
        axis.axvline(0, color="#475569", linestyle="--", linewidth=1.0, alpha=0.65)
        axis.axhline(0, color="#94a3b8", linestyle=":", linewidth=0.9, alpha=0.7)
        axis.set_title(f"{stage} query vs retrieved top-k segments")
        axis.set_ylabel("Delta SOH from anchor")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=7)
    axes[-1].set_xlabel("Relative step: history <= 0, future > 0")
    figure.suptitle(f"Retrieved reference segments | {cell_uid} | {source_dataset}/{chemistry}", y=1.01)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


EXPERT_FACTOR_LABELS = {
    "high_temperature_expert": "temperature stress",
    "high_current_expert": "current stress",
    "high_cycle_expert": "cycle-aging / low-SOH stress",
    "high_power_expert": "power-throughput stress",
}

EXPERT_FACTOR_LABELS_ZH = {
    "high_temperature_expert": "高温度因素",
    "high_current_expert": "高电流因素",
    "high_cycle_expert": "高循环/低 SOH 因素",
    "high_power_expert": "高功率/能量吞吐因素",
}

EXPERT_TO_CONCEPT_COLUMN = {
    "high_temperature_expert": "concept_concept_high_temperature",
    "high_current_expert": "concept_concept_high_current",
    "high_cycle_expert": "concept_concept_high_cycle_aging",
    "high_power_expert": "concept_concept_high_power",
}


def _base_expert_columns(frame: pd.DataFrame) -> List[str]:
    columns = []
    for col in frame.columns:
        if not col.startswith("expert_weight_"):
            continue
        if col.endswith("_first") or col.endswith("_last"):
            continue
        columns.append(col)
    return sorted(columns)


def _weight_change_table(stage: str, blocks: pd.DataFrame, segment_frame: pd.DataFrame | None) -> tuple[List[str], str]:
    if segment_frame is not None and not segment_frame.empty:
        sub_steps = segment_frame[
            (segment_frame["stage"].astype(str) == str(stage))
            & (segment_frame["segment_part"].astype(str) == "future")
        ].sort_values("cycle_idx")
        weight_columns = _base_expert_columns(sub_steps)
        if not sub_steps.empty and weight_columns:
            rows = []
            means = {}
            for col in weight_columns:
                expert = col.replace("expert_weight_", "")
                values = sub_steps[col].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    continue
                means[expert] = float(np.mean(values))
                rows.append(
                    f"| {expert} | {_fmt_float(values[0], 3)} | {_fmt_float(np.mean(values), 3)} | "
                    f"{_fmt_float(values[-1], 3)} | {_fmt_float(values[-1] - values[0], 3)} |"
                )
            dominant = max(means.items(), key=lambda item: item[1])[0] if means else "unknown"
            return rows, dominant

    sub = blocks[blocks["stage"].astype(str) == str(stage)].sort_values("block_index")
    if sub.empty:
        return [], "unknown"
    first = sub.iloc[0]
    rows = []
    means = {}
    for col in _base_expert_columns(sub):
        expert = col.replace("expert_weight_", "")
        values = sub[col].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        first_weight = float(first.get(f"{col}_first", values[0]))
        last_weight = float(first.get(f"{col}_last", values[-1]))
        means[expert] = float(np.mean(values))
        rows.append(
            f"| {expert} | {_fmt_float(first_weight, 3)} | {_fmt_float(np.mean(values), 3)} | "
            f"{_fmt_float(last_weight, 3)} | {_fmt_float(last_weight - first_weight, 3)} |"
        )
    dominant = max(means.items(), key=lambda item: item[1])[0] if means else "unknown"
    return rows, dominant


def _describe_router_stage(stage: str, first: pd.Series, dominant_expert: str) -> List[str]:
    anchor_soh = float(first.get("anchor_soh", float("nan")))
    pred_soh_last = float(first.get("pred_soh_last", float("nan")))
    base_delta_mean = float(first.get("base_delta_mean", float("nan")))
    residual_mean = float(first.get("residual_mean", float("nan")))
    residual_abs_mean = float(first.get("residual_abs_mean", float("nan")))
    retrieval_conf = float(first.get("retrieval_confidence", float("nan")))
    current = float(first.get("policy_active_current_estimate_a", float("nan")))
    temp_mean = float(first.get("policy_temperature_mean_c", float("nan")))
    temp_max = float(first.get("policy_temperature_max_c", float("nan")))
    power_proxy = float(first.get("policy_energy_charge_delta_1", float("nan")))
    concept_col = EXPERT_TO_CONCEPT_COLUMN.get(dominant_expert)
    concept_score = float(first.get(concept_col, float("nan"))) if concept_col else float("nan")

    direction = "向下加速退化" if residual_mean < -1e-5 else ("向上修正 base" if residual_mean > 1e-5 else "接近 0 的微调")
    factor = EXPERT_FACTOR_LABELS_ZH.get(dominant_expert, dominant_expert)
    lines = [
        f"- 主导因素：`{dominant_expert}`（{factor}）；语义概念分数 {_fmt_float(concept_score, 3)}。",
        f"- 残差方向：平均残差 `{_fmt_float(residual_mean, 5)}`，平均绝对残差 `{_fmt_float(residual_abs_mean, 5)}`，表示专家库本段主要在 `{direction}`。",
        f"- 预测终点：anchor SOH `{_fmt_float(anchor_soh, 5)}` -> final predicted SOH `{_fmt_float(pred_soh_last, 5)}`；base 平均 delta `{_fmt_float(base_delta_mean, 5)}`。",
        f"- 工况证据：active abs(I) max `{_fmt_float(current, 3, 'A')}`；temperature mean/max `{_fmt_float(temp_mean, 2, 'degC')}` / `{_fmt_float(temp_max, 2, 'degC')}`；energy-throughput proxy `{_fmt_float(power_proxy, 4)}`。",
        f"- RAG 证据：retrieval confidence `{_fmt_float(retrieval_conf, 3)}`，top-k mean composite distance `{_fmt_float(first.get('topk_mean_composite_distance', float('nan')), 3)}`。",
    ]
    return lines


def _retrieval_top5_rows(retrieval_segments: pd.DataFrame, stage: str) -> List[str]:
    if retrieval_segments is None or retrieval_segments.empty:
        return []
    sub = retrieval_segments[
        (retrieval_segments["stage"].astype(str) == str(stage))
        & (retrieval_segments["series_role"].astype(str) == "reference_history")
        & (retrieval_segments["neighbor_rank"].astype(float) > 0)
    ].copy()
    if sub.empty:
        return []
    top = (
        sub.sort_values(["neighbor_rank", "relative_step"])
        .groupby("neighbor_rank", as_index=False)
        .first()
        .sort_values("neighbor_rank")
        .head(5)
    )
    rows = []
    for _, row in top.iterrows():
        rows.append(
            f"| {int(row['neighbor_rank'])} | {int(row['neighbor_case_id'])} | `{row['cell_uid']}` | "
            f"`{row['source_dataset']}` | `{row['chemistry_family']}` | {_fmt_float(row.get('composite_distance'), 4)} | "
            f"{_fmt_float(row.get('retrieval_alpha'), 4)} | {_fmt_float(row.get('d_soh_state'), 4)} | "
            f"{_fmt_float(row.get('d_qv_shape'), 4)} | {_fmt_float(row.get('d_physics'), 4)} | "
            f"{_fmt_float(row.get('d_metadata'), 4)} | {_fmt_float(row.get('history_shape_rmse'), 5)} |"
        )
    return rows


def _write_summary_md(
    path: Path,
    metrics: Mapping[str, object],
    blocks: pd.DataFrame,
    *,
    segment_frame: pd.DataFrame | None = None,
    retrieval_segments: pd.DataFrame | None = None,
) -> None:
    lines = [
        "# Random Segment SOH Prediction Summary",
        "",
        "## Selected Cell And Query Segments",
        f"- cell_uid: `{metrics['cell_uid']}`",
        f"- source_dataset: `{metrics['source_dataset']}`",
        f"- chemistry_family: `{metrics['chemistry_family']}`",
        f"- split: `{metrics['split']}`",
        f"- window: `N={metrics['history_length']}` history cycles -> `K={metrics['future_length']}` forecast cycles",
        "",
    ]
    if blocks.empty:
        lines.append("_No selected segment records available._")
    else:
        lines.extend(
            [
                "| segment | case_id | anchor_cycle | anchor_soh | raw_cell / condition | current strategy | temperature | voltage / full-cycle status |",
                "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
            ]
        )
        for stage, sub in blocks.groupby("stage"):
            first = sub.sort_values("block_index").iloc[0]
            lines.append(
                f"| {stage} | {int(first['case_id'])} | {int(first['anchor_cycle_idx'])} | {_fmt_float(first['anchor_soh'], 5)} | "
                f"`{first.get('policy_raw_cell_id', 'unknown')}` / `{first.get('policy_condition_label', 'unknown')}` | "
                f"{first.get('policy_current_text', 'unknown')} | {first.get('policy_temperature_text', 'unknown')} | "
                f"Q-V {first.get('policy_voltage_text', 'unknown')}; {first.get('policy_full_charge_discharge_text', 'unknown')} |"
            )
        lines.extend(
            [
                "",
                "Current note: the case bank stores absolute-current statistics, so the charge/discharge current shown above is an active-step `abs(I)` estimate unless the raw adapter exposes a signed split.",
            ]
        )

    lines.extend(["", "## Router Semantic Examples"])
    if blocks.empty:
        lines.append("_No router records available._")
    else:
        for stage, sub in blocks.groupby("stage"):
            first = sub.sort_values("block_index").iloc[0]
            weight_rows, dominant = _weight_change_table(str(stage), blocks, segment_frame)
            lines.append(f"### {stage}")
            lines.append(f"- case_id: `{int(first['case_id'])}`; anchor_cycle_idx: `{int(first['anchor_cycle_idx'])}`")
            lines.extend(_describe_router_stage(str(stage), first, dominant))
            lines.append("")
            lines.append("| expert | first-step weight | mean weight | last-step weight | last-first |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            lines.extend(weight_rows if weight_rows else ["| unavailable | unknown | unknown | unknown | unknown |"])
            evidence = json.loads(first.get("semantic_evidence_json", "{}"))
            if evidence:
                lines.append("")
                lines.append("Semantic evidence:")
                for key, value in evidence.items():
                    lines.append(f"- `{key}`: {value}")
            lines.append("")

    lines.extend(["", "## Retrieval: RAG Top-5 Similarity"])
    if retrieval_segments is None or retrieval_segments.empty:
        lines.append("_No retrieval top-k segment records available._")
    else:
        for stage in sorted(retrieval_segments["stage"].astype(str).unique().tolist()):
            lines.append(f"### {stage}")
            lines.append(
                "| rank | neighbor_case_id | cell_uid | dataset | chemistry | composite_distance | alpha | d_soh_state | d_qv_shape | d_physics | d_metadata | history_shape_RMSE |"
            )
            lines.append("| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
            rows = _retrieval_top5_rows(retrieval_segments, stage)
            lines.extend(rows if rows else ["| - | - | unavailable | unavailable | unavailable | unknown | unknown | unknown | unknown | unknown | unknown | unknown |"])
            lines.append("")

    lines.extend(
        [
            "",
            "## Prediction Error Check",
            "| segment | case_id | anchor_cycle | final_MAE | base_MAE | final_minus_base_MAE | terminal_pred_soh |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in metrics.get("stage_metrics", []):
        final_mae = float(row["mae"])
        base_mae = float(row.get("base_mae", float("nan")))
        stage_blocks = blocks[blocks["stage"].astype(str) == str(row["stage"])] if not blocks.empty else pd.DataFrame()
        terminal = float(stage_blocks.sort_values("block_index").iloc[-1].get("pred_soh_last", float("nan"))) if not stage_blocks.empty else float("nan")
        lines.append(
            f"| {row['stage']} | {int(row['case_id'])} | {int(row['anchor_cycle_idx'])} | "
            f"{final_mae:.6f} | {base_mae:.6f} | {final_mae - base_mae:+.6f} | {_fmt_float(terminal, 5)} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    source_dataset: str = "mit",
    chemistry: str = "LFP",
    split: str = "target_query",
    cell_uid: str | None = None,
    history_length: int | None = None,
    future_length: int | None = None,
    sampling_mode: str = "stage",
    num_segments: int = 3,
    seed: int = 7,
    require_future_at_least_twice_history: bool = False,
) -> Dict[str, object]:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cfg = load_config(str(config_path))
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    model_output_dir = Path(cfg.get("model_output_dir", "output/forecasting"))
    output_root = model_output_dir / "figures" / "random_segments"
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(seed))

    rows = _read_case_rows(case_bank_dir).sort_values("case_id").reset_index(drop=True)
    rows["case_id"] = rows["case_id"].astype(int)
    case_lookback = int(np.load(case_bank_dir / "case_soh_seq.npy", mmap_mode="r").shape[1])
    selected_history_length = int(history_length or case_lookback)
    selected_future_length = int(future_length or max(2 * selected_history_length, int(np.load(case_bank_dir / "case_future_soh.npy", mmap_mode="r").shape[1])))
    if require_future_at_least_twice_history and selected_future_length < 2 * selected_history_length:
        raise ValueError(f"future_length must satisfy K >= 2N; got K={selected_future_length}, N={selected_history_length}.")
    if selected_history_length != case_lookback:
        raise ValueError(f"This trained case bank uses lookback N={case_lookback}; requested history_length={selected_history_length}.")

    selected_cell = _select_cell(
        rows,
        source_dataset=source_dataset,
        chemistry=chemistry,
        split=split,
        future_length=selected_future_length,
        rng=rng,
        cell_uid=cell_uid,
        min_stage_count=1 if str(sampling_mode).lower() == "random" else 2,
    )
    cell_rows = rows[
        (rows["cell_uid"].astype(str) == str(selected_cell))
        & (rows["split"].astype(str) == str(split))
    ].copy()
    if str(sampling_mode).lower() == "random":
        selected_cases = _sample_random_cases(
            cell_rows,
            future_length=selected_future_length,
            rng=rng,
            num_segments=int(num_segments),
        )
    else:
        selected_cases = _sample_stage_cases(cell_rows, future_length=selected_future_length, rng=rng)
    actual_source_dataset = str(cell_rows["source_dataset"].astype(str).iloc[0]) if not cell_rows.empty else str(source_dataset)

    dataset = BatterySOHForecastDataset(
        case_bank_dir=case_bank_dir,
        splits=[split],
        retrieval_cfg=dict(cfg.get("retrieval", {})),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_init = apply_model_init_config_overrides(checkpoint["model_init"], cfg)
    model = BatterySOHForecaster(**model_init)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    critical_missing = [
        key
        for key in missing
        if "horizon_calibrator" not in key and "residual_direction_head" not in key
    ]
    if critical_missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint. missing={critical_missing}, unexpected={list(unexpected)}")
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    model.to(device)
    model.eval()

    truth = _collect_true_lifecycle(
        rows[rows["cell_uid"].astype(str) == str(selected_cell)],
        np.load(case_bank_dir / "case_soh_seq.npy"),
        np.load(case_bank_dir / "case_future_soh.npy"),
    )
    truth_map = dict(zip(truth["cycle_idx"].astype(int).tolist(), truth["true_soh"].astype(float).tolist()))

    segment_frames = []
    block_records: List[Dict[str, object]] = []
    retrieval_segment_frames: List[pd.DataFrame] = []
    for stage, case_row in selected_cases.items():
        case_row = case_row.copy()
        case_row["sampled_stage"] = stage
        case_id = int(case_row["case_id"])
        if case_id not in dataset.query_case_id_to_local:
            raise ValueError(f"case_id={case_id} is missing from the dataset view for split={split!r}.")
        base_batch = move_batch_to_device(_batchify(dataset[dataset.query_case_id_to_local[case_id]]), device)
        retrieval_segment_frames.append(
            _collect_retrieval_topk_segments(
                base_batch=base_batch,
                initial_case=case_row,
                rows=rows,
                history_length=selected_history_length,
                future_length=selected_future_length,
            )
        )
        observed, predicted, blocks = _predict_segment(
            model=model,
            base_batch=base_batch,
            initial_case=case_row,
            truth_map=truth_map,
            future_length=selected_future_length,
            history_length=selected_history_length,
            device=device,
        )
        segment_frames.extend([observed, predicted])
        block_records.extend(blocks)

    segment_frame = pd.concat(segment_frames, ignore_index=True).sort_values(["stage", "cycle_idx"])
    blocks = pd.DataFrame(block_records)
    retrieval_segments = (
        pd.concat(retrieval_segment_frames, ignore_index=True)
        if retrieval_segment_frames
        else pd.DataFrame()
    )
    future = segment_frame[segment_frame["segment_part"] == "future"].copy()
    valid = future[np.isfinite(future["true_soh"].to_numpy(dtype=float))].copy()
    overall = regression_metrics(valid["pred_soh"].to_numpy(dtype=np.float32), valid["true_soh"].to_numpy(dtype=np.float32))
    base_overall = regression_metrics(valid["base_soh"].to_numpy(dtype=np.float32), valid["true_soh"].to_numpy(dtype=np.float32))
    persistence = regression_metrics(valid["persistence_soh"].to_numpy(dtype=np.float32), valid["true_soh"].to_numpy(dtype=np.float32))
    linear = regression_metrics(valid["linear_slope_soh"].to_numpy(dtype=np.float32), valid["true_soh"].to_numpy(dtype=np.float32))

    stage_metrics = []
    for stage, sub in valid.groupby("stage"):
        pred = sub["pred_soh"].to_numpy(dtype=np.float32)
        true = sub["true_soh"].to_numpy(dtype=np.float32)
        pbase = sub["persistence_soh"].to_numpy(dtype=np.float32)
        lbase = sub["linear_slope_soh"].to_numpy(dtype=np.float32)
        bbase = sub["base_soh"].to_numpy(dtype=np.float32)
        case_row = selected_cases[str(stage)]
        stage_metrics.append(
            {
                "stage": str(stage),
                "case_id": int(case_row["case_id"]),
                "anchor_cycle_idx": int(case_row["cycle_idx_end"]),
                "mae": float(regression_metrics(pred, true)["mae"]),
                "rmse": float(regression_metrics(pred, true)["rmse"]),
                "base_mae": float(regression_metrics(bbase, true)["mae"]),
                "base_rmse": float(regression_metrics(bbase, true)["rmse"]),
                "persistence_mae": float(regression_metrics(pbase, true)["mae"]),
                "linear_slope_mae": float(regression_metrics(lbase, true)["mae"]),
            }
        )

    safe_cell = _safe_slug(selected_cell)
    prefix = f"{safe_cell}_{actual_source_dataset.lower()}_{chemistry.lower()}_{sampling_mode.lower()}_N{selected_history_length}_K{selected_future_length}"
    predictions_csv = output_root / f"{prefix}_segment_predictions.csv"
    blocks_csv = output_root / f"{prefix}_expert_blocks.csv"
    retrieval_segments_csv = output_root / f"{prefix}_retrieval_topk_segments.csv"
    metrics_json = output_root / f"{prefix}_metrics.json"
    summary_md = output_root / f"{prefix}_summary.md"
    segment_figure = output_root / f"{prefix}_segments.png"
    expert_figure = output_root / f"{prefix}_expert_weights.png"
    retrieval_figure = output_root / f"{prefix}_retrieval_topk_segments.png"

    metadata = {
        "cell_uid": str(selected_cell),
        "source_dataset": str(actual_source_dataset),
        "chemistry_family": str(chemistry),
        "split": str(split),
        "history_length": int(selected_history_length),
        "future_length": int(selected_future_length),
        "sampling_mode": str(sampling_mode),
        "num_segments": int(len(selected_cases)),
        "seed": int(seed),
        "overall_mae": float(overall["mae"]),
        "overall_rmse": float(overall["rmse"]),
        "overall_mape": float(overall["mape"]),
        "base_overall_mae": float(base_overall["mae"]),
        "base_overall_rmse": float(base_overall["rmse"]),
        "persistence_mae": float(persistence["mae"]),
        "persistence_rmse": float(persistence["rmse"]),
        "linear_slope_mae": float(linear["mae"]),
        "linear_slope_rmse": float(linear["rmse"]),
        "stage_metrics": stage_metrics,
        "note": "Random segment prediction uses N observed SOH values and predicts K future SOH values; non-SOH numerical context is fixed from the initial observed window.",
    }

    segment_frame.to_csv(predictions_csv, index=False)
    blocks.to_csv(blocks_csv, index=False)
    retrieval_segments.to_csv(retrieval_segments_csv, index=False)
    metrics_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot_segments(
        segment_frame,
        segment_figure,
        cell_uid=str(selected_cell),
        source_dataset=str(actual_source_dataset),
        chemistry=str(chemistry),
        history_length=selected_history_length,
        future_length=selected_future_length,
    )
    _plot_expert_weights(blocks, expert_figure, list(model.expert_names), segment_frame=segment_frame)
    _plot_retrieval_topk_segments(
        retrieval_segments,
        retrieval_figure,
        cell_uid=str(selected_cell),
        source_dataset=str(actual_source_dataset),
        chemistry=str(chemistry),
        history_length=selected_history_length,
        future_length=selected_future_length,
    )
    _write_summary_md(
        summary_md,
        metadata,
        blocks,
        segment_frame=segment_frame,
        retrieval_segments=retrieval_segments,
    )
    return {
        "metrics": metadata,
        "predictions_csv": str(predictions_csv),
        "expert_blocks_csv": str(blocks_csv),
        "retrieval_topk_segments_csv": str(retrieval_segments_csv),
        "metrics_json": str(metrics_json),
        "summary_md": str(summary_md),
        "segment_figure": str(segment_figure),
        "expert_figure": str(expert_figure),
        "retrieval_topk_figure": str(retrieval_figure),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Random numerical battery SOH segment prediction experiment.")
    parser.add_argument("--config", default="configs/battery_soh_stratified_fewshot.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--source-dataset", default="mit")
    parser.add_argument("--chemistry", default="LFP")
    parser.add_argument("--split", default="target_query")
    parser.add_argument("--cell-uid", default=None)
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--future-length", type=int, default=None)
    parser.add_argument("--sampling-mode", choices=["stage", "random"], default="stage")
    parser.add_argument("--num-segments", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--require-future-at-least-twice-history",
        action="store_true",
        help="Enforce the earlier long-horizon narrative constraint K >= 2N.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(cfg.get("model_output_dir", "output/forecasting")) / "checkpoints" / "best_adapted.pt"
    result = run_experiment(
        config_path=args.config,
        checkpoint_path=checkpoint,
        source_dataset=args.source_dataset,
        chemistry=args.chemistry,
        split=args.split,
        cell_uid=args.cell_uid,
        history_length=args.history_length,
        future_length=args.future_length,
        sampling_mode=args.sampling_mode,
        num_segments=args.num_segments,
        seed=args.seed,
        require_future_at_least_twice_history=args.require_future_at_least_twice_history,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
