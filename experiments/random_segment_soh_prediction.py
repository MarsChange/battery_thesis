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
    if concepts.get("concept_slow_linear", 0.0) >= 0.55:
        text["slow_linear"] = "历史 SOH 斜率和曲率较稳定，支持缓慢线性退化模式。"
    if concepts.get("concept_accelerating", 0.0) >= 0.55:
        text["accelerating"] = "近期 SOH 趋势存在加速退化证据。"
    if concepts.get("concept_high_polarization", 0.0) >= 0.55:
        text["high_polarization"] = "R(Q) proxy 或运行压力较高，支持高极化模式。"
    if concepts.get("concept_curve_polarization", 0.0) >= 0.55:
        text["curve_polarization_expert"] = "DeltaV(Q)、曲线斜率或 dq/dv 峰值支持曲线极化模式。"
    if concepts.get("concept_high_operation_stress", 0.0) >= 0.55:
        text["operation_stress"] = "电流、温度或协议变化特征显示运行压力较高。"
    if concepts.get("concept_low_retrieval_reliability", 0.0) >= 0.55:
        text["retrieval"] = "RAG 检索置信度偏低，top-k 参考案例应谨慎使用。"
    else:
        text["retrieval"] = "RAG 检索置信度没有被语义层判定为低可靠。"
    return text


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
            expert_weights = _to_numpy(outputs["expert_weights"][0]).reshape(-1)
            concepts = _to_numpy(outputs["semantic_concepts"][0]).reshape(-1)
            retrieval_confidence = float(_to_numpy(outputs["retrieval_confidence"][0]).reshape(-1)[0])
            composite_distance = _to_numpy(batch["retrieval"]["composite_distance"][0]).reshape(-1)
            valid_dist = composite_distance[np.isfinite(composite_distance)]
            step_count = min(horizon, remaining)

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
            for pos, name in enumerate(expert_names):
                block_record[f"expert_weight_{name}"] = float(expert_weights[pos]) if pos < expert_weights.size else float("nan")
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
                prediction_records.append(
                    {
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
                )

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


def _plot_expert_weights(blocks: pd.DataFrame, output_path: Path, expert_names: List[str]) -> None:
    stages = [stage for stage in ["early", "middle", "late"] if stage in set(blocks["stage"].astype(str))]
    if not stages:
        stages = sorted(blocks["stage"].astype(str).unique().tolist())
    colors = {
        "slow_linear": "#2563eb",
        "accelerating": "#dc2626",
        "high_polarization": "#f59e0b",
        "curve_polarization_expert": "#16a34a",
    }
    figure, axes = plt.subplots(len(stages), 1, figsize=(12, 3.6 * len(stages)), dpi=180, sharex=False)
    if len(stages) == 1:
        axes = [axes]
    for axis, stage in zip(axes, stages):
        sub = blocks[blocks["stage"].astype(str) == stage].sort_values("anchor_cycle_idx")
        for name in expert_names:
            col = f"expert_weight_{name}"
            if col in sub.columns:
                axis.plot(
                    sub["anchor_cycle_idx"],
                    sub[col],
                    color=colors.get(name),
                    linewidth=1.8,
                    marker="o",
                    markersize=4.5,
                    label=name,
                )
        axis.set_title(f"{stage} segment expert weights")
        axis.set_xlabel("Segment forecast anchor cycle")
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
        ref_label = (
            f"Top-{rank + 1} | {neighbor_cell} | {neighbor_dataset}/{neighbor_chemistry} "
            f"| d={float(composite_distance[rank]):.3f} | alpha={float(retrieval_alpha[rank]):.2f}"
        )
        history = ref_history[rank, -history_length:]
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


def _write_summary_md(path: Path, metrics: Mapping[str, object], blocks: pd.DataFrame) -> None:
    lines = [
        "# Random Segment SOH Prediction Summary",
        "",
        "## Task Definition",
        f"- Input: `N={metrics['history_length']}` observed historical SOH values from one random segment.",
        f"- Output: next `K={metrics['future_length']}` SOH values.",
        "- Target remains `target_delta_soh = future_soh - anchor_soh`; absolute SOH is reconstructed as `pred_soh = anchor_soh + pred_delta_soh`.",
        "- This is a random segment prediction experiment, not complete lifecycle prediction.",
        "- Retrieval top-k segment figures align query/reference histories and futures by anchor SOH, so the plot compares trajectory shape rather than absolute cell identity.",
        "",
        "## Selected Cell",
        f"- cell_uid: `{metrics['cell_uid']}`",
        f"- source_dataset: `{metrics['source_dataset']}`",
        f"- chemistry_family: `{metrics['chemistry_family']}`",
        f"- split: `{metrics['split']}`",
        "",
        "## Metrics",
        f"- Overall MAE: {float(metrics['overall_mae']):.6f}",
        f"- Overall RMSE: {float(metrics['overall_rmse']):.6f}",
        f"- Base-only MAE: {float(metrics.get('base_overall_mae', float('nan'))):.6f}",
        f"- Base-only RMSE: {float(metrics.get('base_overall_rmse', float('nan'))):.6f}",
        f"- Persistence MAE: {float(metrics['persistence_mae']):.6f}",
        f"- Initial-slope MAE: {float(metrics['linear_slope_mae']):.6f}",
        "",
        "## Stage Metrics",
        "| stage | case_id | anchor_cycle | final_MAE | base_MAE | RMSE | persistence_MAE | linear_slope_MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics.get("stage_metrics", []):
        lines.append(
            f"| {row['stage']} | {int(row['case_id'])} | {int(row['anchor_cycle_idx'])} | "
            f"{float(row['mae']):.6f} | {float(row.get('base_mae', float('nan'))):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['persistence_mae']):.6f} | {float(row['linear_slope_mae']):.6f} |"
        )
    lines.extend(["", "## Router Semantic Examples"])
    if blocks.empty:
        lines.append("_No router records available._")
    else:
        for stage, sub in blocks.groupby("stage"):
            first = sub.sort_values("block_index").iloc[0]
            evidence = json.loads(first.get("semantic_evidence_json", "{}"))
            weights = {
                col.replace("expert_weight_", ""): float(first[col])
                for col in blocks.columns
                if col.startswith("expert_weight_")
            }
            lines.append(f"### {stage}")
            lines.append(f"- case_id: {int(first['case_id'])}")
            lines.append(f"- anchor_cycle_idx: {int(first['anchor_cycle_idx'])}")
            lines.append(f"- expert_weights: `{json.dumps(weights, ensure_ascii=False)}`")
            for key, value in evidence.items():
                lines.append(f"- {key}: {value}")
            lines.append("")
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
    model.load_state_dict(checkpoint["model_state"])
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
    _plot_expert_weights(blocks, expert_figure, list(model.expert_names))
    _plot_retrieval_topk_segments(
        retrieval_segments,
        retrieval_figure,
        cell_uid=str(selected_cell),
        source_dataset=str(actual_source_dataset),
        chemistry=str(chemistry),
        history_length=selected_history_length,
        future_length=selected_future_length,
    )
    _write_summary_md(summary_md, metadata, blocks)
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
