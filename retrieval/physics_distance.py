from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def normalized_l2(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray | None = None,
    scale: np.ndarray | float | None = None,
) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    diff = arr_a - arr_b
    if mask is not None:
        weight = np.asarray(mask, dtype=np.float32)
        if weight.shape != diff.shape:
            weight = np.broadcast_to(weight, diff.shape)
        diff = diff * weight
        denom = float(weight.sum())
        if denom <= 0:
            return 1.0
    else:
        denom = float(diff.size) if diff.size else 1.0
    if scale is not None:
        scale_arr = np.asarray(scale, dtype=np.float32)
        diff = diff / np.maximum(np.abs(scale_arr), 1e-6)
    return float(np.sqrt(np.sum(diff * diff) / max(denom, 1.0)))


def qv_map_distance(
    query_qv: np.ndarray,
    ref_qv: np.ndarray,
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
    channel_weights: Dict[str, float] | Iterable[float],
) -> float:
    q = np.asarray(query_qv, dtype=np.float32)
    r = np.asarray(ref_qv, dtype=np.float32)
    q_mask = np.asarray(query_mask, dtype=np.float32)
    r_mask = np.asarray(ref_mask, dtype=np.float32)
    common = (q_mask > 0) & (r_mask > 0)
    if not common.any():
        return 1.0
    if isinstance(channel_weights, dict):
        weights = np.asarray(
            [
                float(channel_weights.get("Vc", 1.0)),
                float(channel_weights.get("Vd", 1.0)),
                float(channel_weights.get("Ic", 1.0)),
                float(channel_weights.get("Id", 1.0)),
                float(channel_weights.get("DeltaV", 1.0)),
                float(channel_weights.get("R", 1.0)),
            ],
            dtype=np.float32,
        )
    else:
        weights = np.asarray(list(channel_weights), dtype=np.float32)
    weights = weights[: q.shape[0]]
    weighted = []
    total = 0.0
    for idx in range(q.shape[0]):
        if not common[idx]:
            continue
        dist = normalized_l2(q[idx], r[idx])
        weighted.append(weights[idx] * dist)
        total += float(weights[idx])
    if total <= 0:
        return 1.0
    return float(sum(weighted) / total)


def physics_feature_distance(
    query_f: np.ndarray,
    ref_f: np.ndarray,
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> float:
    q = np.asarray(query_f, dtype=np.float32)
    r = np.asarray(ref_f, dtype=np.float32)
    mask = (np.asarray(query_mask, dtype=np.float32) > 0) & (np.asarray(ref_mask, dtype=np.float32) > 0)
    if not mask.any():
        return 1.0
    scale = np.maximum(np.abs(q), np.abs(r))
    return normalized_l2(q, r, mask=mask.astype(np.float32), scale=np.maximum(scale, 1e-4))


def operation_distance(query_op_summary: np.ndarray, ref_op_summary: np.ndarray) -> float:
    q = np.asarray(query_op_summary, dtype=np.float32)
    r = np.asarray(ref_op_summary, dtype=np.float32)
    scale = np.maximum(np.abs(q), np.abs(r))
    return normalized_l2(q, r, scale=np.maximum(scale, 1e-4))


def metadata_distance(query_meta: Dict[str, object], ref_meta: Dict[str, object], weights: Dict[str, float]) -> float:
    score = 0.0
    total = 0.0
    for field, weight in weights.items():
        total += float(weight)
        qv = query_meta.get(field)
        rv = ref_meta.get(field)
        score += float(weight) * (0.0 if qv == rv and qv is not None else 1.0)
    if total <= 0:
        return 0.0
    return float(score / total)


def soh_state_distance(query_state: Dict[str, float], ref_state: Dict[str, float]) -> float:
    q = np.asarray(
        [
            float(query_state.get("anchor_soh", 0.0)),
            float(query_state.get("recent_soh_slope", 0.0)),
            float(query_state.get("recent_soh_curvature", 0.0)),
            float(query_state.get("throughput_recent", 0.0)),
        ],
        dtype=np.float32,
    )
    r = np.asarray(
        [
            float(ref_state.get("anchor_soh", 0.0)),
            float(ref_state.get("recent_soh_slope", 0.0)),
            float(ref_state.get("recent_soh_curvature", 0.0)),
            float(ref_state.get("throughput_recent", 0.0)),
        ],
        dtype=np.float32,
    )
    scale = np.asarray([0.1, 0.02, 0.01, max(abs(q[-1]), abs(r[-1]), 1.0)], dtype=np.float32)
    return normalized_l2(q, r, scale=scale)


def degradation_stage_distance(query_stage: str, ref_stage: str) -> float:
    return 0.0 if str(query_stage) == str(ref_stage) else 1.0


def compute_retrieval_confidence(component_distances: np.ndarray) -> float:
    distances = np.asarray(component_distances, dtype=np.float32)
    if distances.size == 0:
        return 0.0
    mean_distance = float(np.nanmean(distances))
    spread = float(np.nanstd(distances))
    confidence = np.exp(-(mean_distance + 0.5 * spread))
    return float(np.clip(confidence, 0.0, 1.0))
