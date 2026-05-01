"""Named numerical distances for battery RAG retrieval.

The retrieval method uses only interpretable battery-specific components:
`d_soh_state`, `d_qv_shape`, `d_physics`, and `d_metadata`.
`d_operation` is retained as a compatibility output but is disabled by default;
raw current, temperature and normalized-capacity-change matching now belongs to
`d_metadata`.
Each distance is lower-is-more-similar and can be enabled or disabled through
`configs/retrieval_features.yaml`.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


CORE_COMPONENT_NAMES = ["d_soh_state", "d_qv_shape", "d_physics", "d_operation", "d_metadata"]
OPTIONAL_COMPONENT_NAMES: list[str] = []
ALL_COMPONENT_NAMES = list(CORE_COMPONENT_NAMES)
CORE_DISTANCE_NAMES = list(CORE_COMPONENT_NAMES)

QV_CHANNEL_TO_INDEX = {"Vc": 0, "Vd": 1, "Ic": 2, "Id": 3, "DeltaV": 4, "R": 5}
PHYSICS_PROXY_NAMES = [
    "delta_v_mean",
    "delta_v_std",
    "delta_v_q95",
    "r_mean",
    "r_std",
    "r_q95",
    "vc_curve_slope_mean",
    "vd_curve_slope_mean",
]


def _as_float_array(value: object) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.astype(np.float32, copy=False)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _get_value(record: Mapping[str, object] | object, key: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(key, default)
    if hasattr(record, key):
        return getattr(record, key)
    try:
        return record[key]  # type: ignore[index]
    except Exception:
        return default


def _component_enabled(config: Mapping[str, object], name: str) -> bool:
    if "distance_components" in config:
        return bool(dict(config.get("distance_components", {}).get(name, {}) or {}).get("enabled", False))
    return bool(config.get(name, False))


def _component_weight(config: Mapping[str, object], name: str) -> float:
    if "distance_components" in config:
        return max(_safe_float(dict(config.get("distance_components", {}).get(name, {}) or {}).get("weight", 0.0)), 0.0)
    return max(_safe_float(dict(config.get("weights", {}) or {}).get(name, 0.0)), 0.0)


def _component_cfg(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    if "distance_components" in config:
        return dict(config.get("distance_components", {}).get(name, {}) or {})
    return dict(config or {})


def normalized_l2(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, scale: np.ndarray | float | None = None) -> float:
    """计算带 mask 和尺度归一化的 L2 距离。

    中文含义：比较两个连续向量在共同可用维度上的差异。
    输入：`a`、`b` 为同 shape 向量；`mask` 为可选可用维度；`scale` 为量纲归一化尺度。
    输出：float，越小表示越相似。
    """

    arr_a = _as_float_array(a)
    arr_b = _as_float_array(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"normalized_l2 shape mismatch: {arr_a.shape} vs {arr_b.shape}")
    diff = arr_a - arr_b
    if scale is not None:
        scale_arr = np.asarray(scale, dtype=np.float32)
        if scale_arr.ndim == 0:
            scale_arr = np.full_like(diff, max(float(scale_arr), 1e-6))
        diff = diff / np.maximum(np.abs(scale_arr), 1e-6)
    if mask is not None:
        weight = np.asarray(mask, dtype=np.float32)
        if weight.shape != diff.shape:
            weight = np.broadcast_to(weight, diff.shape)
        denom = float(weight.sum())
        if denom <= 0:
            return 1.0
        diff = diff * weight
    else:
        denom = float(diff.size) if diff.size else 1.0
    return float(np.sqrt(np.sum(diff * diff) / max(denom, 1.0)))


def degradation_stage_distance(query_stage: str, ref_stage: str) -> float:
    """计算退化阶段离散距离，0 表示相同，1 表示不同。"""

    return 0.0 if str(query_stage) == str(ref_stage) else 1.0


def compute_soh_state_distance(query: Mapping[str, object] | object, reference: Mapping[str, object] | object, config: Mapping[str, object]) -> float:
    """计算 SOH 状态距离 d_soh_state。

    中文含义：衡量 query 和 reference 是否处于相似健康状态和退化趋势。
    输入：anchor_soh、recent_soh_slope、recent_soh_curvature、degradation_stage。
    输出：float，越小表示当前 SOH、趋势和阶段越相似。
    """

    q = np.asarray(
        [
            _safe_float(_get_value(query, "anchor_soh", 0.0)),
            _safe_float(_get_value(query, "recent_soh_slope", 0.0)),
            _safe_float(_get_value(query, "recent_soh_curvature", 0.0)),
        ],
        dtype=np.float32,
    )
    r = np.asarray(
        [
            _safe_float(_get_value(reference, "anchor_soh", 0.0)),
            _safe_float(_get_value(reference, "recent_soh_slope", 0.0)),
            _safe_float(_get_value(reference, "recent_soh_curvature", 0.0)),
        ],
        dtype=np.float32,
    )
    state_dist = normalized_l2(q, r, scale=np.asarray([0.1, 0.01, 0.005], dtype=np.float32))
    stage_dist = degradation_stage_distance(str(_get_value(query, "degradation_stage", "unknown")), str(_get_value(reference, "degradation_stage", "unknown")))
    return float(0.85 * state_dist + 0.15 * stage_dist)


def _qv_summary_vector(record: Mapping[str, object] | object) -> tuple[np.ndarray, np.ndarray]:
    stats = dict(_get_value(record, "qv_summary_stats", {}) or {})
    values = np.asarray([_safe_float(stats.get(name, 0.0)) for name in PHYSICS_PROXY_NAMES], dtype=np.float32)
    mask = np.asarray([1.0 if name in stats else 0.0 for name in PHYSICS_PROXY_NAMES], dtype=np.float32)
    return values, mask


def compute_qv_shape_distance(query: Mapping[str, object] | object, reference: Mapping[str, object] | object, config: Mapping[str, object]) -> float:
    """计算容量-电压曲线形状距离 d_qv_shape。

    中文含义：比较 Q-indexed Vd(Q)、DeltaV(Q)、R(Q) 和 Q-V 统计量的形状差异。
    输入：qv_map、qv_mask、qv_summary_stats。
    输出：float，越小表示容量-电压曲线越相似。
    """

    query_qv = _as_float_array(_get_value(query, "qv_map", _get_value(query, "qv_maps", np.zeros((6, 0), dtype=np.float32))))
    ref_qv = _as_float_array(_get_value(reference, "qv_map", _get_value(reference, "qv_maps", np.zeros((6, 0), dtype=np.float32))))
    query_mask = _as_float_array(_get_value(query, "qv_mask", _get_value(query, "qv_masks", np.zeros(query_qv.shape[0], dtype=np.float32))))
    ref_mask = _as_float_array(_get_value(reference, "qv_mask", _get_value(reference, "qv_masks", np.zeros(ref_qv.shape[0], dtype=np.float32))))

    qv_cfg = dict(config.get("qv_shape", {}) or config or {})
    weights = dict(qv_cfg.get("channel_weights", {}) or {"Vd": 0.25, "DeltaV": 0.40, "R": 0.35})
    channel_names = ["Vd", "DeltaV", "R"]
    if bool(qv_cfg.get("use_vc", False)):
        channel_names.append("Vc")
    if bool(qv_cfg.get("use_ic", False)):
        channel_names.append("Ic")
    if bool(qv_cfg.get("use_id", False)):
        channel_names.append("Id")

    terms = []
    total_weight = 0.0
    for channel_name in channel_names:
        channel_idx = QV_CHANNEL_TO_INDEX[channel_name]
        if channel_idx >= query_qv.shape[0] or channel_idx >= ref_qv.shape[0]:
            continue
        if channel_idx >= query_mask.shape[0] or channel_idx >= ref_mask.shape[0]:
            continue
        if query_mask[channel_idx] <= 0 or ref_mask[channel_idx] <= 0:
            continue
        weight = _safe_float(weights.get(channel_name, 1.0), 1.0)
        terms.append(weight * normalized_l2(query_qv[channel_idx], ref_qv[channel_idx]))
        total_weight += weight

    if bool(qv_cfg.get("use_summary_stats", True)):
        q_summary, q_summary_mask = _qv_summary_vector(query)
        r_summary, r_summary_mask = _qv_summary_vector(reference)
        common = (q_summary_mask > 0) & (r_summary_mask > 0)
        if common.any():
            scale = np.maximum(np.maximum(np.abs(q_summary), np.abs(r_summary)), 1e-4)
            terms.append(normalized_l2(q_summary, r_summary, mask=common.astype(np.float32), scale=scale))
            total_weight += 1.0

    if total_weight <= 0:
        return 1.0
    return float(sum(terms) / total_weight)


def compute_physics_distance(query: Mapping[str, object] | object, reference: Mapping[str, object] | object, config: Mapping[str, object]) -> float:
    """计算物理 proxy 距离 d_physics。

    中文含义：比较 query 和 reference 的 ΔV(Q) 与 R(Q) 低维物理 proxy。
    输入：physics_features 或 qv_summary_stats 中的 delta_v_mean/std/q95、r_mean/std/q95 和曲线斜率。
    输出：float，越小表示极化/内阻 proxy 越相似。
    """

    def vector(record: Mapping[str, object] | object) -> tuple[np.ndarray, np.ndarray]:
        stats = dict(_get_value(record, "qv_summary_stats", {}) or {})
        if stats:
            return _qv_summary_vector(record)
        feat = _as_float_array(_get_value(record, "physics_features", np.zeros(len(PHYSICS_PROXY_NAMES), dtype=np.float32)))
        mask = _as_float_array(_get_value(record, "physics_feature_mask", _get_value(record, "physics_feature_masks", np.ones_like(feat))))
        return feat[: len(PHYSICS_PROXY_NAMES)], mask[: len(PHYSICS_PROXY_NAMES)]

    q_feat, q_mask = vector(query)
    r_feat, r_mask = vector(reference)
    common = (q_mask > 0) & (r_mask > 0)
    if not common.any():
        return 1.0
    scale = np.maximum(np.maximum(np.abs(q_feat), np.abs(r_feat)), 1e-4)
    return normalized_l2(q_feat, r_feat, mask=common.astype(np.float32), scale=scale)


def compute_operation_distance(query: Mapping[str, object] | object, reference: Mapping[str, object] | object, config: Mapping[str, object]) -> float:
    """计算历史工况距离 d_operation。

    中文含义：兼容旧输出的工况距离。当前主配置默认关闭该分量，
    充放电电流、温度和归一化容量变化的原始滑窗数值已经迁移到
    `d_metadata` 中计算。
    输入：保留参数仅用于兼容旧调用。
    输出：NaN；默认不参与 composite_distance。
    """

    return float("nan")


def compute_metadata_distance(query: Mapping[str, object] | object, reference: Mapping[str, object] | object, config: Mapping[str, object]) -> float:
    """计算 metadata 距离 d_metadata。

    中文含义：类别 metadata soft penalty + 原始滑窗数值匹配。
    输入：
    - 类别项：chemistry_family、domain_label、voltage_window_bucket；
    - 数值项：charge_current_seq、discharge_current_seq、temperature_seq、
      normalized_capacity_delta_seq。
    输出：float，越小表示背景信息越可比。
    """

    penalties = dict(config.get("metadata_penalties", {}) or config.get("categorical_penalties", {}) or {})
    comparisons = [
        ("chemistry_family", "chemistry_family_mismatch", 1.0),
        ("domain_label", "domain_label_mismatch", 0.5),
        ("voltage_window_bucket", "voltage_window_mismatch", 0.5),
    ]
    score = 0.0
    total = 0.0
    for field, penalty_name, default_penalty in comparisons:
        penalty = _safe_float(penalties.get(penalty_name, default_penalty), default_penalty)
        q_value = str(_get_value(query, field, "unknown"))
        r_value = str(_get_value(reference, field, "unknown"))
        total += penalty
        score += penalty * (0.0 if q_value == r_value and q_value != "unknown" else 1.0)
    categorical_distance = float(score / max(total, 1e-6))

    numeric_weights = dict(config.get("metadata_numeric_weights", {}) or {})
    if not numeric_weights:
        return categorical_distance

    numeric_distance_sum = 0.0
    numeric_weight_sum = 0.0
    for field, weight_value in numeric_weights.items():
        weight = _safe_float(weight_value, 0.0)
        if weight <= 0:
            continue
        q_arr = _as_float_array(_get_value(query, field, None)).reshape(-1)
        r_arr = _as_float_array(_get_value(reference, field, None)).reshape(-1)
        length = min(q_arr.size, r_arr.size)
        if length <= 0:
            continue
        q_arr = q_arr[-length:]
        r_arr = r_arr[-length:]
        mask = np.isfinite(q_arr) & np.isfinite(r_arr)
        if not mask.any():
            continue
        scale = np.maximum(np.maximum(np.abs(q_arr), np.abs(r_arr)), 1e-4)
        numeric_distance_sum += weight * normalized_l2(q_arr, r_arr, mask=mask.astype(np.float32), scale=scale)
        numeric_weight_sum += weight

    if numeric_weight_sum <= 0:
        return categorical_distance

    numeric_distance = float(numeric_distance_sum / numeric_weight_sum)
    mix = dict(config.get("metadata_distance_weights", {}) or {})
    categorical_weight = max(_safe_float(mix.get("categorical", 0.4), 0.4), 0.0)
    raw_numeric_weight = max(_safe_float(mix.get("raw_numeric", 0.6), 0.6), 0.0)
    denom = max(categorical_weight + raw_numeric_weight, 1e-6)
    return float((categorical_weight * categorical_distance + raw_numeric_weight * numeric_distance) / denom)


def compute_composite_distance(component_distances: Mapping[str, float], rag_config: Mapping[str, object]) -> float:
    """计算 composite_distance。

    中文含义：只使用 YAML 中布尔开关为 true 的核心距离分量，加权求和。
    输入：命名距离 dict 和 retrieval feature config。
    输出：float，越小表示 reference 越适合作为历史参考案例。
    """

    enabled = {name: _component_weight(rag_config, name) for name in CORE_COMPONENT_NAMES if _component_enabled(rag_config, name)}
    enabled = {name: weight for name, weight in enabled.items() if weight > 0 and np.isfinite(_safe_float(component_distances.get(name), np.nan))}
    if not enabled:
        return 0.0
    weight_sum = float(sum(enabled.values()))
    return float(sum((weight / weight_sum) * _safe_float(component_distances.get(name), 0.0) for name, weight in enabled.items()))


def compute_retrieval_confidence(topk_component_distances: Mapping[str, object] | Sequence[Mapping[str, object]] | np.ndarray, rag_config: Mapping[str, object] | None = None) -> float:
    """计算 retrieval_confidence。

    中文含义：衡量 top-k 检索结果是否整体可靠。
    输入：top-k composite_distance、特征可用率、chemistry match rate 等。
    输出：[0, 1]，越大表示检索越可靠。
    """

    if isinstance(topk_component_distances, Mapping):
        payload = dict(topk_component_distances)
    elif isinstance(topk_component_distances, np.ndarray):
        payload = {"composite_distance": np.asarray(topk_component_distances, dtype=np.float32)}
    else:
        rows = list(topk_component_distances)
        payload = {
            "composite_distance": np.asarray([_safe_float(row.get("composite_distance", 0.0)) for row in rows], dtype=np.float32),
            "chemistry_match_rate": float(np.mean([_safe_float(row.get("chemistry_match", 0.0)) for row in rows])) if rows else 0.0,
            "domain_match_rate": float(np.mean([_safe_float(row.get("domain_match", 0.0)) for row in rows])) if rows else 0.0,
        }
    composite = _as_float_array(payload.get("composite_distance", np.zeros(0, dtype=np.float32)))
    composite = composite[np.isfinite(composite)]
    if composite.size == 0:
        return 0.0
    scores = [
        float(np.exp(-float(np.mean(composite)))),
        float(np.exp(-float(np.std(composite)))),
        float(np.exp(-float(np.min(composite)))),
        float(np.clip(_safe_float(payload.get("feature_availability_ratio", 1.0), 1.0), 0.0, 1.0)),
        float(np.clip(_safe_float(payload.get("chemistry_match_rate", 0.0), 0.0), 0.0, 1.0)),
    ]
    return float(np.clip(float(np.mean(scores)), 0.0, 1.0))


def qv_map_distance(query_qv: np.ndarray, ref_qv: np.ndarray, query_mask: np.ndarray, ref_mask: np.ndarray, channel_weights: Dict[str, float] | Iterable[float]) -> float:
    """兼容旧接口的 Q-V 曲线距离包装器。"""

    return compute_qv_shape_distance(
        {"qv_map": query_qv, "qv_mask": query_mask},
        {"qv_map": ref_qv, "qv_mask": ref_mask},
        {"qv_shape": {"channel_weights": dict(channel_weights) if isinstance(channel_weights, Mapping) else {}}},
    )


def physics_feature_distance(query_f: np.ndarray, ref_f: np.ndarray, query_mask: np.ndarray, ref_mask: np.ndarray) -> float:
    """兼容旧接口的 physics distance 包装器。"""

    return compute_physics_distance(
        {"physics_features": query_f, "physics_feature_mask": query_mask},
        {"physics_features": ref_f, "physics_feature_mask": ref_mask},
        {},
    )


def operation_distance(query_op_summary: np.ndarray, ref_op_summary: np.ndarray) -> float:
    """兼容旧接口的 operation distance 包装器。"""

    q = _as_float_array(query_op_summary)
    r = _as_float_array(ref_op_summary)
    return normalized_l2(q, r, scale=np.maximum(np.maximum(np.abs(q), np.abs(r)), 1e-4))


def metadata_distance(query_meta: Dict[str, object], ref_meta: Dict[str, object], weights: Dict[str, float]) -> float:
    """兼容旧接口的 metadata distance 包装器。"""

    return compute_metadata_distance(query_meta, ref_meta, {"metadata_penalties": weights})


def soh_state_distance(query_state: Dict[str, float], ref_state: Dict[str, float]) -> float:
    """兼容旧接口的 SOH state distance 包装器。"""

    return compute_soh_state_distance(query_state, ref_state, {})
