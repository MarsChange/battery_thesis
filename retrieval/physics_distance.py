"""retrieval.physics_distance

实现 RAG 检索阶段使用的命名距离分量。
本文件只处理具有明确语义的核心距离：
- d_soh_state
- d_qv_shape
- d_physics
- d_operation
- d_metadata
- d_tsfm

所有距离遵循同一方向：
数值越小表示 query 和 reference 越相似。
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


CORE_DISTANCE_NAMES = [
    "d_soh_state",
    "d_qv_shape",
    "d_physics",
    "d_operation",
    "d_metadata",
    "d_tsfm",
]

QV_CHANNEL_TO_INDEX = {
    "Vc": 0,
    "Vd": 1,
    "Ic": 2,
    "Id": 3,
    "DeltaV": 4,
    "R": 5,
}


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


def normalized_l2(
    a: np.ndarray,
    b: np.ndarray,
    mask: np.ndarray | None = None,
    scale: np.ndarray | float | None = None,
) -> float:
    """计算带 mask 和尺度归一化的 L2 距离。

    中文含义：
    - 这是所有连续特征距离的基础工具函数。
    - 用于比较两个连续向量在共同可用维度上的差异。

    输入：
    - a, b: 连续数值向量。
    - mask: 可选的 0/1 mask；1 表示该维度可参与距离计算。
    - scale: 可选的归一化尺度；用于让不同量纲的维度具有可比性。

    输出：
    - float，越小表示两个向量越相似。
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
    """计算退化阶段的离散距离。

    中文含义：
    - 用于判断两个案例是否处于相同退化阶段。
    - 该量通常作为 SOH 状态距离的一部分或单独诊断项。

    输入：
    - query_stage: query 的退化阶段标签。
    - ref_stage: reference 的退化阶段标签。

    输出：
    - 0.0 表示阶段相同。
    - 1.0 表示阶段不同。
    """

    return 0.0 if str(query_stage) == str(ref_stage) else 1.0


def compute_soh_state_distance(
    query: Mapping[str, object] | object,
    reference: Mapping[str, object] | object,
    config: Mapping[str, object],
) -> float:
    """计算 SOH 状态距离 d_soh_state。

    中文含义：
    - 用于衡量 query 和 reference 是否处于相似健康状态和退化趋势。
    - 该距离综合考虑当前 SOH、近期衰减速度、退化曲率和退化阶段。

    输入：
    - query: 包含 `anchor_soh`、`recent_soh_slope`、`recent_soh_curvature`、`degradation_stage` 的 query 特征字典。
    - reference: 包含同名字段的 reference 特征字典。
    - config: `configs/rag_retrieval_features.yaml` 中 `distance_components.d_soh_state` 对应的配置。

    输出：
    - float，越小越相似。
    """

    features_cfg = dict(_get_value(config, "features", {}) or {})
    values_q = []
    values_r = []
    scales = []
    weights = []

    for feature_name, default_scale in [
        ("anchor_soh", 0.1),
        ("recent_soh_slope", 0.01),
        ("recent_soh_curvature", 0.005),
        ("normalized_cycle_index", 0.25),
    ]:
        feature_cfg = dict(features_cfg.get(feature_name, {}) or {})
        if not bool(feature_cfg.get("enabled", False)):
            continue
        values_q.append(_safe_float(_get_value(query, feature_name, 0.0)))
        values_r.append(_safe_float(_get_value(reference, feature_name, 0.0)))
        scales.append(default_scale)
        weights.append(1.0)

    continuous_distance = 0.0
    if values_q:
        continuous_distance = normalized_l2(
            np.asarray(values_q, dtype=np.float32),
            np.asarray(values_r, dtype=np.float32),
            mask=np.asarray(weights, dtype=np.float32),
            scale=np.asarray(scales, dtype=np.float32),
        )

    stage_distance = 0.0
    stage_cfg = dict(features_cfg.get("degradation_stage", {}) or {})
    if bool(stage_cfg.get("enabled", False)):
        stage_distance = degradation_stage_distance(
            str(_get_value(query, "degradation_stage", "unknown")),
            str(_get_value(reference, "degradation_stage", "unknown")),
        )

    enabled_count = int(bool(values_q)) + int(bool(stage_cfg.get("enabled", False)))
    if enabled_count <= 0:
        return 0.0
    return float((continuous_distance + stage_distance) / enabled_count)


def _select_qv_channels(config: Mapping[str, object]) -> list[tuple[str, int, float]]:
    qv_cfg = dict(config or {})
    features_cfg = dict(qv_cfg.get("features", {}) or {})
    channel_weights = dict(qv_cfg.get("channel_weights", {}) or {})
    selected = []
    for yaml_name, channel_name in [
        ("vc_curve_q", "Vc"),
        ("vd_curve_q", "Vd"),
        ("ic_curve_q", "Ic"),
        ("id_curve_q", "Id"),
        ("delta_v_curve_q", "DeltaV"),
        ("r_curve_q", "R"),
    ]:
        if bool(dict(features_cfg.get(yaml_name, {}) or {}).get("enabled", False)):
            selected.append((yaml_name, QV_CHANNEL_TO_INDEX[channel_name], _safe_float(channel_weights.get(channel_name), 1.0)))
    return selected


def _qv_summary_vector(record: Mapping[str, object] | object) -> tuple[np.ndarray, np.ndarray]:
    stats = dict(_get_value(record, "qv_summary_stats", {}) or {})
    names = [
        "delta_v_mean",
        "delta_v_std",
        "delta_v_q95",
        "r_mean",
        "r_std",
        "r_q95",
        "vc_curve_slope_mean",
        "vd_curve_slope_mean",
    ]
    values = np.asarray([_safe_float(stats.get(name, 0.0)) for name in names], dtype=np.float32)
    mask = np.asarray([1.0 if name in stats else 0.0 for name in names], dtype=np.float32)
    return values, mask


def compute_qv_shape_distance(
    query: Mapping[str, object] | object,
    reference: Mapping[str, object] | object,
    config: Mapping[str, object],
) -> float:
    """计算容量-电压曲线形状距离 d_qv_shape。

    中文含义：
    - 用于衡量 query 和 reference 的 Q-indexed Vd(Q)、DeltaV(Q)、R(Q) 或 Q-V summary stats 是否相似。
    - 该距离重点反映放电平台、极化和阻抗 proxy 的曲线形态是否接近。

    输入：
    - query/reference: 至少包含 `qv_map`、`qv_mask`；可选包含 `qv_summary_stats`。
    - config: `configs/rag_retrieval_features.yaml` 中 `distance_components.d_qv_shape` 对应的配置。

    输出：
    - float，越小越相似。
    """

    query_qv = _as_float_array(_get_value(query, "qv_map", _get_value(query, "qv_maps", np.zeros((6, 0), dtype=np.float32))))
    ref_qv = _as_float_array(_get_value(reference, "qv_map", _get_value(reference, "qv_maps", np.zeros((6, 0), dtype=np.float32))))
    query_mask = _as_float_array(_get_value(query, "qv_mask", _get_value(query, "qv_masks", np.zeros(query_qv.shape[0], dtype=np.float32))))
    ref_mask = _as_float_array(_get_value(reference, "qv_mask", _get_value(reference, "qv_masks", np.zeros(ref_qv.shape[0], dtype=np.float32))))

    channel_terms = []
    total_weight = 0.0
    for _, channel_idx, weight in _select_qv_channels(config):
        if channel_idx >= query_qv.shape[0] or channel_idx >= ref_qv.shape[0]:
            continue
        if channel_idx >= query_mask.shape[0] or channel_idx >= ref_mask.shape[0]:
            continue
        if query_mask[channel_idx] <= 0 or ref_mask[channel_idx] <= 0:
            continue
        channel_terms.append(weight * normalized_l2(query_qv[channel_idx], ref_qv[channel_idx]))
        total_weight += weight

    features_cfg = dict(_get_value(config, "features", {}) or {})
    if bool(dict(features_cfg.get("qv_summary_stats", {}) or {}).get("enabled", False)):
        q_summary, q_summary_mask = _qv_summary_vector(query)
        r_summary, r_summary_mask = _qv_summary_vector(reference)
        common_mask = (q_summary_mask > 0) & (r_summary_mask > 0)
        if common_mask.any():
            scale = np.maximum(np.maximum(np.abs(q_summary), np.abs(r_summary)), 1e-4)
            channel_terms.append(normalized_l2(q_summary, r_summary, mask=common_mask.astype(np.float32), scale=scale))
            total_weight += 1.0

    if total_weight <= 0:
        return 1.0
    return float(sum(channel_terms) / total_weight)


def compute_physics_distance(
    query: Mapping[str, object] | object,
    reference: Mapping[str, object] | object,
    config: Mapping[str, object],
) -> float:
    """计算物理启发特征距离 d_physics。

    中文含义：
    - 用于比较 partial charging 和 relaxation 中提取出的 12 维退化 proxy。
    - 该距离反映退化状态在低维物理启发空间中的接近程度。

    输入：
    - query/reference: 包含 `physics_features` 和 `physics_feature_mask` 的字典。
    - config: `configs/rag_retrieval_features.yaml` 中 `distance_components.d_physics` 对应的配置。

    输出：
    - float，越小越相似。
    """

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
    enabled = {
        name: bool(dict(dict(config or {}).get("features", {}) or {}).get(name, {}).get("enabled", False))
        for name in feature_names
    }

    q_feat = _as_float_array(_get_value(query, "physics_features", np.zeros(len(feature_names), dtype=np.float32)))
    r_feat = _as_float_array(_get_value(reference, "physics_features", np.zeros(len(feature_names), dtype=np.float32)))
    q_mask = _as_float_array(_get_value(query, "physics_feature_mask", _get_value(query, "physics_feature_masks", np.ones_like(q_feat))))
    r_mask = _as_float_array(_get_value(reference, "physics_feature_mask", _get_value(reference, "physics_feature_masks", np.ones_like(r_feat))))

    feature_mask = np.asarray([1.0 if enabled[name] else 0.0 for name in feature_names], dtype=np.float32)
    common = (q_mask[: len(feature_names)] > 0) & (r_mask[: len(feature_names)] > 0) & (feature_mask > 0)
    if not common.any():
        return 1.0
    scale = np.maximum(np.maximum(np.abs(q_feat[: len(feature_names)]), np.abs(r_feat[: len(feature_names)])), 1e-4)
    return normalized_l2(
        q_feat[: len(feature_names)],
        r_feat[: len(feature_names)],
        mask=common.astype(np.float32),
        scale=scale,
    )


def compute_operation_distance(
    query: Mapping[str, object] | object,
    reference: Mapping[str, object] | object,
    config: Mapping[str, object],
) -> float:
    """计算历史工况距离 d_operation。

    中文含义：
    - 用于比较充放电倍率、温度、DoD、协议切换频率等工况压力是否相似。
    - 该距离反映 query 和 reference 在历史运行条件上的接近程度。

    输入：
    - query/reference: 包含命名工况特征和可选 `operation_mask` 的字典。
    - config: `configs/rag_retrieval_features.yaml` 中 `distance_components.d_operation` 对应的配置。

    输出：
    - float，越小越相似。
    """

    feature_cfg = dict(dict(config or {}).get("features", {}) or {})
    enabled_features = [name for name, sub_cfg in feature_cfg.items() if bool(dict(sub_cfg or {}).get("enabled", False))]
    if not enabled_features:
        return 0.0

    q_values = np.asarray([_safe_float(_get_value(query, name, 0.0)) for name in enabled_features], dtype=np.float32)
    r_values = np.asarray([_safe_float(_get_value(reference, name, 0.0)) for name in enabled_features], dtype=np.float32)
    q_mask = np.asarray([_safe_float(_get_value(query, "operation_mask", {}).get(name, 1.0) if isinstance(_get_value(query, "operation_mask", {}), Mapping) else 1.0, 1.0) for name in enabled_features], dtype=np.float32)
    r_mask = np.asarray([_safe_float(_get_value(reference, "operation_mask", {}).get(name, 1.0) if isinstance(_get_value(reference, "operation_mask", {}), Mapping) else 1.0, 1.0) for name in enabled_features], dtype=np.float32)
    common = (q_mask > 0) & (r_mask > 0)
    if not common.any():
        return 1.0
    scale = np.maximum(np.maximum(np.abs(q_values), np.abs(r_values)), 1e-4)
    return normalized_l2(q_values, r_values, mask=common.astype(np.float32), scale=scale)


def compute_metadata_distance(
    query: Mapping[str, object] | object,
    reference: Mapping[str, object] | object,
    config: Mapping[str, object],
) -> float:
    """计算元信息距离 d_metadata。

    中文含义：
    - 这是类别变量 penalty，不是连续 L2 距离。
    - chemistry family mismatch 是 soft penalty，不是默认硬过滤。

    输入：
    - query/reference: 包含 chemistry_family、domain_label、voltage_window_bucket、source_dataset 等类别字段。
    - config: `configs/rag_retrieval_features.yaml` 中 `distance_components.d_metadata` 对应的配置。

    输出：
    - float，越小表示背景越可比。
    """

    cfg = dict(config or {})
    penalties = dict(cfg.get("categorical_penalties", {}) or {})
    features_cfg = dict(cfg.get("features", {}) or {})
    comparisons = [
        ("chemistry_family", "chemistry_family_mismatch"),
        ("domain_label", "domain_label_mismatch"),
        ("voltage_window_bucket", "voltage_window_mismatch"),
        ("source_dataset", "source_dataset_mismatch"),
        ("nominal_capacity_bucket", "nominal_capacity_mismatch"),
        ("temperature_bucket", "temperature_bucket_mismatch"),
        ("charge_rate_bucket", "charge_rate_bucket_mismatch"),
    ]

    score = 0.0
    total_weight = 0.0
    for feature_name, penalty_name in comparisons:
        feature_enabled = bool(dict(features_cfg.get(feature_name, {}) or {}).get("enabled", False))
        if not feature_enabled:
            continue
        penalty = _safe_float(penalties.get(penalty_name), 1.0)
        q_value = str(_get_value(query, feature_name, "unknown"))
        r_value = str(_get_value(reference, feature_name, "unknown"))
        total_weight += penalty
        score += penalty * (0.0 if q_value == r_value and q_value != "unknown" else 1.0)
    if total_weight <= 0:
        return 0.0
    return float(score / total_weight)


def compute_tsfm_distance(
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    config: Mapping[str, object],
) -> float:
    """计算时间序列基础模型嵌入距离 d_tsfm。

    中文含义：
    - 用于比较通用时序形态。
    - 主要用于 Stage-1 粗检索和最终重排序中的弱补充项。

    输入：
    - query_embedding: query 的 tsfm_embedding。
    - reference_embedding: reference 的 tsfm_embedding。
    - config: `configs/rag_retrieval_features.yaml` 中 `distance_components.d_tsfm` 对应的配置。

    输出：
    - float，越小越相似。
    """

    if not bool(dict(config or {}).get("enabled", False)):
        return 0.0
    query_arr = _as_float_array(query_embedding)
    ref_arr = _as_float_array(reference_embedding)
    if query_arr.size == 0 or ref_arr.size == 0:
        return 1.0
    return normalized_l2(query_arr, ref_arr)


def compute_composite_distance(
    component_distances: Mapping[str, float],
    rag_config: Mapping[str, object],
) -> float:
    """计算综合检索距离 composite_distance。

    中文含义：
    - 只使用 `configs/rag_retrieval_features.yaml` 中 enabled=true 的核心距离分量。
    - 对 enabled=true 的分量按权重归一化后加权求和。

    输入：
    - component_distances: 形如 `{"d_soh_state": ..., "d_qv_shape": ...}` 的距离字典。
    - rag_config: 完整检索配置字典。

    输出：
    - composite_distance，越小越相似。
    """

    distance_cfg = dict(rag_config.get("distance_components", {}) or {})
    enabled_weights = {}
    for name in CORE_DISTANCE_NAMES:
        cfg = dict(distance_cfg.get(name, {}) or {})
        if not bool(cfg.get("enabled", False)):
            continue
        if name == "d_tsfm":
            features_cfg = dict(cfg.get("features", {}) or {})
            if not bool(dict(features_cfg.get("tsfm_in_final_rerank", {}) or {}).get("enabled", True)):
                continue
        enabled_weights[name] = max(_safe_float(cfg.get("weight", 0.0), 0.0), 0.0)

    if not enabled_weights:
        return 0.0

    normalize = bool(dict(rag_config.get("composite_distance", {}) or {}).get("normalize_weights_over_enabled_components", True))
    weight_sum = float(sum(enabled_weights.values()))
    if normalize and weight_sum > 0:
        weights = {name: value / weight_sum for name, value in enabled_weights.items()}
    else:
        weights = enabled_weights
    return float(sum(weights[name] * _safe_float(component_distances.get(name), 0.0) for name in weights))


def compute_retrieval_confidence(
    topk_component_distances: Mapping[str, object] | Sequence[Mapping[str, object]] | np.ndarray,
    rag_config: Mapping[str, object] | None = None,
) -> float:
    """计算检索置信度 retrieval_confidence。

    中文含义：
    - 表示 top-k 检索结果是否可靠。
    - 数值范围 [0, 1]，越大越可靠。

    输入：
    - topk_component_distances:
      1. 推荐传入 dict，至少包含 `composite_distance` 列表；
      2. 也可传入 query-neighbor 距离字典列表；
      3. 为兼容旧接口，也接受距离数组。
    - rag_config: 完整检索配置字典；若为空则退化为基于均值和方差的旧式置信度。

    输出：
    - retrieval_confidence，范围在 [0, 1]。
    """

    if rag_config is None:
        distances = np.asarray(topk_component_distances, dtype=np.float32)
        if distances.size == 0:
            return 0.0
        mean_distance = float(np.nanmean(distances))
        spread = float(np.nanstd(distances))
        return float(np.clip(np.exp(-(mean_distance + 0.5 * spread)), 0.0, 1.0))

    payload: Dict[str, object]
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
    if composite.size == 0:
        return 0.0

    confidence_cfg = dict((rag_config or {}).get("retrieval_confidence", {}) or {})
    factors_cfg = dict(confidence_cfg.get("factors", {}) or {})
    scores = []

    def _enabled(name: str) -> bool:
        return bool(dict(factors_cfg.get(name, {}) or {}).get("enabled", False))

    if _enabled("mean_topk_composite_distance"):
        scores.append(float(np.exp(-float(np.nanmean(composite)))))
    if _enabled("std_topk_composite_distance"):
        scores.append(float(np.exp(-float(np.nanstd(composite)))))
    if _enabled("top1_distance"):
        scores.append(float(np.exp(-float(np.nanmin(composite)))))
    if _enabled("feature_availability_ratio"):
        availability = _safe_float(payload.get("feature_availability_ratio", 1.0), 1.0)
        scores.append(float(np.clip(availability, 0.0, 1.0)))
    if _enabled("chemistry_match_rate"):
        chemistry_match = _safe_float(payload.get("chemistry_match_rate", 0.0), 0.0)
        scores.append(float(np.clip(chemistry_match, 0.0, 1.0)))
    if _enabled("domain_match_rate"):
        domain_match = _safe_float(payload.get("domain_match_rate", 0.0), 0.0)
        scores.append(float(np.clip(domain_match, 0.0, 1.0)))

    if not scores:
        return 0.0
    return float(np.clip(float(np.mean(scores)), 0.0, 1.0))


def qv_map_distance(
    query_qv: np.ndarray,
    ref_qv: np.ndarray,
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
    channel_weights: Dict[str, float] | Iterable[float],
) -> float:
    """兼容旧接口的 Q-V 曲线距离包装器。"""

    qv_cfg = {
        "features": {
            "vd_curve_q": {"enabled": True},
            "vc_curve_q": {"enabled": False},
            "delta_v_curve_q": {"enabled": True},
            "r_curve_q": {"enabled": True},
            "ic_curve_q": {"enabled": False},
            "id_curve_q": {"enabled": False},
            "qv_summary_stats": {"enabled": False},
        },
        "channel_weights": dict(channel_weights) if isinstance(channel_weights, Mapping) else {
            "Vc": 1.0,
            "Vd": 1.0,
            "Ic": 1.0,
            "Id": 1.0,
            "DeltaV": 1.0,
            "R": 1.0,
        },
    }
    return compute_qv_shape_distance(
        {"qv_map": query_qv, "qv_mask": query_mask},
        {"qv_map": ref_qv, "qv_mask": ref_mask},
        qv_cfg,
    )


def physics_feature_distance(
    query_f: np.ndarray,
    ref_f: np.ndarray,
    query_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> float:
    """兼容旧接口的 physics distance 包装器。"""

    physics_cfg = {
        "features": {name: {"enabled": True} for name in [
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
        ]}
    }
    return compute_physics_distance(
        {"physics_features": query_f, "physics_feature_mask": query_mask},
        {"physics_features": ref_f, "physics_feature_mask": ref_mask},
        physics_cfg,
    )


def operation_distance(query_op_summary: np.ndarray, ref_op_summary: np.ndarray) -> float:
    """兼容旧接口的 operation distance 包装器。"""

    enabled_names = [f"unknown_operation_feature_{idx}" for idx in range(len(_as_float_array(query_op_summary)))]
    op_cfg = {"features": {name: {"enabled": True} for name in enabled_names}}
    return compute_operation_distance(
        {name: float(value) for name, value in zip(enabled_names, _as_float_array(query_op_summary).tolist())},
        {name: float(value) for name, value in zip(enabled_names, _as_float_array(ref_op_summary).tolist())},
        op_cfg,
    )


def metadata_distance(query_meta: Dict[str, object], ref_meta: Dict[str, object], weights: Dict[str, float]) -> float:
    """兼容旧接口的 metadata distance 包装器。"""

    features = {}
    penalties = {}
    for field, penalty in weights.items():
        features[field] = {"enabled": True}
        penalties[f"{field}_mismatch"] = penalty
    cfg = {"features": features, "categorical_penalties": penalties}
    return compute_metadata_distance(query_meta, ref_meta, cfg)


def soh_state_distance(query_state: Dict[str, float], ref_state: Dict[str, float]) -> float:
    """兼容旧接口的 SOH state distance 包装器。"""

    cfg = {
        "features": {
            "anchor_soh": {"enabled": True},
            "recent_soh_slope": {"enabled": True},
            "recent_soh_curvature": {"enabled": True},
            "degradation_stage": {"enabled": False},
            "normalized_cycle_index": {"enabled": False},
        }
    }
    return compute_soh_state_distance(query_state, ref_state, cfg)
