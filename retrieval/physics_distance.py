"""Named numerical distances for battery RAG retrieval.

The retrieval method uses only interpretable battery-specific components:
`d_soh_state`, `d_qv_shape`, `d_physics`, and `d_metadata`.
`d_operation` is retained as a compatibility output but is disabled by default.
Temperature/current/power summaries can be matched through `d_metadata` when
enabled in the retrieval config.
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

QV_CHANNEL_TO_INDEX = {
    "V_window": 0,
    "V": 0,
    "Qd": 1,
    "Qd(V)": 1,
    "dQdV": 2,
    "dQdV(V)": 2,
    "Id_abs": 3,
    "Temp": 4,
    "Power_abs": 5,
    # Backward-compatible aliases for old test fixtures/artifacts.
    "Vc": 0,
    "Vd": 1,
    "Ic": 3,
    "Id": 3,
    "DeltaV": 2,
    "R": 5,
}
PHYSICS_PROXY_NAMES = [
    "qv_dqdv_peak_value",
    "qv_dqdv_peak_voltage",
    "qv_dqdv_area",
    "qv_capacity_span",
]


def _recency_weights(length: int, power: float = 1.5) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    steps = np.linspace(1.0 / float(length), 1.0, length, dtype=np.float32)
    weights = np.power(steps, max(float(power), 0.0)).astype(np.float32)
    return weights / max(float(np.mean(weights)), 1e-6)


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


def _anchored_soh_history(record: Mapping[str, object] | object) -> np.ndarray:
    """Return SOH history aligned so the anchor cycle is zero.

    The retrieval plot compares every reference and query as delta-SOH from
    the window anchor. This helper uses the same representation, so retrieval
    ranking is driven by the visible historical degradation shape rather than
    only by anchor SOH or a short-window slope.
    """

    seq = _as_float_array(_get_value(record, "soh_seq", np.zeros(0, dtype=np.float32))).reshape(-1)
    finite = seq[np.isfinite(seq)]
    if finite.size <= 1:
        return np.zeros(0, dtype=np.float32)
    seq = seq.astype(np.float32, copy=True)
    if not np.isfinite(seq[-1]):
        seq[-1] = finite[-1]
    return (seq - seq[-1]).astype(np.float32)


def _weighted_rmse(diff: np.ndarray, mask: np.ndarray, weights: np.ndarray, scale: float) -> float:
    common = mask.astype(bool) & np.isfinite(diff)
    if not common.any():
        return 1.0
    w = np.asarray(weights, dtype=np.float32)
    if w.shape != diff.shape:
        w = np.ones_like(diff, dtype=np.float32)
    w = np.where(common, w, 0.0).astype(np.float32)
    denom = max(float(w.sum()), 1e-6)
    normalized = diff / max(float(scale), 1e-8)
    return float(np.sqrt(float(np.sum(w * normalized * normalized)) / denom))


def soh_history_shape_distance(
    query_soh_seq: np.ndarray,
    ref_soh_seq: np.ndarray,
    *,
    history_scale: float = 0.015,
    slope_scale: float = 0.0008,
    recency_power: float = 1.5,
) -> tuple[float, float]:
    """计算历史 SOH 形态距离和历史斜率形态距离。

    中文含义：把 query/ref 的历史 SOH 都对齐到窗口锚点，比较过去
    lookback 个循环的退化轨迹形状，以及相邻循环 SOH 变化序列。
    输入：`query_soh_seq`、`ref_soh_seq` 为历史窗口 SOH 序列。
    输出：两个 float，分别为历史形状距离和斜率形状距离，越小表示越相似。
    """

    q_hist = _as_float_array(query_soh_seq).reshape(-1)
    r_hist = _as_float_array(ref_soh_seq).reshape(-1)
    length = min(q_hist.size, r_hist.size)
    if length <= 1:
        return 1.0, 1.0
    q_hist = q_hist[-length:]
    r_hist = r_hist[-length:]
    q_anchor = q_hist[np.isfinite(q_hist)][-1] if np.isfinite(q_hist).any() else 0.0
    r_anchor = r_hist[np.isfinite(r_hist)][-1] if np.isfinite(r_hist).any() else 0.0
    q_delta = q_hist - q_anchor
    r_delta = r_hist - r_anchor
    mask = np.isfinite(q_delta) & np.isfinite(r_delta)
    weights = _recency_weights(length, power=recency_power)
    history_distance = _weighted_rmse(q_delta - r_delta, mask, weights, history_scale)

    q_slope = np.diff(q_hist)
    r_slope = np.diff(r_hist)
    slope_mask = np.isfinite(q_slope) & np.isfinite(r_slope)
    slope_weights = _recency_weights(length - 1, power=recency_power)
    slope_distance = _weighted_rmse(q_slope - r_slope, slope_mask, slope_weights, slope_scale)
    return float(history_distance), float(slope_distance)


def curve_shape_distance(query_curve: np.ndarray, ref_curve: np.ndarray) -> float:
    """计算单通道 Q-V 曲线形态距离。

    中文含义：先对每条曲线做均值中心化和标准差归一化，再比较形状；
    同时保留一个较小的均值偏移项，避免不同量纲的 Qd(V) 或 dQ/dV(V)
    数值幅值压倒 SOH 历史形状或其他检索分量。
    输出：float，越小表示曲线形状越相似。
    """

    q = _as_float_array(query_curve).reshape(-1)
    r = _as_float_array(ref_curve).reshape(-1)
    length = min(q.size, r.size)
    if length <= 1:
        return 1.0
    q = q[-length:]
    r = r[-length:]
    mask = np.isfinite(q) & np.isfinite(r)
    if not mask.any():
        return 1.0
    q_valid = q[mask]
    ref_valid = r[mask]
    q_mean = float(np.mean(q_valid))
    ref_mean = float(np.mean(ref_valid))
    q_std = max(float(np.std(q_valid)), 1e-4)
    ref_std = max(float(np.std(ref_valid)), 1e-4)
    q_shape = (q_valid - q_mean) / q_std
    ref_shape = (ref_valid - ref_mean) / ref_std
    shape_distance = float(np.sqrt(np.mean((q_shape - ref_shape) ** 2)))
    level_scale = max(abs(q_mean), abs(ref_mean), q_std, ref_std, 1e-4)
    level_distance = abs(q_mean - ref_mean) / level_scale
    return float(0.85 * shape_distance + 0.15 * level_distance)


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
    输入：anchor_soh、recent_soh_slope、recent_soh_curvature、
    degradation_stage，以及可选的 soh_seq 历史窗口。
    输出：float，越小表示当前 SOH、趋势、阶段和历史 64 步形态越相似。
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
    soh_cfg = dict(config.get("soh_state", {}) or {})
    state_dist = normalized_l2(q, r, scale=np.asarray([0.1, 0.01, 0.005], dtype=np.float32))
    stage_dist = degradation_stage_distance(str(_get_value(query, "degradation_stage", "unknown")), str(_get_value(reference, "degradation_stage", "unknown")))
    history_dist, slope_shape_dist = soh_history_shape_distance(
        _get_value(query, "soh_seq", np.zeros(0, dtype=np.float32)),
        _get_value(reference, "soh_seq", np.zeros(0, dtype=np.float32)),
        history_scale=_safe_float(soh_cfg.get("history_shape_scale", 0.015), 0.015),
        slope_scale=_safe_float(soh_cfg.get("history_slope_scale", 0.0008), 0.0008),
        recency_power=_safe_float(soh_cfg.get("recency_weight_power", 1.5), 1.5),
    )
    weights = {
        "current": max(_safe_float(soh_cfg.get("current_state_weight", 0.30), 0.30), 0.0),
        "history": max(_safe_float(soh_cfg.get("history_shape_weight", 0.50), 0.50), 0.0),
        "slope": max(_safe_float(soh_cfg.get("history_slope_weight", 0.15), 0.15), 0.0),
        "stage": max(_safe_float(soh_cfg.get("stage_weight", 0.05), 0.05), 0.0),
    }
    denom = max(float(sum(weights.values())), 1e-6)
    return float(
        (
            weights["current"] * state_dist
            + weights["history"] * history_dist
            + weights["slope"] * slope_shape_dist
            + weights["stage"] * stage_dist
        )
        / denom
    )


def _qv_summary_vector(record: Mapping[str, object] | object) -> tuple[np.ndarray, np.ndarray]:
    stats = dict(_get_value(record, "qv_summary_stats", {}) or {})
    values = np.asarray([_safe_float(stats.get(name, 0.0)) for name in PHYSICS_PROXY_NAMES], dtype=np.float32)
    mask = np.asarray([1.0 if name in stats else 0.0 for name in PHYSICS_PROXY_NAMES], dtype=np.float32)
    return values, mask


def compute_qv_shape_distance(query: Mapping[str, object] | object, reference: Mapping[str, object] | object, config: Mapping[str, object]) -> float:
    """计算容量-电压曲线形状距离 d_qv_shape。

    中文含义：比较放电电压窗口内的 Qd(V)、dQ/dV(V) 和 Q-V summary 统计量。
    输入：qv_map、qv_mask、qv_summary_stats。
    输出：float，越小表示放电 Q-V 窗口峰形和容量分布越相似。
    """

    query_qv = _as_float_array(_get_value(query, "qv_map", _get_value(query, "qv_maps", np.zeros((6, 0), dtype=np.float32))))
    ref_qv = _as_float_array(_get_value(reference, "qv_map", _get_value(reference, "qv_maps", np.zeros((6, 0), dtype=np.float32))))
    query_mask = _as_float_array(_get_value(query, "qv_mask", _get_value(query, "qv_masks", np.zeros(query_qv.shape[0], dtype=np.float32))))
    ref_mask = _as_float_array(_get_value(reference, "qv_mask", _get_value(reference, "qv_masks", np.zeros(ref_qv.shape[0], dtype=np.float32))))

    qv_cfg = dict(config.get("qv_shape", {}) or config or {})
    weights = dict(qv_cfg.get("channel_weights", {}) or {"Qd": 0.35, "dQdV": 0.65})
    channel_names = list(qv_cfg.get("channels", ["Qd", "dQdV"]))

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
        terms.append(weight * curve_shape_distance(query_qv[channel_idx], ref_qv[channel_idx]))
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

    中文含义：比较 query 和 reference 的放电 Q-V 窗口 summary。
    输入：physics_features 或 qv_summary_stats 中的 dQ/dV 峰值、峰值电压、
    dQ/dV 面积和窗口容量跨度。
    输出：float，越小表示 Q-V 窗口峰形统计越相似。
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
    - 数值项：current_abs_seq、temperature_seq、power_energy_seq、
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
    future_delta_dispersion = _safe_float(payload.get("future_delta_dispersion", np.nan), np.nan)
    dispersion_scale = _safe_float(payload.get("future_delta_dispersion_scale", 0.015), 0.015)
    alpha_entropy = _safe_float(payload.get("alpha_entropy", np.nan), np.nan)
    unique_cell_ratio = _safe_float(payload.get("unique_cell_ratio", np.nan), np.nan)
    scores = [
        float(np.exp(-float(np.mean(composite)))),
        float(np.exp(-float(np.std(composite)))),
        float(np.exp(-float(np.min(composite)))),
        float(np.clip(_safe_float(payload.get("feature_availability_ratio", 1.0), 1.0), 0.0, 1.0)),
        float(np.clip(_safe_float(payload.get("chemistry_match_rate", 0.0), 0.0), 0.0, 1.0)),
    ]
    if np.isfinite(future_delta_dispersion):
        scores.append(float(np.exp(-max(future_delta_dispersion, 0.0) / max(dispersion_scale, 1e-6))))
    if np.isfinite(alpha_entropy):
        # Normalized entropy close to 1 means all neighbors have similar alpha,
        # usually indicating no clearly dominant reference case. Keep this as a
        # soft reliability term rather than a hard penalty because equal alpha is
        # acceptable when the reference trajectories themselves agree.
        scores.append(float(np.clip(1.0 - 0.7 * alpha_entropy, 0.0, 1.0)))
    if np.isfinite(unique_cell_ratio):
        scores.append(float(np.clip(unique_cell_ratio, 0.0, 1.0)))
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
