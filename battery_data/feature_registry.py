"""Unified feature registry for the numerical battery SOH pipeline.

Every core feature has an explicit Chinese/English meaning, source signal,
unit, extraction method, usage role and missing-data policy. The registry is
used by documentation, tests and retrieval diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    chinese_name: str
    english_name: str
    group: str
    role: List[str]
    shape: str
    unit: str
    source_signal: str
    extraction_method: str
    description: str
    missing_handling: str
    default_enabled: bool


def _spec(name: str, chinese: str, english: str, group: str, role: List[str], shape: str, unit: str, source: str, method: str, desc: str, missing: str, enabled: bool = True) -> FeatureSpec:
    return FeatureSpec(name, chinese, english, group, list(role), shape, unit, source, method, desc, missing, bool(enabled))


def get_feature_registry() -> Dict[str, FeatureSpec]:
    specs = [
        _spec("anchor_soh", "锚点 SOH", "anchor state of health", "state", ["prediction_input", "retrieval", "router"], "[]", "ratio", "SOH sequence", "last SOH in lookback window", "当前窗口最后一个循环的 SOH，用于恢复 pred_soh = anchor_soh + pred_delta_soh。", "必需特征，缺失时该 case 不用于监督预测。"),
        _spec("soh_seq", "历史 SOH 序列", "historical SOH sequence", "state", ["prediction_input"], "[lookback]", "ratio", "cycle SOH", "collect SOH over the lookback window", "输入窗口内连续循环 SOH 序列。", "必需特征。"),
        _spec("recent_soh_slope", "近期 SOH 衰减斜率", "recent SOH degradation slope", "state", ["retrieval", "router"], "[]", "SOH/cycle", "soh_seq", "linear or finite-difference slope", "近几循环 SOH 变化率，用于捕捉衰退趋势。", "可由 soh_seq 计算。"),
        _spec("recent_soh_curvature", "近期 SOH 退化曲率", "recent SOH degradation curvature", "state", ["retrieval", "router"], "[]", "SOH/cycle^2", "soh_seq", "second finite difference", "反映 SOH 是否出现加速退化。", "lookback 太短时填 0。"),
        _spec("degradation_stage", "退化阶段", "degradation stage", "state", ["retrieval", "diagnostics"], "[]", "category", "anchor_soh", "bucket by SOH", "early/middle/late 等退化阶段标签。", "缺失记为 unknown。"),
        _spec("target_delta_soh", "未来 SOH 变化量标签", "future delta SOH target", "target", ["label"], "[horizon]", "SOH difference", "future_soh and anchor_soh", "future_soh - anchor_soh", "统一监督目标：target_delta_soh[h] = future_soh[h] - anchor_soh。", "必需标签。"),
        _spec("target_soh", "未来 SOH 真实值", "future SOH target", "target", ["evaluation"], "[horizon]", "ratio", "future SOH", "read future SOH", "未来 horizon 的真实 SOH。", "评估必需。"),
        _spec("qv_maps", "放电电压窗口 Q-V 多通道曲线", "discharge-window Q-V feature maps", "qv_shape", ["prediction_input", "retrieval", "visualization"], "[lookback,6,q_grid]", "V/A/Ah/proxy", "raw discharge voltage/current/capacity/temperature", "interpolate Qd(V), dQ/dV(V), abs(I), temperature and abs(V times I) on a chemistry-specific voltage window", "MIT/HUST LFP 使用 2.8-3.6V；TJU/XJTU NCM/NCA 使用 3.6-4.1V。", "缺失通道由 qv_masks 标记；MIT 第一个标定 cycle 不用于 Q-V 特征。"),
        _spec("qv_dqdv_peak_value", "Q-V 区段 dQ/dV 峰值", "dQ/dV peak value in voltage window", "qv_summary", ["retrieval", "router", "expert_input"], "[]", "capacity/V", "qv_maps channel dQdV", "maximum dQ/dV in selected discharge voltage window", "描述 Q-V 区段最强峰值，反映局部容量-电压形态。", "窗口不可用时填 0 且 mask=0。"),
        _spec("qv_dqdv_peak_voltage", "Q-V 区段 dQ/dV 峰值电压", "voltage at dQ/dV peak", "qv_summary", ["retrieval", "router", "expert_input"], "[]", "V", "qv_maps channel V_window and dQdV", "voltage coordinate of dQ/dV maximum", "描述峰值位置，用于比较平台/峰位偏移。", "窗口不可用时填 0 且 mask=0。"),
        _spec("qv_dqdv_area", "Q-V 区段 dQ/dV 面积", "area under dQ/dV over voltage window", "qv_summary", ["retrieval", "router", "expert_input"], "[]", "capacity", "qv_maps channel dQdV", "integral of dQ/dV over selected voltage window", "近似表示该电压窗口覆盖的容量贡献。", "窗口不可用时填 0 且 mask=0。"),
        _spec("qv_capacity_span", "Q-V 区段容量跨度", "capacity span in voltage window", "qv_summary", ["retrieval", "router", "expert_input"], "[]", "Ah or normalized capacity", "qv_maps channel Qd", "max(Qd)-min(Qd) inside selected voltage window", "表示放电窗口内可观测容量变化。", "窗口不可用时填 0 且 mask=0。"),
        _spec("qv_power_energy_proxy", "Q-V 区段功率/能量 proxy", "power or energy proxy in Q-V window", "qv_summary", ["router", "expert_input"], "[]", "W or proxy", "voltage and current", "integral/summary of |V*I| in selected voltage window", "用于高功率专家的语义证据。", "窗口不可用时填 0 且 mask=0。"),
        _spec("partial_charge_curve", "partial charging 曲线", "partial charging curve", "partial_charge", ["diagnostics"], "[50]", "Ah or proxy", "charge segment", "legacy optional extraction only", "当前主线不使用该特征训练或检索，仅保留数组兼容和诊断。", "不可用时填 0。", False),
        _spec("physics_features", "12 维 Q-V/温度/电流/功率状态特征", "12-D Q-V and operation stress features", "physics", ["prediction_input", "retrieval", "router"], "[12]", "mixed", "qv_summary and operation summaries", "compose named qv_dqdv, temperature, current, power and cycle-aging features", "包含 dQ/dV 峰值、峰值电压、面积、容量跨度、温度、电流、功率和循环老化。", "逐维 mask 标记。"),
        _spec("current_abs_seq", "绝对电流滑窗序列", "absolute-current lookback sequence", "metadata", ["retrieval"], "[lookback]", "A", "qv_maps Id_abs(V) or operation_seq", "cycle-wise mean absolute current", "用于 d_metadata 的数值匹配，比较 query/reference 在历史窗口内的电流压力。", "缺失 cycle 记为 NaN，仅共同可用位置参与距离。"),
        _spec("power_energy_seq", "功率/能量 proxy 滑窗序列", "power or energy proxy lookback sequence", "metadata", ["retrieval"], "[lookback]", "proxy", "qv_maps Power_abs(V) or energy deltas", "cycle-wise mean abs(V times I) or energy delta summary", "用于 d_metadata 的数值匹配，比较历史窗口内功率/能量压力。", "缺失 cycle 记为 NaN，仅共同可用位置参与距离。"),
        _spec("temperature_seq", "温度滑窗序列", "temperature lookback sequence", "metadata", ["retrieval"], "[lookback]", "degC", "operation_seq temp_mean", "read per-cycle temp_mean", "用于 d_metadata 的原始数值匹配，比较历史窗口内温度轨迹。", "缺失 cycle 记为 NaN，仅共同可用位置参与距离。"),
        _spec("normalized_capacity_delta_seq", "归一化容量变化滑窗序列", "normalized-capacity-change lookback sequence", "metadata", ["retrieval"], "[lookback]", "SOH difference", "soh_seq", "cycle-to-cycle difference of normalized capacity/SOH", "用于 d_metadata 的原始数值匹配，比较历史窗口内归一化容量变化。", "首个位置填 0；缺失位置记为 NaN。"),
        _spec("temperature_mean", "温度均值", "mean temperature", "operation", ["router"], "[]", "degC", "temperature", "mean over window", "平均热环境；不作为默认检索工况距离。", "不可计算时 mask 为 0。"),
        _spec("temperature_std", "温度标准差", "standard deviation of temperature", "operation", ["router"], "[]", "degC", "temperature", "std over window", "热环境波动；不作为默认检索工况距离。", "不可计算时 mask 为 0。"),
        _spec("temperature_max", "最高温度", "maximum temperature", "operation", ["router", "expert_input"], "[]", "degC", "temperature", "max over window", "高温暴露 proxy；检索中的温度原始数值由 metadata.temperature_seq 表示。", "不可计算时 mask 为 0。"),
        _spec("protocol_change_rate", "协议切换频率", "protocol change rate", "operation", ["router", "expert_input"], "[]", "ratio", "operation sequence", "fraction of significant rate/temperature changes", "衡量变工况强度；当前不作为独立 RAG 检索细项。", "不可计算时填 0。"),
        _spec("chemistry_family", "电池化学体系", "battery chemistry family", "metadata", ["retrieval", "router"], "[]", "category", "adapter metadata", "read from dataset metadata", "LFP/NCM/NCA 等材料体系。", "缺失记为 unknown。"),
        _spec("domain_label", "工况域标签", "domain label", "metadata", ["retrieval"], "[]", "category", "chemistry and operation metadata", "domain labeling rule", "材料体系与工况组合域。", "缺失记为 unknown。"),
        _spec("voltage_window_bucket", "电压窗口类别", "voltage window bucket", "metadata", ["retrieval"], "[]", "category", "voltage min/max", "bucket voltage limits", "避免比较不可比电压窗口。", "缺失记为 unknown。"),
        _spec("expert_seq", "专家 attention-GRU 输入序列", "expert attention-GRU input sequence", "expert", ["expert_input"], "[lookback,F]", "mixed", "SOH/Q-V/operation features", "stack named per-cycle features", "小专家模型的滑动窗口数值序列输入。", "缺失维度填 0。"),
        _spec("base_delta", "基础多步预测", "base multi-step delta prediction", "model_output", ["diagnostics"], "[horizon]", "SOH difference", "fm/rag/pair branches", "fusion_base(fm_delta, rag_delta, pair_delta)", "专家残差修正前的基础 delta SOH 预测。", "模型实时计算。"),
        _spec("moe_residual", "专家残差修正", "mixture-of-experts residual correction", "model_output", ["diagnostics"], "[horizon]", "SOH difference", "attention-GRU experts", "router-weighted residual sum", "小专家库输出的 residual correction。", "模型实时计算。"),
        _spec("d_soh_state", "SOH 状态距离", "SOH state distance", "retrieval", ["retrieval_output", "diagnostics"], "[]", "distance", "anchor_soh/slope/curvature/stage", "normalized distance", "query 与 reference 当前健康状态和趋势距离。", "必需检索分量。"),
        _spec("d_qv_shape", "容量-电压曲线形状距离", "Q-V curve shape distance", "retrieval", ["retrieval_output", "diagnostics"], "[]", "distance", "qv_maps and qv_summary", "masked curve distance", "容量-电压曲线形态距离。", "共同可用通道参与计算。"),
        _spec("d_physics", "Q-V 窗口物理启发特征距离", "Q-V window physics-inspired summary distance", "retrieval", ["retrieval_output", "diagnostics"], "[]", "distance", "dQ/dV peak, peak voltage, area and capacity span", "mask-aware normalized distance", "比较 dQ/dV 峰值、峰值电压、面积和容量跨度。", "共同可用维度参与计算。"),
        _spec("d_operation", "工况距离", "operation condition distance", "retrieval", ["retrieval_output", "diagnostics"], "[]", "distance", "legacy operation summary", "disabled compatibility distance", "兼容旧输出的工况距离；当前默认关闭，不参与主检索。", "默认不使用。", False),
        _spec("d_metadata", "元信息与数值工况距离", "metadata and numeric operating-signal distance", "retrieval", ["retrieval_output", "diagnostics"], "[]", "distance", "metadata plus current/temperature/power/capacity-change sequences", "categorical penalty plus mask-aware sequence distance", "化学体系、domain、电压窗口，以及电流、温度、功率/能量 proxy、归一化容量变化滑窗序列的综合距离。", "类别 unknown 按 mismatch；数值项仅共同可用位置参与计算。"),
        _spec("composite_distance", "综合检索距离", "composite retrieval distance", "retrieval", ["retrieval_output", "diagnostics"], "[]", "distance", "enabled retrieval components", "weighted sum", "由启用距离分量加权得到，越小越相似。", "只使用 YAML 中启用的分量。"),
        _spec("retrieval_confidence", "检索置信度", "retrieval confidence", "retrieval", ["retrieval_output", "router"], "[]", "ratio", "top-k distances and match rates", "bounded confidence score", "top-k 检索结果可靠性。", "无有效邻居时为 0。"),
    ]
    return {spec.name: spec for spec in specs}


def get_features_by_group(group: str) -> List[FeatureSpec]:
    return [spec for spec in get_feature_registry().values() if spec.group == group]


def get_features_by_role(role: str) -> List[FeatureSpec]:
    return [spec for spec in get_feature_registry().values() if role in spec.role]


def write_feature_registry_json(path: str | Path) -> None:
    registry = get_feature_registry()
    payload = {name: asdict(spec) for name, spec in sorted(registry.items())}
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_features_markdown(path: str | Path) -> None:
    registry = get_feature_registry()
    lines = [
        "# Features",
        "",
        "本文件由 `battery_data.feature_registry` 生成，说明当前数值 SOH 预测框架的检索特征、预测输入和 Router 输入。",
        "",
        "| feature_name | 中文含义 | 英文含义 | group | role | shape | unit | source_signal | extraction_method | missing_handling | default_enabled |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, spec in sorted(registry.items(), key=lambda item: (item[1].group, item[0])):
        lines.append(
            "| {name} | {cn} | {en} | {group} | {role} | {shape} | {unit} | {source} | {method} | {missing} | {enabled} |".format(
                name=name,
                cn=spec.chinese_name.replace("|", "/"),
                en=spec.english_name.replace("|", "/"),
                group=spec.group,
                role=", ".join(spec.role).replace("|", "/"),
                shape=spec.shape.replace("|", "/"),
                unit=spec.unit.replace("|", "/"),
                source=spec.source_signal.replace("|", "/"),
                method=spec.extraction_method.replace("|", "/"),
                missing=spec.missing_handling.replace("|", "/"),
                enabled=str(spec.default_enabled).lower(),
            )
        )
    lines.extend(["", "## Feature Descriptions", ""])
    for name, spec in sorted(registry.items(), key=lambda item: (item[1].group, item[0])):
        lines.append(f"### `{name}`")
        lines.append(f"- 中文含义：{spec.chinese_name}")
        lines.append(f"- 英文含义：{spec.english_name}")
        lines.append(f"- 说明：{spec.description}")
        lines.append("")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
