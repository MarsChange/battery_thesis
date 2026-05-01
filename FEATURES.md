# Features

本文件由 `battery_data.feature_registry` 生成，说明当前数值 SOH 预测框架的检索特征、预测输入和 Router 输入。

| feature_name | 中文含义 | 英文含义 | group | role | shape | unit | source_signal | extraction_method | missing_handling | default_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expert_seq | 专家 LSTM 输入序列 | expert LSTM input sequence | expert | expert_input | [lookback,F] | mixed | SOH/Q-V/operation features | stack named per-cycle features | 缺失维度填 0。 | true |
| charge_current_seq | 充电电流滑窗序列 | charge-current lookback sequence | metadata | retrieval | [lookback] | A or normalized current | qv_maps channel Ic(Q) | cycle-wise mean absolute Ic(Q) | 缺失 cycle 记为 NaN，仅共同可用位置参与距离。 | true |
| chemistry_family | 电池化学体系 | battery chemistry family | metadata | retrieval, router | [] | category | adapter metadata | read from dataset metadata | 缺失记为 unknown。 | true |
| discharge_current_seq | 放电电流滑窗序列 | discharge-current lookback sequence | metadata | retrieval | [lookback] | A or normalized current | qv_maps channel Id(Q) | cycle-wise mean absolute Id(Q) | 缺失 cycle 记为 NaN，仅共同可用位置参与距离。 | true |
| domain_label | 工况域标签 | domain label | metadata | retrieval | [] | category | chemistry and operation metadata | domain labeling rule | 缺失记为 unknown。 | true |
| normalized_capacity_delta_seq | 归一化容量变化滑窗序列 | normalized-capacity-change lookback sequence | metadata | retrieval | [lookback] | SOH difference | soh_seq | cycle-to-cycle difference of normalized capacity/SOH | 首个位置填 0；缺失位置记为 NaN。 | true |
| temperature_seq | 温度滑窗序列 | temperature lookback sequence | metadata | retrieval | [lookback] | degC | operation_seq temp_mean | read per-cycle temp_mean | 缺失 cycle 记为 NaN，仅共同可用位置参与距离。 | true |
| voltage_window_bucket | 电压窗口类别 | voltage window bucket | metadata | retrieval | [] | category | voltage min/max | bucket voltage limits | 缺失记为 unknown。 | true |
| base_delta | 基础多步预测 | base multi-step delta prediction | model_output | diagnostics | [horizon] | SOH difference | fm/rag/pair branches | fusion_base(fm_delta, rag_delta, pair_delta) | 模型实时计算。 | true |
| moe_residual | 专家残差修正 | mixture-of-experts residual correction | model_output | diagnostics | [horizon] | SOH difference | LSTM experts | router-weighted residual sum | 模型实时计算。 | true |
| protocol_change_rate | 协议切换频率 | protocol change rate | operation | router, expert_input | [] | ratio | operation sequence | fraction of significant rate/temperature changes | 不可计算时填 0。 | true |
| temperature_max | 最高温度 | maximum temperature | operation | router, expert_input | [] | degC | temperature | max over window | 不可计算时 mask 为 0。 | true |
| temperature_mean | 温度均值 | mean temperature | operation | router | [] | degC | temperature | mean over window | 不可计算时 mask 为 0。 | true |
| temperature_std | 温度标准差 | standard deviation of temperature | operation | router | [] | degC | temperature | std over window | 不可计算时 mask 为 0。 | true |
| partial_charge_curve | partial charging 曲线 | partial charging curve | partial_charge | prediction_input, diagnostics | [50] | Ah or proxy | charge segment | integrate /I/ dt over voltage grid | 不可用时填 0 并由 partial_charge_mask 标记。 | true |
| physics_features | ΔV/R 物理 proxy 特征 | DeltaV/R physics proxy features | physics | prediction_input, retrieval, router | [12] | mixed | qv_summary and partial_charge | compose named proxy vector | 逐维 mask 标记。 | true |
| delta_v_curve_q | Q 轴充放电电压差曲线 | charge-discharge voltage gap curve | qv_shape | retrieval, router | [q_grid] | V | qv_maps channel DeltaV | Vc(Q)-Vd(Q) | 缺失由 qv_masks 标记。 | true |
| qv_maps | Q 轴对齐容量-电压多通道曲线 | Q-indexed voltage-current feature maps | qv_shape | prediction_input, retrieval, visualization | [lookback,6,q_grid] | V/A/proxy | raw voltage/current/capacity | interpolate Vc,Vd,Ic,Id,DeltaV,R on normalized Q | 缺失通道由 qv_masks 标记。 | true |
| r_curve_q | Q 轴极化/内阻 proxy 曲线 | Q-indexed polarization or resistance proxy | qv_shape | retrieval, router | [q_grid] | proxy | qv_maps channel R | DeltaV/(Ic-Id+eps) | 分母过小时用 eps，极端值 clip。 | true |
| vd_curve_q | Q 轴放电电压曲线 | discharge voltage curve indexed by normalized capacity | qv_shape | retrieval | [q_grid] | V | qv_maps channel Vd | select Vd(Q) | 缺失由 qv_masks 标记。 | true |
| delta_v_mean | 充放电电压差均值 | mean charge-discharge voltage gap | qv_summary | retrieval, router, expert_input | [] | V | DeltaV(Q) | mean over Q grid | 缺失填 0 并由 mask 标记。 | true |
| delta_v_q95 | 充放电电压差 95 分位数 | 95th percentile of charge-discharge voltage gap | qv_summary | retrieval, router, expert_input | [] | V | DeltaV(Q) | 95th percentile over Q grid | 缺失时可用 delta_v_max 近似。 | true |
| delta_v_std | 充放电电压差标准差 | standard deviation of charge-discharge voltage gap | qv_summary | retrieval, router, expert_input | [] | V | DeltaV(Q) | standard deviation over Q grid | 缺失填 0 并由 mask 标记。 | true |
| r_mean | 内阻 proxy 均值 | mean resistance proxy | qv_summary | retrieval, router, expert_input | [] | proxy | R(Q) | mean over Q grid | 缺失填 0 并由 mask 标记。 | true |
| r_q95 | 内阻 proxy 95 分位数 | 95th percentile of resistance proxy | qv_summary | retrieval, router, expert_input | [] | proxy | R(Q) | 95th percentile over Q grid | 缺失填 0 并由 mask 标记。 | true |
| r_std | 内阻 proxy 标准差 | standard deviation of resistance proxy | qv_summary | retrieval, router, expert_input | [] | proxy | R(Q) | standard deviation over Q grid | 缺失填 0 并由 mask 标记。 | true |
| vc_curve_slope_mean | 充电电压曲线平均斜率 | mean slope of charge voltage curve | qv_summary | retrieval, router | [] | V/Q | Vc(Q) | mean gradient over Q | 缺失填 0 并由 mask 标记。 | true |
| vd_curve_slope_mean | 放电电压曲线平均斜率 | mean slope of discharge voltage curve | qv_summary | retrieval, router | [] | V/Q | Vd(Q) | mean gradient over Q | 缺失填 0 并由 mask 标记。 | true |
| composite_distance | 综合检索距离 | composite retrieval distance | retrieval | retrieval_output, diagnostics | [] | distance | enabled retrieval components | weighted sum | 只使用 YAML 中启用的分量。 | true |
| d_metadata | 元信息与原始工况数值距离 | metadata and raw operating-signal distance | retrieval | retrieval_output, diagnostics | [] | distance | metadata plus raw current/temperature/capacity-change sequences | categorical penalty plus mask-aware sequence distance | 类别 unknown 按 mismatch；数值项仅共同可用位置参与计算。 | true |
| d_operation | 工况距离 | operation condition distance | retrieval | retrieval_output, diagnostics | [] | distance | legacy operation summary | disabled compatibility distance | 默认不使用。 | false |
| d_physics | 物理启发特征距离 | physics-inspired feature distance | retrieval | retrieval_output, diagnostics | [] | distance | DeltaV/R proxy features | mask-aware normalized distance | 共同可用维度参与计算。 | true |
| d_qv_shape | 容量-电压曲线形状距离 | Q-V curve shape distance | retrieval | retrieval_output, diagnostics | [] | distance | qv_maps and qv_summary | masked curve distance | 共同可用通道参与计算。 | true |
| d_soh_state | SOH 状态距离 | SOH state distance | retrieval | retrieval_output, diagnostics | [] | distance | anchor_soh/slope/curvature/stage | normalized distance | 必需检索分量。 | true |
| retrieval_confidence | 检索置信度 | retrieval confidence | retrieval | retrieval_output, router | [] | ratio | top-k distances and match rates | bounded confidence score | 无有效邻居时为 0。 | true |
| anchor_soh | 锚点 SOH | anchor state of health | state | prediction_input, retrieval, router | [] | ratio | SOH sequence | last SOH in lookback window | 必需特征，缺失时该 case 不用于监督预测。 | true |
| degradation_stage | 退化阶段 | degradation stage | state | retrieval, diagnostics | [] | category | anchor_soh | bucket by SOH | 缺失记为 unknown。 | true |
| recent_soh_curvature | 近期 SOH 退化曲率 | recent SOH degradation curvature | state | retrieval, router | [] | SOH/cycle^2 | soh_seq | second finite difference | lookback 太短时填 0。 | true |
| recent_soh_slope | 近期 SOH 衰减斜率 | recent SOH degradation slope | state | retrieval, router | [] | SOH/cycle | soh_seq | linear or finite-difference slope | 可由 soh_seq 计算。 | true |
| soh_seq | 历史 SOH 序列 | historical SOH sequence | state | prediction_input | [lookback] | ratio | cycle SOH | collect SOH over the lookback window | 必需特征。 | true |
| target_delta_soh | 未来 SOH 变化量标签 | future delta SOH target | target | label | [horizon] | SOH difference | future_soh and anchor_soh | future_soh - anchor_soh | 必需标签。 | true |
| target_soh | 未来 SOH 真实值 | future SOH target | target | evaluation | [horizon] | ratio | future SOH | read future SOH | 评估必需。 | true |

## Feature Descriptions

### `expert_seq`
- 中文含义：专家 LSTM 输入序列
- 英文含义：expert LSTM input sequence
- 说明：小专家模型的滑动窗口数值序列输入。

### `charge_current_seq`
- 中文含义：充电电流滑窗序列
- 英文含义：charge-current lookback sequence
- 说明：用于 d_metadata 的原始数值匹配，比较 query/reference 在历史窗口内的充电电流水平。

### `chemistry_family`
- 中文含义：电池化学体系
- 英文含义：battery chemistry family
- 说明：LFP/NCM/NCA 等材料体系。

### `discharge_current_seq`
- 中文含义：放电电流滑窗序列
- 英文含义：discharge-current lookback sequence
- 说明：用于 d_metadata 的原始数值匹配，比较历史窗口内放电电流水平。

### `domain_label`
- 中文含义：工况域标签
- 英文含义：domain label
- 说明：材料体系与工况组合域。

### `normalized_capacity_delta_seq`
- 中文含义：归一化容量变化滑窗序列
- 英文含义：normalized-capacity-change lookback sequence
- 说明：用于 d_metadata 的原始数值匹配，比较历史窗口内归一化容量变化。

### `temperature_seq`
- 中文含义：温度滑窗序列
- 英文含义：temperature lookback sequence
- 说明：用于 d_metadata 的原始数值匹配，比较历史窗口内温度轨迹。

### `voltage_window_bucket`
- 中文含义：电压窗口类别
- 英文含义：voltage window bucket
- 说明：避免比较不可比电压窗口。

### `base_delta`
- 中文含义：基础多步预测
- 英文含义：base multi-step delta prediction
- 说明：专家残差修正前的基础 delta SOH 预测。

### `moe_residual`
- 中文含义：专家残差修正
- 英文含义：mixture-of-experts residual correction
- 说明：小专家库输出的 residual correction。

### `protocol_change_rate`
- 中文含义：协议切换频率
- 英文含义：protocol change rate
- 说明：衡量变工况强度；当前不作为独立 RAG 检索细项。

### `temperature_max`
- 中文含义：最高温度
- 英文含义：maximum temperature
- 说明：高温暴露 proxy；检索中的温度原始数值由 metadata.temperature_seq 表示。

### `temperature_mean`
- 中文含义：温度均值
- 英文含义：mean temperature
- 说明：平均热环境；不作为默认检索工况距离。

### `temperature_std`
- 中文含义：温度标准差
- 英文含义：standard deviation of temperature
- 说明：热环境波动；不作为默认检索工况距离。

### `partial_charge_curve`
- 中文含义：partial charging 曲线
- 英文含义：partial charging curve
- 说明：可选充电容量-电压辅助曲线。

### `physics_features`
- 中文含义：ΔV/R 物理 proxy 特征
- 英文含义：DeltaV/R physics proxy features
- 说明：包含 ΔV/R 统计和 partial-charge summary 的低维物理特征。

### `delta_v_curve_q`
- 中文含义：Q 轴充放电电压差曲线
- 英文含义：charge-discharge voltage gap curve
- 说明：表示极化强度的曲线 proxy。

### `qv_maps`
- 中文含义：Q 轴对齐容量-电压多通道曲线
- 英文含义：Q-indexed voltage-current feature maps
- 说明：用于比较容量-电压曲线形状和极化 proxy。

### `r_curve_q`
- 中文含义：Q 轴极化/内阻 proxy 曲线
- 英文含义：Q-indexed polarization or resistance proxy
- 说明：近似表示容量维度上的极化或内阻状态。

### `vd_curve_q`
- 中文含义：Q 轴放电电压曲线
- 英文含义：discharge voltage curve indexed by normalized capacity
- 说明：比较放电平台和曲线形状。

### `delta_v_mean`
- 中文含义：充放电电压差均值
- 英文含义：mean charge-discharge voltage gap
- 说明：平均极化 proxy。

### `delta_v_q95`
- 中文含义：充放电电压差 95 分位数
- 英文含义：95th percentile of charge-discharge voltage gap
- 说明：捕捉局部强极化。

### `delta_v_std`
- 中文含义：充放电电压差标准差
- 英文含义：standard deviation of charge-discharge voltage gap
- 说明：极化沿容量轴的变化程度。

### `r_mean`
- 中文含义：内阻 proxy 均值
- 英文含义：mean resistance proxy
- 说明：平均内阻/极化 proxy。

### `r_q95`
- 中文含义：内阻 proxy 95 分位数
- 英文含义：95th percentile of resistance proxy
- 说明：捕捉局部高极化/高阻抗 proxy。

### `r_std`
- 中文含义：内阻 proxy 标准差
- 英文含义：standard deviation of resistance proxy
- 说明：R(Q) 波动程度。

### `vc_curve_slope_mean`
- 中文含义：充电电压曲线平均斜率
- 英文含义：mean slope of charge voltage curve
- 说明：表征充电平台形态。

### `vd_curve_slope_mean`
- 中文含义：放电电压曲线平均斜率
- 英文含义：mean slope of discharge voltage curve
- 说明：表征放电平台形态。

### `composite_distance`
- 中文含义：综合检索距离
- 英文含义：composite retrieval distance
- 说明：由启用距离分量加权得到，越小越相似。

### `d_metadata`
- 中文含义：元信息与原始工况数值距离
- 英文含义：metadata and raw operating-signal distance
- 说明：化学体系、domain、电压窗口，以及充放电电流、温度、归一化容量变化原始滑窗序列的综合距离。

### `d_operation`
- 中文含义：工况距离
- 英文含义：operation condition distance
- 说明：兼容旧输出的工况距离；当前默认关闭，不参与主检索。

### `d_physics`
- 中文含义：物理启发特征距离
- 英文含义：physics-inspired feature distance
- 说明：ΔV(Q) 与 R(Q) 物理 proxy 距离。

### `d_qv_shape`
- 中文含义：容量-电压曲线形状距离
- 英文含义：Q-V curve shape distance
- 说明：容量-电压曲线形态距离。

### `d_soh_state`
- 中文含义：SOH 状态距离
- 英文含义：SOH state distance
- 说明：query 与 reference 当前健康状态和趋势距离。

### `retrieval_confidence`
- 中文含义：检索置信度
- 英文含义：retrieval confidence
- 说明：top-k 检索结果可靠性。

### `anchor_soh`
- 中文含义：锚点 SOH
- 英文含义：anchor state of health
- 说明：当前窗口最后一个循环的 SOH，用于恢复 pred_soh = anchor_soh + pred_delta_soh。

### `degradation_stage`
- 中文含义：退化阶段
- 英文含义：degradation stage
- 说明：early/middle/late 等退化阶段标签。

### `recent_soh_curvature`
- 中文含义：近期 SOH 退化曲率
- 英文含义：recent SOH degradation curvature
- 说明：反映 SOH 是否出现加速退化。

### `recent_soh_slope`
- 中文含义：近期 SOH 衰减斜率
- 英文含义：recent SOH degradation slope
- 说明：近几循环 SOH 变化率，用于捕捉衰退趋势。

### `soh_seq`
- 中文含义：历史 SOH 序列
- 英文含义：historical SOH sequence
- 说明：输入窗口内连续循环 SOH 序列。

### `target_delta_soh`
- 中文含义：未来 SOH 变化量标签
- 英文含义：future delta SOH target
- 说明：统一监督目标：target_delta_soh[h] = future_soh[h] - anchor_soh。

### `target_soh`
- 中文含义：未来 SOH 真实值
- 英文含义：future SOH target
- 说明：未来 horizon 的真实 SOH。
