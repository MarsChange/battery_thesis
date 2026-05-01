# 当前数值 SOH 预测框架特征说明

本文档汇总当前主线框架实际使用的特征，覆盖 case bank、RAG 检索、Router、小专家模型和预测输出。本文不包含任何数据集专项验证设置。

当前主线是数值预测框架：

- 不使用 LLM。
- 不使用机理文本知识库。
- 不使用 TSFM/Chronos embedding。
- 不使用 relaxation voltage 特征。
- 预测目标固定为 `target_delta_soh = future_soh - anchor_soh`。
- 最终输出固定为 `pred_soh = anchor_soh + pred_delta_soh`。

## 特征文件与数组

| 文件或字段 | shape | 当前用途 | 说明 |
| --- | --- | --- | --- |
| `case_rows.parquet` | `[num_cases]` | metadata、split、窗口索引、锚点状态 | 不保存大数组，只保存 case id、cell id、split、domain、chemistry、anchor SOH、recent slope 等可索引字段。 |
| `case_soh_seq.npy` | `[num_cases, lookback]` | 预测输入、状态趋势、容量变化序列 | 每个滑窗内的历史 SOH 序列。 |
| `case_cycle_stats.npy` | `[num_cases, lookback, F_cycle]` | 预测输入、general sequence encoder | 每个 cycle 的命名统计特征，不包含 `soh`，因为 `soh` 单独保存为 `case_soh_seq.npy`。 |
| `case_qv_maps.npy` | `[num_cases, lookback, 6, q_grid]` | Q-V 曲线编码、RAG `d_qv_shape`、可视化 | 每个 cycle 的 Q 轴对齐曲线，通道为 `Vc,Vd,Ic,Id,DeltaV,R`。 |
| `case_qv_masks.npy` | `[num_cases, lookback, 6]` | 缺失处理 | 标记每个 Q-V 通道是否可用。 |
| `case_partial_charge.npy` | `[num_cases, lookback, 50]` | partial charge encoder、物理 proxy | 充电段累计输入电荷随电压变化的曲线。 |
| `case_partial_charge_mask.npy` | `[num_cases, lookback]` | 缺失处理 | 标记 partial charging 曲线是否可提取。 |
| `case_physics_features.npy` | `[num_cases, lookback, 12]` | RAG `d_physics`、Router、小专家上下文 | 由 Q-V 极化 proxy 和 partial charge summary 组成的 12 维物理启发特征。 |
| `case_physics_feature_masks.npy` | `[num_cases, lookback, 12]` | 缺失处理 | 标记每个 physics feature 是否可用。 |
| `case_anchor_physics_features.npy` | `[num_cases, 12]` | Router、小专家上下文、Stage-1 embedding | 当前窗口最后一个有效 cycle 的 12 维物理 proxy。 |
| `case_operation_seq.npy` | `[num_cases, lookback, 8]` | future operation encoder、Router 统计、metadata numeric matching | 每个 cycle 的工况统计特征。 |
| `case_future_ops.npy` | `[num_cases, horizon, 8]` | future operation encoder、小专家输入 | 未来 horizon 的已知或配置生成工况。 |
| `case_future_ops_mask.npy` | `[num_cases, horizon, 8]` | 缺失处理 | 标记 future operation 是否真实可用。 |
| `case_expert_seq.npy` | `[num_cases, lookback, 14]` | LSTM 小专家输入 | 低维、命名明确的逐 cycle 数值序列。 |
| `case_future_delta_soh.npy` | `[num_cases, horizon]` | 监督训练标签、RAG reference future | `future_soh - anchor_soh`。 |
| `case_future_soh.npy` | `[num_cases, horizon]` | 评估、可视化 | 未来 SOH 真实值。 |

## 状态与标签特征

| feature_name | 中文含义 | 英文含义 | shape | 单位 | 来源 | 用途 | 缺失处理 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `anchor_soh` | 锚点 SOH | anchor state of health | `[]` | ratio | 输入窗口最后一个 cycle 的 SOH | 还原绝对预测值；RAG `d_soh_state`；Router 状态输入；专家上下文 | 必需，缺失时该窗口不能作为监督样本。 |
| `soh_seq` | 历史 SOH 序列 | historical SOH sequence | `[lookback]` | ratio | canonical cycle table 的 `soh` | 序列预测输入；计算 `recent_soh_slope`、`recent_soh_curvature`、`normalized_capacity_delta_seq` | 必需，缺失窗口会被跳过。 |
| `recent_soh_slope` | 近期 SOH 衰减斜率 | recent SOH degradation slope | `[]` | SOH/cycle | `soh_seq` 末端差分均值 | 衡量近期退化速度；RAG `d_soh_state`；Router 输入 | 由 `soh_seq` 计算；长度不足时填 0。 |
| `recent_soh_curvature` | 近期 SOH 退化曲率 | recent SOH degradation curvature | `[]` | SOH/cycle² | `soh_seq` 二阶差分均值 | 判断退化是否加速；RAG `d_soh_state`；Router 输入 | 由 `soh_seq` 计算；长度不足时填 0。 |
| `degradation_stage` | 退化阶段 | degradation stage | `[]` | category | `anchor_soh` 分桶 | RAG 状态距离的阶段软约束；诊断输出 | 缺失记为 `unknown`。 |
| `target_delta_soh` | 未来 SOH 变化量标签 | future delta SOH target | `[horizon]` | SOH difference | `future_soh - anchor_soh` | 监督训练目标；RAG reference future delta | 必需标签，缺失则窗口不可用于监督训练。 |
| `target_soh` | 未来 SOH 真实值 | future SOH target | `[horizon]` | ratio | future cycles | 评估和可视化 | 评估必需。 |

## cycle_stats 序列特征

`cycle_stats` 是每个输入窗口内逐 cycle 的统计序列，当前配置来自 `configs/battery_soh.yaml` 的 `memory.token_features`，其中 `soh` 被单独保存为 `soh_seq`，不会重复放入 `case_cycle_stats.npy`。

| feature_name | 中文含义 | 英文含义 | 单位 | 提取方式 | 用途 | 缺失处理 |
| --- | --- | --- | --- | --- | --- | --- |
| `soh_diff_1` | SOH 一阶差分 | one-cycle SOH difference | SOH/cycle | 当前 cycle SOH 减去前一 cycle SOH | 表示局部退化增量 | 缺失填 0。 |
| `soh_slope_5` | 5-cycle SOH 平均斜率 | 5-cycle SOH rolling slope | SOH/cycle | 最近 5 个 cycle 的 SOH 差分滚动均值 | 表示短期退化速度 | 缺失填 0。 |
| `voltage_mean` | 平均电压 | mean voltage | V | 单 cycle 电压均值 | 基础电压状态输入 | 缺失填 0。 |
| `voltage_range` | 电压范围 | voltage range | V | `voltage_max - voltage_min` | 表示单 cycle 电压摆幅 | 缺失填 0。 |
| `voltage_mean_diff_1` | 平均电压一阶差分 | one-cycle mean-voltage difference | V/cycle | 当前 `voltage_mean` 减去前一 cycle | 表示电压均值漂移 | 缺失填 0。 |
| `voltage_mean_std_5` | 平均电压 5-cycle 滚动标准差 | rolling std of mean voltage | V | 最近 5 个 cycle 的 `voltage_mean` 标准差 | 表示电压状态波动 | 缺失填 0。 |
| `voltage_mean_fft_entropy_16` | 平均电压频谱熵 | FFT entropy of mean voltage | dimensionless | 最近 16 个 cycle 的 `voltage_mean` 去均值后频谱熵 | 表示局部波动复杂度 | 数据不足或全常数时填 0。 |
| `temp_mean` | 平均温度 | mean temperature | °C | 单 cycle 温度均值 | 热环境输入；operation seq；metadata numeric matching | 缺失填 0。 |
| `temp_range` | 温度范围 | temperature range | °C | `temp_max - temp_min` | 表示单 cycle 温度摆幅 | 缺失填 0。 |
| `temp_mean_diff_1` | 平均温度一阶差分 | one-cycle mean-temperature difference | °C/cycle | 当前 `temp_mean` 减去前一 cycle | 表示热环境漂移 | 缺失填 0。 |
| `temp_mean_std_5` | 平均温度 5-cycle 滚动标准差 | rolling std of mean temperature | °C | 最近 5 个 cycle 的 `temp_mean` 标准差 | 表示温度波动 | 缺失填 0。 |
| `temp_mean_fft_entropy_16` | 平均温度频谱熵 | FFT entropy of mean temperature | dimensionless | 最近 16 个 cycle 的 `temp_mean` 频谱熵 | 表示温度变化复杂度 | 数据不足或全常数时填 0。 |
| `current_abs_mean` | 平均电流绝对值 | mean absolute current | A or original current unit | `abs(current_mean)` | 工况强度 proxy；expert seq；operation seq | 缺失填 0。 |
| `current_mean_diff_1` | 平均电流一阶差分 | one-cycle mean-current difference | A/cycle | 当前 `current_mean` 减去前一 cycle | 表示电流策略变化 | 缺失填 0。 |
| `current_mean_std_5` | 平均电流 5-cycle 滚动标准差 | rolling std of mean current | A | 最近 5 个 cycle 的 `current_mean` 标准差 | 表示倍率/电流波动 | 缺失填 0。 |
| `current_mean_fft_entropy_16` | 平均电流频谱熵 | FFT entropy of mean current | dimensionless | 最近 16 个 cycle 的 `current_mean` 频谱熵 | 表示电流策略复杂度 | 数据不足或全常数时填 0。 |
| `cc_time` | 恒流充电时间 | constant-current charge time | time unit from adapter | 单 cycle CC 阶段时长 | 充电协议特征 | 缺失填 0。 |
| `cv_time` | 恒压充电时间 | constant-voltage charge time | time unit from adapter | 单 cycle CV 阶段时长 | 充电协议特征 | 缺失填 0。 |
| `charge_throughput_delta_1` | 充电吞吐量增量 | one-cycle charge-throughput increment | Ah or dataset unit | 当前 `charge_throughput` 减去前一 cycle | 表示充电累计量变化 | 缺失填 0。 |
| `discharge_throughput_delta_1` | 放电吞吐量增量 | one-cycle discharge-throughput increment | Ah or dataset unit | 当前 `discharge_throughput` 减去前一 cycle | 表示放电累计量变化 | 缺失填 0。 |
| `energy_charge_delta_1` | 充电能量增量 | one-cycle charge-energy increment | Wh or dataset unit | 当前 `energy_charge` 减去前一 cycle | 表示充电能量变化 | 缺失填 0。 |
| `energy_discharge_delta_1` | 放电能量增量 | one-cycle discharge-energy increment | Wh or dataset unit | 当前 `energy_discharge` 减去前一 cycle | 表示放电能量变化 | 缺失填 0。 |

## Q-V 曲线特征

Q-V 特征来自每个 raw cycle 的 `voltage/current/capacity/time/step` 信号。框架先对电压、电流、容量做 rolling median，再把充电段和放电段分别插值到归一化容量轴 `Q ∈ [0,1]`。

| feature_name | 中文含义 | 英文含义 | shape | 单位 | 来源 | 用途 | 缺失处理 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Vc(Q)` | Q 轴充电电压曲线 | charge voltage curve indexed by Q | `[q_grid]` | V | charge segment voltage | Q-V map 通道；计算 `DeltaV(Q)` | 充电段不可用时该通道 mask=0。 |
| `Vd(Q)` | Q 轴放电电压曲线 | discharge voltage curve indexed by Q | `[q_grid]` | V | discharge segment voltage | RAG `d_qv_shape` 核心通道；比较放电平台 | 放电段不可用时该通道 mask=0。 |
| `Ic(Q)` | Q 轴充电电流曲线 | charge current curve indexed by Q | `[q_grid]` | A or original current unit | charge segment current | 生成 `charge_current_seq`；默认不直接作为 `d_qv_shape` 通道 | 缺失通道 mask=0。 |
| `Id(Q)` | Q 轴放电电流曲线 | discharge current curve indexed by Q | `[q_grid]` | A or original current unit | discharge segment current | 生成 `discharge_current_seq`；默认不直接作为 `d_qv_shape` 通道 | 缺失通道 mask=0。 |
| `DeltaV(Q)` | Q 轴充放电电压差 | charge-discharge voltage gap curve | `[q_grid]` | V | `Vc(Q)-Vd(Q)` | 极化强度 proxy；RAG `d_qv_shape`；physics features；Router | 需要 Vc/Vd 同时可用，否则 mask=0 或填 0。 |
| `R(Q)` | Q 轴极化/内阻 proxy | Q-indexed polarization or resistance proxy | `[q_grid]` | proxy | `DeltaV(Q)/(Ic(Q)-Id(Q)+eps)` | 极化/内阻 proxy；RAG `d_qv_shape`；physics features；Router | 分母过小时用 eps；极端值按分位数截断；缺失用 mask 标记。 |

当前 `d_qv_shape` 默认使用 `Vd(Q)`、`DeltaV(Q)`、`R(Q)` 和 Q-V summary stats；默认不使用 `Vc(Q)`、`Ic(Q)`、`Id(Q)` 作为直接曲线距离，以降低充电协议差异带来的排序干扰。

## Q-V summary 与物理 proxy 特征

| feature_name | 中文含义 | 英文含义 | 单位 | 来源 | 当前用途 | 缺失处理 |
| --- | --- | --- | --- | --- | --- | --- |
| `delta_v_mean` | 充放电电压差均值 | mean charge-discharge voltage gap | V | `DeltaV(Q)` 均值 | RAG `d_qv_shape`、`d_physics`、Router、小专家输入 | 缺失填 0，mask=0。 |
| `delta_v_std` | 充放电电压差标准差 | std of charge-discharge voltage gap | V | `DeltaV(Q)` 标准差 | 表示极化沿容量轴的变化程度 | 缺失填 0，mask=0。 |
| `delta_v_q95` | 充放电电压差 95 分位数 | 95th percentile of charge-discharge voltage gap | V | `DeltaV(Q)` 高分位 | 捕捉局部强极化 | 若没有 q95 可回退到 `delta_v_max`。 |
| `r_mean` | R proxy 均值 | mean resistance proxy | proxy | `R(Q)` 均值 | RAG `d_qv_shape`、`d_physics`、Router、小专家输入 | 缺失填 0，mask=0。 |
| `r_std` | R proxy 标准差 | std of resistance proxy | proxy | `R(Q)` 标准差 | 表示 R(Q) 波动 | 缺失填 0，mask=0。 |
| `r_q95` | R proxy 95 分位数 | 95th percentile of resistance proxy | proxy | `R(Q)` 高分位 | 捕捉局部高极化/高阻抗 proxy | 缺失填 0，mask=0。 |
| `vc_curve_slope_mean` | 充电电压曲线平均斜率 | mean slope of charge voltage curve | V/Q | `Vc(Q)` 梯度均值 | 表征充电平台形态；RAG summary；Router | 缺失填 0，mask=0。 |
| `vd_curve_slope_mean` | 放电电压曲线平均斜率 | mean slope of discharge voltage curve | V/Q | `Vd(Q)` 梯度均值 | 表征放电平台形态；RAG summary；Router | 缺失填 0，mask=0。 |
| `ic_mean` | 充电电流均值 | mean charge current | A or original current unit | `Ic(Q)` 均值 | case bank 统计与诊断 | 缺失填 0。 |
| `id_mean` | 放电电流均值 | mean discharge current | A or original current unit | `Id(Q)` 均值 | case bank 统计与诊断 | 缺失填 0。 |
| `v_charge_mean` | 充电电压均值 | mean charge voltage | V | `Vc(Q)` 均值 | case bank 统计与诊断 | 缺失填 0。 |
| `v_discharge_mean` | 放电电压均值 | mean discharge voltage | V | `Vd(Q)` 均值 | case bank 统计与诊断 | 缺失填 0。 |

## partial charging 与 12 维 physics features

`partial_charge_curve` 只使用充电段，计算电压网格上的累计输入电荷 `q(V_i)=∫|I(t)|dt`。如果原始数据无法稳定识别充电段或电压区间太窄，曲线填 0，并通过 `partial_charge_mask=0` 标记，不丢弃样本。

`physics_features` 当前为 12 维：

| index | feature_name | 中文含义 | 英文含义 | 单位 | 来源 | 用途 | 缺失处理 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `delta_v_mean` | 充放电电压差均值 | mean charge-discharge voltage gap | V | Q-V summary | `d_physics`、Router、专家上下文 | Q-V 不可用时 mask=0。 |
| 1 | `delta_v_std` | 充放电电压差标准差 | std of charge-discharge voltage gap | V | Q-V summary | `d_physics`、Router、专家上下文 | Q-V 不可用时 mask=0。 |
| 2 | `delta_v_q95` | 充放电电压差高分位 | high-percentile voltage gap | V | Q-V summary | 捕捉局部强极化 | Q-V 不可用时 mask=0。 |
| 3 | `r_mean` | R proxy 均值 | mean resistance proxy | proxy | Q-V summary | `d_physics`、Router、专家上下文 | Q-V 不可用时 mask=0。 |
| 4 | `r_std` | R proxy 标准差 | std of resistance proxy | proxy | Q-V summary | 表示 R(Q) 波动 | Q-V 不可用时 mask=0。 |
| 5 | `r_q95` | R proxy 高分位 | high-percentile resistance proxy | proxy | Q-V summary | 捕捉局部高极化 | Q-V 不可用时 mask=0。 |
| 6 | `vc_curve_slope_mean` | 充电电压曲线平均斜率 | mean slope of charge voltage curve | V/Q | Q-V summary | 表征充电平台形状 | Q-V 不可用时 mask=0。 |
| 7 | `vd_curve_slope_mean` | 放电电压曲线平均斜率 | mean slope of discharge voltage curve | V/Q | Q-V summary | 表征放电平台形状 | Q-V 不可用时 mask=0。 |
| 8 | `q_total` | partial charging 累计电荷总量 | total cumulative charge in partial charging | Ah or proxy | partial charge curve 末值 | 表示选定充电区间容量接受能力 | partial charge 不可用时填 0，mask=0。 |
| 9 | `q_mean` | partial charging 曲线均值 | mean partial-charge curve value | Ah or proxy | partial charge curve 均值 | 表示曲线整体水平 | partial charge 不可用时填 0，mask=0。 |
| 10 | `q_std` | partial charging 曲线标准差 | std of partial-charge curve | Ah or proxy | partial charge curve 标准差 | 表示曲线变化幅度 | partial charge 不可用时填 0，mask=0。 |
| 11 | `dq_dv_peak_value` | dq/dv 峰值大小 | peak value of dq/dv curve | proxy | partial charge curve 梯度最大值 | 表示充电曲线局部平台变化 | partial charge 不可用时填 0，mask=0。 |

## operation 与 future operation 特征

当前 `case_operation_seq.npy` 和 `case_future_ops.npy` 都使用同一组 8 维特征。注意：这些特征用于预测模型输入、Router 统计和 future operation encoder；当前主线 RAG 不再单独启用 `d_operation`，而是把电流、温度、容量变化的原始滑窗序列并入 `d_metadata`。

| feature_name | 中文含义 | 英文含义 | 单位 | 来源 | 用途 | 缺失处理 |
| --- | --- | --- | --- | --- | --- | --- |
| `current_abs_mean` | 平均电流绝对值 | mean absolute current | A or original current unit | `abs(current_mean)` | 工况强度；operation seq；expert seq | 缺失填 0。 |
| `temp_mean` | 平均温度 | mean temperature | °C | canonical `temp_mean` | 热环境；metadata `temperature_seq` | 缺失填 0。 |
| `cc_time` | 恒流充电时间 | constant-current time | time unit from adapter | adapter cycle stats | 充电协议上下文 | 缺失填 0。 |
| `cv_time` | 恒压充电时间 | constant-voltage time | time unit from adapter | adapter cycle stats | 充电协议上下文 | 缺失填 0。 |
| `charge_throughput_delta_1` | 充电吞吐量增量 | charge-throughput increment | Ah or dataset unit | `charge_throughput.diff()` | 历史充电吞吐变化 | 缺失填 0。 |
| `discharge_throughput_delta_1` | 放电吞吐量增量 | discharge-throughput increment | Ah or dataset unit | `discharge_throughput.diff()` | 历史放电吞吐变化 | 缺失填 0。 |
| `energy_charge_delta_1` | 充电能量增量 | charge-energy increment | Wh or dataset unit | `energy_charge.diff()` | 充电能量变化 | 缺失填 0。 |
| `energy_discharge_delta_1` | 放电能量增量 | discharge-energy increment | Wh or dataset unit | `energy_discharge.diff()` | 放电能量变化 | 缺失填 0。 |

## RAG 检索特征

RAG reference database 默认只允许 `source_train`，`target_query` 不得进入 reference database。Stage-1 粗检索使用 `handcrafted_retrieval_embedding`，不是 TSFM embedding。

### Stage-1 handcrafted retrieval embedding

| 组成 | 具体特征 | 说明 |
| --- | --- | --- |
| SOH 状态 | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature` | 用于快速召回健康状态和退化趋势相近的窗口。 |
| Q-V summary | `delta_v_mean`, `delta_v_std`, `delta_v_q95`, `r_mean`, `r_std`, `r_q95`, `vc_curve_slope_mean`, `vd_curve_slope_mean` | 用低维曲线统计近似容量-电压曲线形态。 |
| physics anchor | `case_anchor_physics_features[0:12]` | 当前窗口锚点 cycle 的 12 维物理 proxy。 |
| metadata numeric summary | `charge_current_seq`, `discharge_current_seq`, `temperature_seq`, `normalized_capacity_delta_seq` 的均值、标准差、末值 | 用原始滑窗数值概括工况和容量变化轨迹。 |

### Stage-2 named distances

| distance_name | 默认启用 | 中文含义 | 具体使用特征 | 数值含义 |
| --- | --- | --- | --- | --- |
| `d_soh_state` | true | SOH 状态距离 | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature`, `degradation_stage` | 越小表示当前健康状态、近期退化速度和退化阶段越相似。 |
| `d_qv_shape` | true | 容量-电压曲线形状距离 | `Vd(Q)`, `DeltaV(Q)`, `R(Q)`, Q-V summary stats | 越小表示放电平台、极化曲线和曲线统计越相似。 |
| `d_physics` | true | 物理 proxy 距离 | 12 维 `physics_features`，当前主要是 `DeltaV/R` 统计和 partial charge summary | 越小表示极化/内阻 proxy 与 partial charge 状态越相似。 |
| `d_metadata` | true | 元信息与原始滑窗数值距离 | categorical: `chemistry_family`, `domain_label`, `voltage_window_bucket`; numeric: `charge_current_seq`, `discharge_current_seq`, `temperature_seq`, `normalized_capacity_delta_seq` | 越小表示材料体系、工况域、电压窗口和原始运行轨迹越可比。 |
| `d_operation` | false | 兼容旧输出的工况距离 | 当前不作为独立检索分量 | 默认返回 NaN，不参与 `composite_distance`。 |
| `composite_distance` | true | 综合检索距离 | YAML 中启用距离的归一化加权和 | 越小表示 reference 越适合作为历史参考案例。 |
| `retrieval_confidence` | true | 检索置信度 | top-k 综合距离均值/离散度、top1 距离、特征可用率、chemistry match rate | `[0,1]`，越大表示 top-k 检索结果越可靠。 |

当前 `configs/retrieval_features.yaml` 权重为：

| component | weight |
| --- | --- |
| `d_soh_state` | 0.25 |
| `d_qv_shape` | 0.30 |
| `d_physics` | 0.20 |
| `d_metadata` | 0.25 |
| `d_operation` | 0.00 |

### d_metadata 内部细项

| feature_name | 中文含义 | 英文含义 | 类型 | 来源 | d_metadata 中的作用 |
| --- | --- | --- | --- | --- | --- |
| `chemistry_family` | 电池化学体系 | battery chemistry family | categorical | adapter metadata | mismatch penalty，优先材料体系相同或相近的案例。 |
| `domain_label` | 工况域标签 | domain label | categorical | chemistry + operation metadata rule | mismatch penalty，软约束工况域相近。 |
| `voltage_window_bucket` | 电压窗口类别 | voltage window bucket | categorical | voltage min/max bucket | mismatch penalty，避免比较不可比电压窗口。 |
| `charge_current_seq` | 充电电流滑窗序列 | charge-current lookback sequence | numeric sequence | `Ic(Q)` 每 cycle 均值绝对值 | 比较历史窗口内充电电流水平和变化。 |
| `discharge_current_seq` | 放电电流滑窗序列 | discharge-current lookback sequence | numeric sequence | `Id(Q)` 每 cycle 均值绝对值 | 比较历史窗口内放电电流水平和变化。 |
| `temperature_seq` | 温度滑窗序列 | temperature lookback sequence | numeric sequence | `operation_seq.temp_mean` | 比较历史窗口内温度轨迹。 |
| `normalized_capacity_delta_seq` | 归一化容量变化滑窗序列 | normalized-capacity-change lookback sequence | numeric sequence | `soh_seq` 的 cycle-to-cycle diff | 比较历史窗口内容量/SOH 变化轨迹。 |

`source_dataset`、`nominal_capacity_bucket`、`temperature_bucket`、`charge_rate_bucket` 当前保留在 metadata 或 case rows 中用于诊断和扩展，但不参与当前主线 `d_metadata` 距离计算。

## Router 输入特征

`PhysicalDegradationRouter` 是 grouped additive router。它不使用黑箱 TSFM embedding，而是按语义分组输入，每组对每个专家 logit 有可导出的 contribution。

| router group | 中文含义 | 具体输入 | 作用 |
| --- | --- | --- | --- |
| `soh_state` | SOH 状态组 | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature` | 判断当前窗口处于什么健康状态、退化速度是否快、是否存在加速退化。 |
| `qv_polarization` | Q-V 极化组 | `anchor_physics_features` 的前 8 维：`delta_v_*`, `r_*`, `vc/vd_curve_slope_mean` | 让 router 根据极化、内阻 proxy 和平台斜率选择曲线/极化相关专家。 |
| `operation` | 工况组 | operation summary，当前包括 `current_abs_mean/temp_mean/cc_time/cv_time/throughput/energy` 相关统计和 `protocol_change_rate` proxy | 表示运行压力和协议变化强度。 |
| `chemistry` | 化学体系组 | `chemistry_family` 的 metadata embedding | 区分 LFP/NCM/NCA 等材料体系专家。 |
| `retrieval` | 检索可靠性组 | `retrieval_confidence`, top-k `composite_distance` 均值和标准差、top-1 权重、reference compatibility 均值 | 检索越可靠时，router 可以更相信 reference-conditioned 信息。 |
| `neighbor_vote` | 邻居物理模式投票组 | top-k neighbor 的 chemistry/stage/模式统计 | 把历史参考案例的模式分布传给 router。 |

Router 输出：

| output | shape | 含义 |
| --- | --- | --- |
| `expert_logits` | `[batch, num_experts]` | 每个专家的未归一化选择分数。 |
| `expert_weights` | `[batch, num_experts]` | Top-k softmax 后的专家权重。 |
| `expert_router_contributions` | `dict[group, [batch, num_experts]]` | 每个语义组对专家 logits 的贡献，供解释使用。 |

## LSTM 小专家输入特征

当前专家都是 residual LSTM experts。专家不输出完整 SOH 预测，而输出 `moe_residual`，最终 `pred_delta = base_delta + moe_residual`。

`case_expert_seq.npy` 的 14 维逐 cycle 特征如下：

| feature_name | 中文含义 | 英文含义 | 单位 | 来源 | 用途 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| `soh_t` | 当前 cycle SOH | SOH at cycle t | ratio | `feature_frame.soh` | 专家序列核心状态 | 表示该 cycle 的健康状态。 |
| `voltage_mean_t` | 当前 cycle 平均电压 | mean voltage at cycle t | V | `voltage_mean` | 专家序列输入 | 表示电压工作点。 |
| `current_abs_mean_t` | 当前 cycle 平均电流绝对值 | mean absolute current at cycle t | A or original unit | `current_abs_mean` | 专家序列输入 | 表示电流强度。 |
| `temp_mean_t` | 当前 cycle 平均温度 | mean temperature at cycle t | °C | `temp_mean` | 专家序列输入 | 表示热环境。 |
| `delta_v_mean_t` | 当前 cycle ΔV 均值 | mean voltage gap at cycle t | V | Q-V summary | 专家序列输入 | 表示平均极化 proxy。 |
| `delta_v_std_t` | 当前 cycle ΔV 标准差 | std voltage gap at cycle t | V | Q-V summary | 专家序列输入 | 表示极化沿容量轴的变化。 |
| `r_mean_t` | 当前 cycle R proxy 均值 | mean resistance proxy at cycle t | proxy | Q-V summary | 专家序列输入 | 表示平均内阻/极化 proxy。 |
| `r_std_t` | 当前 cycle R proxy 标准差 | std resistance proxy at cycle t | proxy | Q-V summary | 专家序列输入 | 表示 R(Q) 波动。 |
| `delta_v_q95_t` | 当前 cycle ΔV 高分位 | high-percentile voltage gap at cycle t | V | Q-V summary | 专家序列输入 | 捕捉局部强极化。 |
| `r_q95_t` | 当前 cycle R proxy 高分位 | high-percentile resistance proxy at cycle t | proxy | Q-V summary | 专家序列输入 | 捕捉局部高阻抗 proxy。 |
| `charge_c_rate_mean_t` | 充电倍率强度 proxy | charge C-rate proxy at cycle t | 1/h proxy | `current_abs_mean / capacity` | 专家序列输入 | 当前实现由平均电流绝对值除以容量得到，是倍率强度 proxy，不是严格分段充电倍率。 |
| `discharge_c_rate_mean_t` | 放电倍率强度 proxy | discharge C-rate proxy at cycle t | 1/h proxy | `current_abs_mean / capacity` | 专家序列输入 | 当前实现与上一项同源，是倍率强度 proxy，不是严格分段放电倍率。 |
| `temperature_max_t` | 当前 cycle 最高温度 | maximum temperature at cycle t | °C | `temp_max`，缺失时回退 `temp_mean` | 专家序列输入 | 捕捉高温暴露。 |
| `protocol_change_rate_t` | 协议切换强度 proxy | protocol-change-rate proxy at cycle t | ratio/proxy | `abs(diff(charge_c_rate_proxy))` | 专家序列输入 | 表示相邻 cycle 工况强度变化。 |

小专家上下文 `expert_context` 还包含：

| context part | 具体特征 | 说明 |
| --- | --- | --- |
| state context | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature` | 当前窗口健康状态。 |
| physics context | `anchor_physics_features[0:12]` | 当前锚点 cycle 的物理 proxy。 |
| chemistry context | `chemistry_family` embedding | 材料体系上下文。 |
| retrieval context | `retrieval_confidence`, top-k distance mean/std, feature availability, compatibility mean | 历史参考案例可靠性。 |
| baseline summary | `base_delta_mean`, `base_delta_std`, `base_delta_last` | 基础预测形状概括，传入专家时使用 detach，避免专家反向改变 base model。 |
| future operation summary | `future_ops.mean(axis=horizon)` | 如果配置提供未来工况，则作为可选上下文。 |

## 模型分支输出

| output_name | 中文含义 | shape | 计算方式 | 说明 |
| --- | --- | --- | --- | --- |
| `fm_delta` | 基础序列预测分支输出 | `[batch, horizon]` | cycle/SOH/meta 编码后由 generalist head 输出 | 当前不使用 TSFM embedding；这是普通数值序列 generalist。 |
| `rag_delta` | RAG 加权先验 | `[batch, horizon]` | top-k `neighbor_future_delta_soh` 按 `retrieval_alpha` 加权平均 | 表示历史相似案例未来 delta SOH 的直接参考。 |
| `pair_delta` | reference-conditioned 差分分支输出 | `[batch, horizon]` | `ref_future_delta_soh + pair_residual` 后按 `retrieval_alpha` 加权 | 学习 query 与 reference 的轨迹差异。 |
| `base_delta` | 基础 delta SOH 预测 | `[batch, horizon]` | `fusion_base(fm_delta, rag_delta, pair_delta)` | residual experts 修正前的基础预测。 |
| `moe_residual` | 专家残差修正 | `[batch, horizon]` | LSTM experts 输出按 `expert_weights` 加权 | 专家只学习 residual correction。 |
| `pred_delta` | 最终 delta SOH 预测 | `[batch, horizon]` | `base_delta + moe_residual` | 最终训练和评估的 delta SOH 输出。 |
| `pred_soh` | 最终 SOH 预测 | `[batch, horizon]` | `anchor_soh + pred_delta` | 绝对 SOH 预测。 |

## 当前未作为主线特征使用的项

| name | 当前状态 | 原因 |
| --- | --- | --- |
| `d_operation` | 保留兼容输出，默认关闭，不参与 `composite_distance` | 充放电电流、温度、容量变化已经以原始滑窗数值形式并入 `d_metadata`，避免独立工况均值/标准差损失时序信息。 |
| `d_tsfm` / TSFM embedding | 当前主线不使用 | 框架已切换为数值统计和物理启发特征，不让基础时序 embedding 决定 RAG 候选。 |
| relaxation 特征 | 当前主线不使用 | 当前公开数据集难以稳定提取 charge-end rest segment；相关特征不进入 RAG、Router 或专家。 |
