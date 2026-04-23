# Features

下表由 `battery_data.feature_registry` 生成，统一说明 RAG 检索特征、预测输入特征和 Router 输入特征的语义。

| feature_name | 中文含义 | 英文含义 | group | role | shape | unit | source_signal | extraction_method | missing_handling | default_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| charge_rate_bucket | 充电倍率分桶 | charge-rate bucket | metadata | retrieval, diagnostics | [] | categorical | charge-rate metadata | bucket the charge rate category | 缺失时记为 unknown category。 | false |
| chemistry_family | 电池化学体系 | battery chemistry family | metadata | retrieval, router, diagnostics | [] | categorical | dataset metadata | read the chemistry family label from metadata | 缺失时记为 unknown category。 | true |
| domain_label | 工况域标签 | domain label | metadata | retrieval, router, diagnostics | [] | categorical | domain labeling rules | combine chemistry, rate, temperature and protocol metadata into a domain label | 缺失时记为 unknown category。 | true |
| nominal_capacity_bucket | 额定容量分桶 | nominal capacity bucket | metadata | retrieval, diagnostics | [] | categorical | nominal capacity estimate | bucket the estimated nominal capacity | 缺失时记为 unknown category。 | false |
| source_dataset | 数据集来源 | source dataset | metadata | retrieval, diagnostics | [] | categorical | dataset metadata | read dataset source label | 缺失时记为 unknown category。 | false |
| temperature_bucket | 温度分桶 | temperature bucket | metadata | retrieval, diagnostics | [] | categorical | temperature metadata | bucket the environmental or test temperature | 缺失时记为 unknown category。 | false |
| voltage_window_bucket | 电压窗口类别 | voltage window bucket | metadata | retrieval, diagnostics | [] | categorical | canonical cycle metadata | bucket the lower and upper voltage bounds | 缺失时记为 unknown category。 | true |
| charge_c_rate_mean | 充电倍率均值 | mean charge C-rate | operation | retrieval, router | [] | C-rate | cycle-level current summary and nominal capacity | average estimated charge C-rate over the lookback window | 缺失时填 0，并在 d_operation 中通过可用特征 mask 处理。 | true |
| charge_c_rate_std | 充电倍率标准差 | standard deviation of charge C-rate | operation | retrieval, router | [] | C-rate | cycle-level current summary and nominal capacity | standard deviation of estimated charge C-rate over the lookback window | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| charge_duration_mean | 平均充电时长 | mean charge duration | operation | retrieval, router | [] | minute or second | cycle-level charge duration | average charge duration over the lookback window | 很多数据集记录不一致，默认关闭；缺失时填 0。 | false |
| discharge_c_rate_mean | 放电倍率均值 | mean discharge C-rate | operation | retrieval, router | [] | C-rate | cycle-level current summary and nominal capacity | average estimated discharge C-rate over the lookback window | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| discharge_c_rate_std | 放电倍率标准差 | standard deviation of discharge C-rate | operation | retrieval, router | [] | C-rate | cycle-level current summary and nominal capacity | standard deviation of estimated discharge C-rate over the lookback window | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| discharge_duration_mean | 平均放电时长 | mean discharge duration | operation | retrieval, router | [] | minute or second | cycle-level discharge duration | average discharge duration over the lookback window | 很多数据集记录不一致，默认关闭；缺失时填 0。 | false |
| dod_estimate | 估计放电深度 | estimated depth of discharge | operation | retrieval, router | [] | ratio | capacity swing or SOC span | estimate DoD from capacity swing or SOC range | 很多数据集不稳定，默认可关闭；缺失时填 0。 | false |
| protocol_change_rate | 协议切换频率 | protocol change rate | operation | retrieval, router | [] | ratio | window-level operation sequence | count significant changes in current or temperature regime across cycles | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| rest_duration_mean | 平均静置时长 | mean rest duration | operation | retrieval, router | [] | minute or second | cycle-level rest duration | average rest duration over the lookback window | 很多数据集不稳定，默认关闭；缺失时填 0。 | false |
| temperature_max | 最高温度 | maximum temperature | operation | retrieval, router | [] | Celsius | cycle-level temperature summary | maximum temperature observed over the lookback window | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| temperature_mean | 温度均值 | mean temperature | operation | retrieval, router | [] | Celsius | cycle-level temperature summary | average cycle temperature over the lookback window | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| temperature_std | 温度标准差 | standard deviation of temperature | operation | retrieval, router | [] | Celsius | cycle-level temperature summary | standard deviation of cycle temperature over the lookback window | 缺失时填 0，并在 d_operation 中通过 mask 处理。 | true |
| dq_dv_peak_position | dq/dv 峰值位置 | voltage position of dq/dv peak | physics_12d | prediction_input, retrieval, router | [] | V or normalized index | partial_charge_curve | locate the peak index or voltage of the approximate dq/dv curve | partial charging 不可用时填 0 并由 mask 标记。 | true |
| dq_dv_peak_value | dq/dv 峰值大小 | peak value of dq/dv curve | physics_12d | prediction_input, retrieval, router | [] | charge per volt | partial_charge_curve | approximate dq/dv and take the peak magnitude | partial charging 不可用时填 0 并由 mask 标记。 | true |
| q_high_voltage_slope | 高电压区 partial charging 斜率 | high-voltage slope of partial charging curve | physics_12d | prediction_input, retrieval, router | [] | charge per volt | partial_charge_curve | fit a slope over the high-voltage segment | partial charging 不可用时填 0 并由 mask 标记。 | true |
| q_low_voltage_slope | 低电压区 partial charging 斜率 | low-voltage slope of partial charging curve | physics_12d | prediction_input, retrieval, router | [] | charge per volt | partial_charge_curve | fit a slope over the low-voltage segment | partial charging 不可用时填 0 并由 mask 标记。 | true |
| q_mean | partial charging 曲线均值 | mean of partial charging curve | physics_12d | prediction_input, retrieval, router | [] | Ah or normalized charge | partial_charge_curve | mean over the voltage grid | partial charging 不可用时填 0 并由 mask 标记。 | true |
| q_mid_voltage_slope | 中电压区 partial charging 斜率 | mid-voltage slope of partial charging curve | physics_12d | prediction_input, retrieval, router | [] | charge per volt | partial_charge_curve | fit a slope over the mid-voltage segment | partial charging 不可用时填 0 并由 mask 标记。 | true |
| q_std | partial charging 曲线标准差 | standard deviation of partial charging curve | physics_12d | prediction_input, retrieval, router | [] | Ah or normalized charge | partial_charge_curve | standard deviation over the voltage grid | partial charging 不可用时填 0 并由 mask 标记。 | true |
| q_total | partial charging 累计电荷总量 | total cumulative charge in partial charging curve | physics_12d | prediction_input, retrieval, router | [] | Ah or normalized charge | partial_charge_curve | sum or terminal value of the cumulative partial charging curve | partial charging 不可用时填 0 并由 physics_feature_mask 标记。 | true |
| relaxation_area_or_tau_proxy | 弛豫面积或时间常数 proxy | relaxation area or time-constant proxy | physics_12d | prediction_input, retrieval, router | [] | V·minute or proxy | relaxation_curve | integrate the relaxation curve or estimate a tau-like proxy | 弛豫不可用时填 0 并由 mask 标记。 | true |
| relaxation_delta_v | 弛豫电压变化量 | voltage change during relaxation | physics_12d | prediction_input, retrieval, router | [] | V | relaxation_curve | difference between the start and end of the relaxation voltage curve | 弛豫不可用时填 0 并由 mask 标记。 | true |
| relaxation_initial_slope | 弛豫初期斜率 | initial slope of relaxation voltage | physics_12d | prediction_input, retrieval, router | [] | V per minute | relaxation_curve | fit a short-horizon slope near the start of relaxation | 弛豫不可用时填 0 并由 mask 标记。 | true |
| relaxation_late_slope | 弛豫后期斜率 | late slope of relaxation voltage | physics_12d | prediction_input, retrieval, router | [] | V per minute | relaxation_curve | fit a slope over the late relaxation segment | 弛豫不可用时填 0 并由 mask 标记。 | true |
| partial_charge_curve | 局部充电曲线 | partial charging curve | physics_curve | prediction_input, visualization | [partial_charge_points] | Ah or normalized charge | charge segment in raw cycle table | integrate /I(t)/ over a selected voltage window | 缺失时曲线填 0，并通过 partial_charge_mask 标记。 | true |
| partial_charge_mask | 局部充电曲线可用标记 | partial charging availability mask | physics_curve | mask, diagnostics | [] | bool | partial charge extraction status | mark whether the partial charging curve is valid | 不可用时为 0，不丢弃样本。 | true |
| relaxation_curve | 弛豫电压曲线 | relaxation voltage curve | physics_curve | prediction_input, visualization | [relax_points] | V | rest segment after charge end | sample relaxation voltage after the charge-rest transition | 缺失时曲线填 0，并通过 relaxation_mask 标记。 | true |
| relaxation_mask | 弛豫曲线可用标记 | relaxation availability mask | physics_curve | mask, diagnostics | [] | bool | relaxation extraction status | mark whether the relaxation curve is valid | 不可用时为 0，不丢弃样本。 | true |
| delta_v_curve_q | Q 轴充放电电压差曲线 | charge-discharge voltage gap curve | qv_shape | retrieval, router, visualization | [q_grid_size] | V | Vc(Q) and Vd(Q) | DeltaV(Q) = Vc(Q) - Vd(Q) | 由 qv_masks 标记可用性。 | true |
| qv_maps | Q 轴对齐的容量-电压多通道曲线 | Q-indexed voltage-current feature maps | qv_shape | prediction_input, retrieval, visualization | [lookback, 6, q_grid_size] | mixed; voltage/current/proxy by channel | raw cycle voltage, current, capacity curves | interpolate charge and discharge curves onto normalized capacity Q in [0, 1] | 使用 qv_masks 标记可用通道，不丢弃样本。 | true |
| r_curve_q | Q 轴近似极化/阻抗曲线 | Q-indexed polarization or resistance proxy | qv_shape | retrieval, router, visualization | [q_grid_size] | proxy ohm | DeltaV(Q), Ic(Q), Id(Q) | R(Q)=DeltaV(Q)/(Ic(Q)-Id(Q)+eps) with clipping | 分母过小时使用 eps；极端值 clip；由 qv_masks 标记可用性。 | true |
| vc_curve_q | Q 轴充电电压曲线 | charge voltage curve indexed by normalized capacity | qv_shape | retrieval, visualization | [q_grid_size] | V | qv_maps channel Vc(Q) | select the charge-voltage channel from qv_maps | 由 qv_masks 标记可用性。 | false |
| vd_curve_q | Q 轴放电电压曲线 | discharge voltage curve indexed by normalized capacity | qv_shape | retrieval, visualization | [q_grid_size] | V | qv_maps channel Vd(Q) | select the discharge-voltage channel from qv_maps | 由 qv_masks 标记可用性。 | true |
| delta_v_mean | 充放电电压差均值 | mean charge-discharge voltage gap | qv_summary | retrieval, router | [] | V | delta_v_curve_q | mean over the normalized-capacity axis | 若 qv 曲线不可用则填 0 并由 mask 标记。 | true |
| delta_v_q95 | 充放电电压差 95 分位数 | 95th percentile of charge-discharge voltage gap | qv_summary | retrieval, router | [] | V | delta_v_curve_q | 95th percentile over the normalized-capacity axis | 若 qv 曲线不可用则填 0 并由 mask 标记。 | true |
| delta_v_std | 充放电电压差标准差 | standard deviation of charge-discharge voltage gap | qv_summary | retrieval, router | [] | V | delta_v_curve_q | standard deviation over the normalized-capacity axis | 若 qv 曲线不可用则填 0 并由 mask 标记。 | true |
| r_mean | 近似极化/阻抗均值 | mean resistance proxy | qv_summary | retrieval, router | [] | proxy ohm | r_curve_q | mean over the normalized-capacity axis | 若 r_curve_q 不可用则填 0 并由 mask 标记。 | true |
| r_q95 | 近似极化/阻抗 95 分位数 | 95th percentile of resistance proxy | qv_summary | retrieval, router | [] | proxy ohm | r_curve_q | 95th percentile over the normalized-capacity axis | 若 r_curve_q 不可用则填 0 并由 mask 标记。 | true |
| r_std | 近似极化/阻抗标准差 | standard deviation of resistance proxy | qv_summary | retrieval, router | [] | proxy ohm | r_curve_q | standard deviation over the normalized-capacity axis | 若 r_curve_q 不可用则填 0 并由 mask 标记。 | true |
| vc_curve_slope_mean | 充电电压曲线平均斜率 | mean slope of charge voltage curve | qv_summary | retrieval, router | [] | V per normalized capacity | vc_curve_q | finite-difference slope averaged over Q | 若 vc_curve_q 不可用则填 0 并由 mask 标记。 | false |
| vd_curve_slope_mean | 放电电压曲线平均斜率 | mean slope of discharge voltage curve | qv_summary | retrieval, router | [] | V per normalized capacity | vd_curve_q | finite-difference slope averaged over Q | 若 vd_curve_q 不可用则填 0 并由 mask 标记。 | true |
| composite_distance | 综合检索距离 | composite retrieval distance | retrieval | retrieval_output, diagnostics | [] | distance | enabled core distance components | weighted sum of enabled distance components in the retrieval YAML | 只由启用的距离分量组成；若某分量关闭则不会进入综合距离。 | true |
| d_metadata | 元信息距离 | metadata distance | retrieval | retrieval_output, diagnostics | [] | distance or categorical penalty | chemistry, domain, voltage-window and dataset metadata | apply categorical mismatch penalties from the retrieval YAML | 缺失时按 unknown category 处理。 | true |
| d_operation | 工况距离 | operation condition distance | retrieval | retrieval_output, diagnostics | [] | distance | operation summary features | compute a normalized distance over enabled operation features | 缺失时只比较可用特征并记录可用率。 | true |
| d_physics | 物理启发特征距离 | physics-inspired feature distance | retrieval | retrieval_output, diagnostics | [] | distance | 12-D partial-charge and relaxation features | compute a mask-aware normalized distance over physics features | 缺失时按 physics_feature_mask 只比较共同可用维度。 | true |
| d_qv_shape | 容量-电压曲线形状距离 | Q-V curve shape distance | retrieval | retrieval_output, diagnostics | [] | distance | Vd(Q), DeltaV(Q), R(Q), optional summary stats | compare enabled Q-indexed curve channels and summary statistics | 曲线缺失时根据 qv_masks 只比较双方共同可用通道。 | true |
| d_soh_state | SOH 状态距离 | SOH state distance | retrieval | retrieval_output, diagnostics | [] | distance | anchor_soh, recent_soh_slope, recent_soh_curvature, degradation_stage | compute a normalized distance over state features | 如果状态特征缺失则按照配置中的 missing policy 处理。 | true |
| d_tsfm | 时间序列基础模型嵌入距离 | TSFM embedding distance | retrieval | retrieval_output, diagnostics, coarse_retrieval | [] | distance | tsfm_embedding | compute cosine or L2 distance in the TSFM embedding space | embedding 缺失时回退到 handcrafted embedding；不应主导最终检索。 | true |
| retrieval_confidence | 检索置信度 | retrieval confidence | retrieval | retrieval_output, router, diagnostics | [] | score in [0, 1] | top-k composite distances, availability ratio and metadata match statistics | aggregate the enabled confidence factors defined in the retrieval YAML | 若没有有效邻居则为 0。 | true |
| partial_charge_availability_ratio | 局部充电曲线可用率 | partial charging availability ratio | router | router | [] | ratio | partial_charge_mask over the lookback window | mean mask value over the lookback window | 无可用特征时为 0。 | true |
| physics_availability_ratio | 物理特征可用率 | physics feature availability ratio | router | router | [] | ratio | physics_feature_masks over the lookback window | mean mask value over physics features and time | 无可用特征时为 0。 | true |
| relaxation_availability_ratio | 弛豫曲线可用率 | relaxation availability ratio | router | router | [] | ratio | relaxation_mask over the lookback window | mean mask value over the lookback window | 无可用特征时为 0。 | true |
| anchor_soh | 锚点 SOH | anchor state of health | state | prediction_input, retrieval, router | [] | ratio | window-level SOH summary | read the last SOH value in the lookback window | 必需特征；缺失时该 case 不可用于监督预测和检索。 | true |
| degradation_stage | 退化阶段 | degradation stage | state | retrieval, router, diagnostics | [] | categorical | anchor_soh and degradation heuristics | bucket anchor_soh or degradation indicators into stage labels | 缺失时记为 Unknown。 | true |
| recent_soh_curvature | 近期 SOH 退化曲率 | recent SOH degradation curvature | state | retrieval, router, diagnostics | [] | SOH per cycle squared | soh_seq | second-order trend or second finite difference over recent SOH points | 由 soh_seq 计算；lookback 太短时填 0 并记录 warning。 | true |
| recent_soh_slope | 近期 SOH 衰减斜率 | recent SOH degradation slope | state | retrieval, router, diagnostics | [] | SOH per cycle | soh_seq | linear fit or mean first difference over recent SOH points | 由 soh_seq 计算；若 lookback 太短则退化为最近差分。 | true |
| soh_seq | 历史 SOH 序列 | historical SOH sequence | state | prediction_input, tsfm_source | [lookback] | ratio | cycle-level SOH sequence | collect the SOH values over the lookback window | 必需特征；缺失时该 case 不可用于监督预测。 | true |
| target_delta_soh | 未来 SOH 变化量标签 | future delta SOH target | target | supervision | [horizon] | SOH difference | future_soh and anchor_soh | target_delta_soh[h] = future_soh[h] - anchor_soh | 必需标签；缺失则该 case 不可用于监督训练。 | true |
| target_soh | 未来 SOH 真实值 | future SOH target | target | evaluation, visualization | [horizon] | ratio | future SOH trajectory | read the ground-truth future SOH values | 评估必需。 | true |
| tsfm_embedding | 时间序列基础模型嵌入 | time-series foundation model embedding | tsfm | retrieval, coarse_retrieval, prediction_input_optional | [embedding_dim] | embedding | TSFM encoder over historical battery trajectories | encode the historical window into a fixed-dimensional time-series embedding | 缺失时可回退到 handcrafted embedding；不应主导最终检索。 | true |
| unknown_cycle_feature_0 | 未知循环统计特征 0 | unknown cycle feature 0 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_1 | 未知循环统计特征 1 | unknown cycle feature 1 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_10 | 未知循环统计特征 10 | unknown cycle feature 10 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_11 | 未知循环统计特征 11 | unknown cycle feature 11 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_12 | 未知循环统计特征 12 | unknown cycle feature 12 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_13 | 未知循环统计特征 13 | unknown cycle feature 13 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_14 | 未知循环统计特征 14 | unknown cycle feature 14 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_15 | 未知循环统计特征 15 | unknown cycle feature 15 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_16 | 未知循环统计特征 16 | unknown cycle feature 16 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_17 | 未知循环统计特征 17 | unknown cycle feature 17 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_18 | 未知循环统计特征 18 | unknown cycle feature 18 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_19 | 未知循环统计特征 19 | unknown cycle feature 19 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_2 | 未知循环统计特征 2 | unknown cycle feature 2 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_20 | 未知循环统计特征 20 | unknown cycle feature 20 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_21 | 未知循环统计特征 21 | unknown cycle feature 21 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_22 | 未知循环统计特征 22 | unknown cycle feature 22 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_23 | 未知循环统计特征 23 | unknown cycle feature 23 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_24 | 未知循环统计特征 24 | unknown cycle feature 24 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_25 | 未知循环统计特征 25 | unknown cycle feature 25 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_26 | 未知循环统计特征 26 | unknown cycle feature 26 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_27 | 未知循环统计特征 27 | unknown cycle feature 27 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_28 | 未知循环统计特征 28 | unknown cycle feature 28 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_29 | 未知循环统计特征 29 | unknown cycle feature 29 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_3 | 未知循环统计特征 3 | unknown cycle feature 3 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_30 | 未知循环统计特征 30 | unknown cycle feature 30 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_31 | 未知循环统计特征 31 | unknown cycle feature 31 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_4 | 未知循环统计特征 4 | unknown cycle feature 4 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_5 | 未知循环统计特征 5 | unknown cycle feature 5 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_6 | 未知循环统计特征 6 | unknown cycle feature 6 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_7 | 未知循环统计特征 7 | unknown cycle feature 7 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_8 | 未知循环统计特征 8 | unknown cycle feature 8 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_cycle_feature_9 | 未知循环统计特征 9 | unknown cycle feature 9 | unknown | diagnostics | [] | unknown | dataset-specific cycle summary | placeholder name for an unexplained cycle-level feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_0 | 未知工况特征 0 | unknown operation feature 0 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_1 | 未知工况特征 1 | unknown operation feature 1 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_10 | 未知工况特征 10 | unknown operation feature 10 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_11 | 未知工况特征 11 | unknown operation feature 11 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_12 | 未知工况特征 12 | unknown operation feature 12 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_13 | 未知工况特征 13 | unknown operation feature 13 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_14 | 未知工况特征 14 | unknown operation feature 14 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_15 | 未知工况特征 15 | unknown operation feature 15 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_2 | 未知工况特征 2 | unknown operation feature 2 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_3 | 未知工况特征 3 | unknown operation feature 3 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_4 | 未知工况特征 4 | unknown operation feature 4 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_5 | 未知工况特征 5 | unknown operation feature 5 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_6 | 未知工况特征 6 | unknown operation feature 6 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_7 | 未知工况特征 7 | unknown operation feature 7 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_8 | 未知工况特征 8 | unknown operation feature 8 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |
| unknown_operation_feature_9 | 未知工况特征 9 | unknown operation feature 9 | unknown | diagnostics | [] | unknown | dataset-specific operation summary | placeholder name for an unexplained operation feature | 默认不启用；仅用于兼容旧数据或 smoke 测试。 | false |

## Feature Descriptions

### `charge_rate_bucket`
- 中文含义：充电倍率分桶
- 英文含义：charge-rate bucket
- 说明：充电倍率类别，默认由 d_operation 的连续倍率特征替代。

### `chemistry_family`
- 中文含义：电池化学体系
- 英文含义：battery chemistry family
- 说明：正极化学体系，例如 LFP、NCM、NCA。参与 d_metadata，作为 soft penalty。

### `domain_label`
- 中文含义：工况域标签
- 英文含义：domain label
- 说明：由 chemistry、倍率、温度、协议等组合得到的 domain。用于 d_metadata 的软约束。

### `nominal_capacity_bucket`
- 中文含义：额定容量分桶
- 英文含义：nominal capacity bucket
- 说明：电芯额定容量范围类别，可作为可选元信息。

### `source_dataset`
- 中文含义：数据集来源
- 英文含义：source dataset
- 说明：样本来自 TJU、HUST、MIT 或其他数据集。主要用于诊断跨数据集偏移。

### `temperature_bucket`
- 中文含义：温度分桶
- 英文含义：temperature bucket
- 说明：环境温度或测试温度类别，默认由 d_operation 的连续温度特征替代。

### `voltage_window_bucket`
- 中文含义：电压窗口类别
- 英文含义：voltage window bucket
- 说明：电池充放电截止电压范围的类别，用于避免比较不可比的电压曲线。

### `charge_c_rate_mean`
- 中文含义：充电倍率均值
- 英文含义：mean charge C-rate
- 说明：历史窗口内充电倍率平均值，用于衡量平均充电强度。

### `charge_c_rate_std`
- 中文含义：充电倍率标准差
- 英文含义：standard deviation of charge C-rate
- 说明：历史窗口内充电倍率波动，用于衡量变工况强度。

### `charge_duration_mean`
- 中文含义：平均充电时长
- 英文含义：mean charge duration
- 说明：窗口内每个循环平均充电时间，可作为可选工况特征。

### `discharge_c_rate_mean`
- 中文含义：放电倍率均值
- 英文含义：mean discharge C-rate
- 说明：历史窗口内放电倍率平均值，用于衡量平均放电强度。

### `discharge_c_rate_std`
- 中文含义：放电倍率标准差
- 英文含义：standard deviation of discharge C-rate
- 说明：历史窗口内放电倍率波动，用于衡量放电工况变化。

### `discharge_duration_mean`
- 中文含义：平均放电时长
- 英文含义：mean discharge duration
- 说明：窗口内每个循环平均放电时间，可作为可选工况特征。

### `dod_estimate`
- 中文含义：估计放电深度
- 英文含义：estimated depth of discharge
- 说明：由容量摆幅或 SOC 区间估计的 DoD。很多数据集无法稳定计算，因此默认关闭。

### `protocol_change_rate`
- 中文含义：协议切换频率
- 英文含义：protocol change rate
- 说明：窗口内倍率、温度或其他策略发生变化的频率，用于衡量变工况程度。

### `rest_duration_mean`
- 中文含义：平均静置时长
- 英文含义：mean rest duration
- 说明：窗口内每个循环平均静置时间，尤其和 relaxation 可用性相关。

### `temperature_max`
- 中文含义：最高温度
- 英文含义：maximum temperature
- 说明：历史窗口内最高温度，用于捕捉高温暴露。

### `temperature_mean`
- 中文含义：温度均值
- 英文含义：mean temperature
- 说明：历史窗口内温度平均值，用于衡量平均热环境。

### `temperature_std`
- 中文含义：温度标准差
- 英文含义：standard deviation of temperature
- 说明：历史窗口内温度波动，用于衡量热环境变化。

### `dq_dv_peak_position`
- 中文含义：dq/dv 峰值位置
- 英文含义：voltage position of dq/dv peak
- 说明：dq/dv 峰值所在的电压位置或归一化索引，用于反映反应峰位置偏移。

### `dq_dv_peak_value`
- 中文含义：dq/dv 峰值大小
- 英文含义：peak value of dq/dv curve
- 说明：由 partial charging curve 近似计算 dq/dv 后得到的最大峰值。

### `q_high_voltage_slope`
- 中文含义：高电压区 partial charging 斜率
- 英文含义：high-voltage slope of partial charging curve
- 说明：partial charging curve 在高电压区间的斜率，用于刻画高电压区容量-电压响应。

### `q_low_voltage_slope`
- 中文含义：低电压区 partial charging 斜率
- 英文含义：low-voltage slope of partial charging curve
- 说明：partial charging curve 在低电压区间的斜率，用于刻画低电压区容量-电压响应。

### `q_mean`
- 中文含义：partial charging 曲线均值
- 英文含义：mean of partial charging curve
- 说明：partial charging curve 的平均值，低维表示充电曲线整体水平。

### `q_mid_voltage_slope`
- 中文含义：中电压区 partial charging 斜率
- 英文含义：mid-voltage slope of partial charging curve
- 说明：partial charging curve 在中电压区间的斜率，用于刻画主要平台区间容量-电压响应。

### `q_std`
- 中文含义：partial charging 曲线标准差
- 英文含义：standard deviation of partial charging curve
- 说明：partial charging curve 在电压段上的标准差，表示曲线变化程度。

### `q_total`
- 中文含义：partial charging 累计电荷总量
- 英文含义：total cumulative charge in partial charging curve
- 说明：在选定电压区间内由 partial charging curve 计算得到的累计输入电荷总量。

### `relaxation_area_or_tau_proxy`
- 中文含义：弛豫面积或时间常数 proxy
- 英文含义：relaxation area or time-constant proxy
- 说明：弛豫电压曲线面积或近似时间常数，用于表示整体弛豫恢复程度和速度。

### `relaxation_delta_v`
- 中文含义：弛豫电压变化量
- 英文含义：voltage change during relaxation
- 说明：弛豫开始和结束之间的电压差，用于反映极化恢复幅度。

### `relaxation_initial_slope`
- 中文含义：弛豫初期斜率
- 英文含义：initial slope of relaxation voltage
- 说明：弛豫开始阶段电压恢复速度，用于反映快速极化恢复。

### `relaxation_late_slope`
- 中文含义：弛豫后期斜率
- 英文含义：late slope of relaxation voltage
- 说明：弛豫后期电压恢复速度，用于反映慢速恢复过程。

### `partial_charge_curve`
- 中文含义：局部充电曲线
- 英文含义：partial charging curve
- 说明：在选定电压区间内计算 q(V_i)=∫|I(t)|dt，用于刻画部分充电响应。

### `partial_charge_mask`
- 中文含义：局部充电曲线可用标记
- 英文含义：partial charging availability mask
- 说明：标记 partial charging curve 是否可用。

### `relaxation_curve`
- 中文含义：弛豫电压曲线
- 英文含义：relaxation voltage curve
- 说明：充电结束后的静置弛豫电压曲线，用于观察极化恢复。

### `relaxation_mask`
- 中文含义：弛豫曲线可用标记
- 英文含义：relaxation availability mask
- 说明：标记 relaxation curve 是否可用。

### `delta_v_curve_q`
- 中文含义：Q 轴充放电电压差曲线
- 英文含义：charge-discharge voltage gap curve
- 说明：DeltaV(Q)=Vc(Q)-Vd(Q)，可作为极化和阻抗变化的宏观 proxy。

### `qv_maps`
- 中文含义：Q 轴对齐的容量-电压多通道曲线
- 英文含义：Q-indexed voltage-current feature maps
- 说明：将每个循环的充电/放电电压和电流插值到归一化容量 Q 轴上。通道包括 Vc(Q)、Vd(Q)、Ic(Q)、Id(Q)、DeltaV(Q)、R(Q)。

### `r_curve_q`
- 中文含义：Q 轴近似极化/阻抗曲线
- 英文含义：Q-indexed polarization or resistance proxy
- 说明：R(Q)=DeltaV(Q)/(Ic(Q)-Id(Q)+eps)，用于比较容量维度上的极化/内阻相关状态。

### `vc_curve_q`
- 中文含义：Q 轴充电电压曲线
- 英文含义：charge voltage curve indexed by normalized capacity
- 说明：Vc(Q)，归一化容量位置 Q 上的充电电压曲线。默认不作为主检索通道，以避免协议差异过大。

### `vd_curve_q`
- 中文含义：Q 轴放电电压曲线
- 英文含义：discharge voltage curve indexed by normalized capacity
- 说明：Vd(Q)，归一化容量位置 Q 上的放电电压曲线，用于比较放电平台和曲线形状。

### `delta_v_mean`
- 中文含义：充放电电压差均值
- 英文含义：mean charge-discharge voltage gap
- 说明：DeltaV(Q) 在 Q 轴上的均值，用于稳定比较整体极化差异。

### `delta_v_q95`
- 中文含义：充放电电压差 95 分位数
- 英文含义：95th percentile of charge-discharge voltage gap
- 说明：DeltaV(Q) 的高分位统计，用于捕捉局部较强极化。

### `delta_v_std`
- 中文含义：充放电电压差标准差
- 英文含义：standard deviation of charge-discharge voltage gap
- 说明：DeltaV(Q) 在 Q 轴上的标准差，反映极化差异在容量区间内的变化程度。

### `r_mean`
- 中文含义：近似极化/阻抗均值
- 英文含义：mean resistance proxy
- 说明：R(Q) 的均值，用于低维比较整体极化/阻抗 proxy。

### `r_q95`
- 中文含义：近似极化/阻抗 95 分位数
- 英文含义：95th percentile of resistance proxy
- 说明：R(Q) 的高分位统计，用于捕捉局部高极化 / 高阻抗 proxy。

### `r_std`
- 中文含义：近似极化/阻抗标准差
- 英文含义：standard deviation of resistance proxy
- 说明：R(Q) 的标准差，反映容量区间内的极化波动。

### `vc_curve_slope_mean`
- 中文含义：充电电压曲线平均斜率
- 英文含义：mean slope of charge voltage curve
- 说明：Vc(Q) 对 Q 的平均斜率，表征充电平台形态。

### `vd_curve_slope_mean`
- 中文含义：放电电压曲线平均斜率
- 英文含义：mean slope of discharge voltage curve
- 说明：Vd(Q) 对 Q 的平均斜率，表征放电平台形态。

### `composite_distance`
- 中文含义：综合检索距离
- 英文含义：composite retrieval distance
- 说明：由 enabled=true 的核心距离分量加权求和得到。数值越小表示 reference 越适合作为历史参考案例。

### `d_metadata`
- 中文含义：元信息距离
- 英文含义：metadata distance
- 说明：query 和 reference 在 chemistry、domain、voltage window 等类别信息上的 penalty。数值越小表示背景越可比。

### `d_operation`
- 中文含义：工况距离
- 英文含义：operation condition distance
- 说明：query 和 reference 在 C-rate、温度、DoD、协议变化等工况特征上的距离。数值越小表示运行压力历史越相似。

### `d_physics`
- 中文含义：物理启发特征距离
- 英文含义：physics-inspired feature distance
- 说明：query 和 reference 在 12 维 partial charge / relaxation 特征上的距离。数值越小表示退化状态 proxy 越相似。

### `d_qv_shape`
- 中文含义：容量-电压曲线形状距离
- 英文含义：Q-V curve shape distance
- 说明：query 和 reference 在 Vd(Q)、DeltaV(Q)、R(Q) 或 Q-V summary stats 上的距离。数值越小表示电化学曲线形态越相似。

### `d_soh_state`
- 中文含义：SOH 状态距离
- 英文含义：SOH state distance
- 说明：query 和 reference 在 anchor_soh、slope、curvature、degradation_stage 上的距离。数值越小越相似。

### `d_tsfm`
- 中文含义：时间序列基础模型嵌入距离
- 英文含义：TSFM embedding distance
- 说明：query 和 reference 在 TSFM embedding 空间中的距离。数值越小表示通用时序形态越相似。

### `retrieval_confidence`
- 中文含义：检索置信度
- 英文含义：retrieval confidence
- 说明：根据 top-k 综合距离、距离离散度、特征可用性和 metadata match rate 计算得到。数值越大表示这次检索越可靠。

### `partial_charge_availability_ratio`
- 中文含义：局部充电曲线可用率
- 英文含义：partial charging availability ratio
- 说明：历史窗口内 partial charging curve 的可用率。

### `physics_availability_ratio`
- 中文含义：物理特征可用率
- 英文含义：physics feature availability ratio
- 说明：历史窗口内 physics features 的整体可用率，用于提醒 router 当前物理信息是否可靠。

### `relaxation_availability_ratio`
- 中文含义：弛豫曲线可用率
- 英文含义：relaxation availability ratio
- 说明：历史窗口内 relaxation curve 的可用率。

### `anchor_soh`
- 中文含义：锚点 SOH
- 英文含义：anchor state of health
- 说明：历史输入窗口最后一个循环的 SOH。用于还原 pred_soh = anchor_soh + pred_delta_soh，也用于检索当前健康状态。

### `degradation_stage`
- 中文含义：退化阶段
- 英文含义：degradation stage
- 说明：根据 anchor_soh 或退化状态划分的 early/middle/late 类别。用于诊断和软约束检索。

### `recent_soh_curvature`
- 中文含义：近期 SOH 退化曲率
- 英文含义：recent SOH degradation curvature
- 说明：SOH 序列的二阶趋势或二阶差分统计，用于表示退化是否加速。

### `recent_soh_slope`
- 中文含义：近期 SOH 衰减斜率
- 英文含义：recent SOH degradation slope
- 说明：对历史窗口内最近若干 SOH 点做线性拟合得到的一阶斜率。用于衡量近期退化速度。

### `soh_seq`
- 中文含义：历史 SOH 序列
- 英文含义：historical SOH sequence
- 说明：输入窗口内连续 N 个循环的 SOH 序列。用于预测主干输入、近期斜率与曲率计算，也可作为 TSFM embedding 的输入来源。

### `target_delta_soh`
- 中文含义：未来 SOH 变化量标签
- 英文含义：future delta SOH target
- 说明：未来 H 个循环的 SOH 相对 anchor_soh 的变化量，是监督训练的主标签。

### `target_soh`
- 中文含义：未来 SOH 真实值
- 英文含义：future SOH target
- 说明：未来 H 个循环的真实 SOH。用于评估和可视化。

### `tsfm_embedding`
- 中文含义：时间序列基础模型嵌入
- 英文含义：time-series foundation model embedding
- 说明：由 Chronos、Timer 或其他时间序列基础模型从历史窗口中提取的固定维度向量。主要用于 Stage-1 粗检索，缺乏直接物理解释。

### `unknown_cycle_feature_0`
- 中文含义：未知循环统计特征 0
- 英文含义：unknown cycle feature 0
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_1`
- 中文含义：未知循环统计特征 1
- 英文含义：unknown cycle feature 1
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_10`
- 中文含义：未知循环统计特征 10
- 英文含义：unknown cycle feature 10
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_11`
- 中文含义：未知循环统计特征 11
- 英文含义：unknown cycle feature 11
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_12`
- 中文含义：未知循环统计特征 12
- 英文含义：unknown cycle feature 12
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_13`
- 中文含义：未知循环统计特征 13
- 英文含义：unknown cycle feature 13
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_14`
- 中文含义：未知循环统计特征 14
- 英文含义：unknown cycle feature 14
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_15`
- 中文含义：未知循环统计特征 15
- 英文含义：unknown cycle feature 15
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_16`
- 中文含义：未知循环统计特征 16
- 英文含义：unknown cycle feature 16
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_17`
- 中文含义：未知循环统计特征 17
- 英文含义：unknown cycle feature 17
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_18`
- 中文含义：未知循环统计特征 18
- 英文含义：unknown cycle feature 18
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_19`
- 中文含义：未知循环统计特征 19
- 英文含义：unknown cycle feature 19
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_2`
- 中文含义：未知循环统计特征 2
- 英文含义：unknown cycle feature 2
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_20`
- 中文含义：未知循环统计特征 20
- 英文含义：unknown cycle feature 20
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_21`
- 中文含义：未知循环统计特征 21
- 英文含义：unknown cycle feature 21
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_22`
- 中文含义：未知循环统计特征 22
- 英文含义：unknown cycle feature 22
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_23`
- 中文含义：未知循环统计特征 23
- 英文含义：unknown cycle feature 23
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_24`
- 中文含义：未知循环统计特征 24
- 英文含义：unknown cycle feature 24
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_25`
- 中文含义：未知循环统计特征 25
- 英文含义：unknown cycle feature 25
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_26`
- 中文含义：未知循环统计特征 26
- 英文含义：unknown cycle feature 26
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_27`
- 中文含义：未知循环统计特征 27
- 英文含义：unknown cycle feature 27
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_28`
- 中文含义：未知循环统计特征 28
- 英文含义：unknown cycle feature 28
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_29`
- 中文含义：未知循环统计特征 29
- 英文含义：unknown cycle feature 29
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_3`
- 中文含义：未知循环统计特征 3
- 英文含义：unknown cycle feature 3
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_30`
- 中文含义：未知循环统计特征 30
- 英文含义：unknown cycle feature 30
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_31`
- 中文含义：未知循环统计特征 31
- 英文含义：unknown cycle feature 31
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_4`
- 中文含义：未知循环统计特征 4
- 英文含义：unknown cycle feature 4
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_5`
- 中文含义：未知循环统计特征 5
- 英文含义：unknown cycle feature 5
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_6`
- 中文含义：未知循环统计特征 6
- 英文含义：unknown cycle feature 6
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_7`
- 中文含义：未知循环统计特征 7
- 英文含义：unknown cycle feature 7
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_8`
- 中文含义：未知循环统计特征 8
- 英文含义：unknown cycle feature 8
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_cycle_feature_9`
- 中文含义：未知循环统计特征 9
- 英文含义：unknown cycle feature 9
- 说明：原始数据中存在但尚未完成语义映射的逐循环统计特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_0`
- 中文含义：未知工况特征 0
- 英文含义：unknown operation feature 0
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_1`
- 中文含义：未知工况特征 1
- 英文含义：unknown operation feature 1
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_10`
- 中文含义：未知工况特征 10
- 英文含义：unknown operation feature 10
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_11`
- 中文含义：未知工况特征 11
- 英文含义：unknown operation feature 11
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_12`
- 中文含义：未知工况特征 12
- 英文含义：unknown operation feature 12
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_13`
- 中文含义：未知工况特征 13
- 英文含义：unknown operation feature 13
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_14`
- 中文含义：未知工况特征 14
- 英文含义：unknown operation feature 14
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_15`
- 中文含义：未知工况特征 15
- 英文含义：unknown operation feature 15
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_2`
- 中文含义：未知工况特征 2
- 英文含义：unknown operation feature 2
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_3`
- 中文含义：未知工况特征 3
- 英文含义：unknown operation feature 3
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_4`
- 中文含义：未知工况特征 4
- 英文含义：unknown operation feature 4
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_5`
- 中文含义：未知工况特征 5
- 英文含义：unknown operation feature 5
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_6`
- 中文含义：未知工况特征 6
- 英文含义：unknown operation feature 6
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_7`
- 中文含义：未知工况特征 7
- 英文含义：unknown operation feature 7
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_8`
- 中文含义：未知工况特征 8
- 英文含义：unknown operation feature 8
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。

### `unknown_operation_feature_9`
- 中文含义：未知工况特征 9
- 英文含义：unknown operation feature 9
- 说明：原始数据中存在但尚未完成语义映射的工况特征。该类特征不得作为默认核心检索特征使用。
