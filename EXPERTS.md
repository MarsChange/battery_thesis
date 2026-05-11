# Expert Model Library

本文档说明当前小专家模型库。项目任务是少样本、跨工况电池 SOH 多步预测。专家模型只接收可解释的数值电池特征，不接收 TSFM/Chronos/foundation embedding，不使用文本知识库，也不使用 relaxation voltage。

## 角色定位

基础模型先融合三路预测：

```text
base_delta = fusion_base(fm_delta, rag_delta, pair_delta)
```

专家库只输出 residual correction：

```text
moe_residual = sum_e expert_weight_e * residual_expert_e(...)
pred_delta = base_delta + moe_residual
pred_soh = anchor_soh + pred_delta
```

专家学习目标是：

```text
residual_target = target_delta_soh - base_delta
```

因此专家不是完整预测器，不直接替代 base model。

## 双层专家结构

第一层是 chemistry hard routing。模型直接使用 case metadata 中已知的 `chemistry_family` 选择分支，不训练 chemistry classifier。

| branch | 中文含义 | routing 依据 | 说明 |
| --- | --- | --- | --- |
| `LFP` | 磷酸铁锂分支 | `chemistry_family == "LFP"` | 处理 LFP 窗口。 |
| `NCM` | 镍钴锰分支 | `chemistry_family == "NCM"` | 处理 NCM 窗口。 |
| `NCA` | 镍钴铝分支 | `chemistry_family == "NCA"` | 处理 NCA 窗口。 |

第二层在选中的 chemistry branch 内，对 4 个因子 residual experts 做 soft routing。

| expert | 中文含义 | 设计用途 | 主要证据 |
| --- | --- | --- | --- |
| `high_temperature_expert` | 高温专家 | 修正高温暴露下 base model 低估或高估的退化趋势。 | `temperature_mean`、`temperature_std`、`temperature_max`。 |
| `high_current_expert` | 高电流专家 | 修正大绝对电流或电流波动带来的退化偏差。 | `current_abs_mean`、`current_abs_std`、`current_abs_max`。 |
| `high_cycle_expert` | 高循环老化专家 | 修正低 SOH 或后期循环阶段的长期退化偏差。 | `anchor_soh`、`cycle_aging_index`、`soh_below_0p9_t`。 |
| `high_power_expert` | 高功率专家 | 修正高功率或高能量吞吐窗口的退化偏差。 | `power_energy_proxy`、charge/discharge energy deltas。 |

总计 `3 chemistry branches * 4 factor experts = 12` 个 residual experts。不保留 `shared` expert；通用预测能力由 base model 承担。

## 单个专家结构

当前专家类为 `AttentionGRUResidualExpert`，定义在 [forecasting/model.py](/Users/marc/DeepScientist/battery_ts_rag/forecasting/model.py)。为兼容旧测试，代码保留 `ResidualLSTMExpert` 别名，但实际结构是 attention-GRU residual expert。

| input | shape | 含义 |
| --- | --- | --- |
| `expert_seq` | `[batch, lookback, 14]` | 逐 cycle 的低维数值特征序列。 |
| `expert_context` | `[batch, context_dim]` | 窗口级状态、Q-V summary、检索可靠性和 base prediction summary。 |
| `baseline_delta_detached` | `[batch, horizon]` | `base_delta.detach()`，防止专家训练反向改变 base model。 |
| `future_ops` | `[batch, horizon, 8]` | 可选未来工况摘要；不可用时由 mask 控制。 |

结构：

```text
expert_seq -> Linear -> GRU -> attention pooling + last hidden state
expert_context -> Linear
future_ops -> optional GRU summary
concat(attention_summary, last_hidden, context_embedding, future_embedding, baseline_delta_detached)
  -> MLP decoder
  -> residual_pred
```

长 horizon 配置中，专家 residual 支持平滑化输出，避免局部突刺直接破坏 `base_delta`。最终公式仍保持 `pred_delta = base_delta + moe_residual`。

## expert_seq 特征

`expert_seq` 由 case bank 预处理阶段生成：

```text
case_expert_seq.npy
case_expert_seq_feature_names.json
```

默认每个时间步 14 维：

| feature_name | 中文含义 | 来源 | 用途 |
| --- | --- | --- | --- |
| `soh_t` | 当前循环 SOH | `case_soh_seq` | 表示窗口内健康状态演化。 |
| `qv_dqdv_peak_value_t` | dQ/dV 峰值 | 放电 Q-V 电压窗口 | 表示局部 Q-V 峰强度。 |
| `qv_dqdv_peak_voltage_t` | dQ/dV 峰值电压 | 放电 Q-V 电压窗口 | 表示峰位或平台位置。 |
| `qv_dqdv_area_t` | dQ/dV 面积 | 放电 Q-V 电压窗口 | 表示该电压窗口覆盖的容量贡献。 |
| `qv_capacity_span_t` | Q-V 窗口容量跨度 | 放电 Q-V 电压窗口 | 表示窗口内容量变化幅度。 |
| `temperature_mean_t` | 温度均值 | cycle statistics / operation sequence | 高温专家输入。 |
| `temperature_std_t` | 温度标准差 | cycle statistics / operation sequence | 温度波动输入。 |
| `temperature_max_t` | 最高温度 | cycle statistics / operation sequence | 高温暴露输入。 |
| `current_abs_mean_t` | 绝对电流均值 | cycle statistics / Q-V map | 高电流专家输入。 |
| `current_abs_std_t` | 绝对电流标准差 | cycle statistics / Q-V map | 电流波动输入。 |
| `current_abs_max_t` | 最大绝对电流 | cycle statistics / Q-V map | 高电流暴露输入。 |
| `power_energy_proxy_t` | 功率/能量 proxy | `|V*I|` 或 energy delta | 高功率专家输入。 |
| `cycle_aging_index_t` | 循环老化索引 | SOH/cycle position proxy | 高循环老化专家输入。 |
| `soh_below_0p9_t` | SOH 低于 0.9 指示 | `soh_t` | 后期退化状态输入。 |

若某个原始信号不可用，预处理按特征注册表中的缺失策略填 0，并在日志中记录缺失比例。

## Semantic Router

`SemanticHierarchicalRouter` 严格区分三层量：

| layer | 含义 | 输出 |
| --- | --- | --- |
| raw named features | SOH 状态、Q-V 窗口 summary、温度、电流、功率、检索统计和邻居投票 | 分组输入 |
| semantic concepts | 从 raw features 构造的可解释概念分数 | `concept_*` |
| expert weights | semantic prior、top-k vote 和小校准器结合后的逐 horizon 权重 | `expert_weights_by_horizon` |

语义概念：

| concept | 中文含义 | 典型判定依据 |
| --- | --- | --- |
| `concept_high_temperature` | 高温暴露 | 温度均值或最高温度接近/超过高温区间。 |
| `concept_high_current` | 高电流压力 | 绝对电流均值、标准差或最大值偏高。 |
| `concept_high_cycle_aging` | 高循环老化 | `anchor_soh` 偏低、SOH 低于 0.9 或循环老化 proxy 偏高。 |
| `concept_high_power` | 高功率/高能量吞吐 | 功率或能量 proxy 偏高。 |
| `concept_low_retrieval_reliability` | 检索低可信 | `retrieval_confidence` 低或 top-k 距离离散。 |

Router 输出还包括 `semantic_explanation_json`。该字段用于记录数值证据、主导专家和权重，不调用 LLM。

## 训练阶段

| stage | 训练内容 | 冻结内容 | 主要输出 |
| --- | --- | --- | --- |
| Stage A | general sequence branch、RAG prior branch、pairwise branch、base fusion router | semantic router 和 expert bank | `base_model_best.pt` |
| Stage B | 按 `cell_uid` 生成 OOF baseline | 不训练模型 | `case_baseline_delta_oof.npy`, `case_residual_target_oof.npy` |
| Stage C | semantic router 和 chemistry-specific residual experts | base model 和 retrieval pipeline | `residual_experts_best.pt` |

## 评估输出

评估和随机片段实验中专家相关字段：

| field | 含义 |
| --- | --- |
| `base_delta_h*` | 基础分支融合后的 delta SOH。 |
| `residual_h*` | 专家库输出的残差修正。 |
| `pred_delta_h*` | `base_delta_h* + residual_h*`。 |
| `pred_soh_h*` | `anchor_soh + pred_delta_h*`。 |
| `expert_weights_json` | 选中 chemistry branch 内 4 个 factor experts 的 horizon 平均权重。 |
| `expert_weights_by_horizon_json` | 每个预测 step 上 4 个 factor experts 的动态权重。 |
| `dominant_expert` | 权重最大的因子专家。 |
| `semantic_explanation_json` | chemistry branch、专家权重、主导专家和语义证据。 |
