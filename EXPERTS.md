# Expert Model Library

本文档说明当前项目中的小专家模型库。项目任务是少样本、跨工况电池 SOH 多步预测。专家模型只接收可解释的数值电池特征，不接收外部序列 embedding，也不使用充电后静置电压恢复曲线。

## 角色定位

基础分支先给出多步 delta SOH 的初始预测：

```text
base_delta = fusion_base(fm_delta, rag_delta, pair_delta)
```

专家库根据 Router 权重输出每个专家的 LSTM 预测，并形成最终修正：

```text
moe_residual = sum_e expert_weights[e] * residual_expert_e(...)
pred_delta = base_delta + moe_residual
pred_soh = anchor_soh + pred_delta
```

这里 `moe_residual` 表示专家库对基础预测的多步残差修正。若实验配置要求专家直接输出未来 K 步 SOH delta，也会先转换为相对 `base_delta` 的修正量再进入最终公式，保证 `pred_soh = anchor_soh + pred_delta` 不变。

## 默认专家列表

默认专家列表定义在 [configs/battery_soh.yaml](/Users/marc/DeepScientist/battery_ts_rag/configs/battery_soh.yaml)：

| expert_name | 中文含义 | 设计用途 |
| --- | --- | --- |
| `LFP` | LFP 化学体系专家 | 学习 LFP 电池在相似工况和相似退化状态下的预测修正模式。 |
| `NCM` | NCM 化学体系专家 | 学习 NCM 电池的化学体系相关预测修正模式。 |
| `NCA` | NCA 化学体系专家 | 学习 NCA 电池的化学体系相关预测修正模式。 |
| `slow_linear` | 慢速近线性退化专家 | 处理 SOH 下降平滑、近期斜率较小、曲率不明显的窗口。 |
| `accelerating` | 加速退化专家 | 处理近期 SOH 斜率变陡或曲率显示加速退化的窗口。 |
| `high_polarization` | 高极化专家 | 处理 `DeltaV(Q)` 或 `R(Q)` 偏高的窗口。 |
| `curve_polarization_expert` | Q-V 极化曲线专家 | 处理容量-电压曲线形态、平台偏移、DeltaV/R proxy 显示异常的窗口。 |

不再保留 `shared` 专家。通用预测由 base branches 承担，专家库只保留具有明确化学体系、退化形态或曲线极化语义的小模型。

## 单个专家结构

单个专家类是 `ResidualLSTMExpert`，定义在 [forecasting/model.py](/Users/marc/DeepScientist/battery_ts_rag/forecasting/model.py)。

输入：

- `expert_seq`: shape `[batch, lookback, feature_dim]`，窗口内逐循环低维数值特征。
- `expert_context`: shape `[batch, context_dim]`，窗口级全局上下文。
- `baseline_delta_detached`: shape `[batch, horizon]`，基础预测的 detach 版本。
- `future_ops`: shape `[batch, horizon, future_op_dim]`，可选未来工况。

结构：

```text
expert_seq -> Linear -> LSTM -> last hidden state
expert_context -> Linear
future_ops -> optional summary encoder
concat(last_hidden, context_embedding, future_embedding, baseline_delta_detached)
  -> MLP decoder
  -> residual_pred
```

`baseline_delta_detached` 必须 detach，避免专家训练反向改变基础分支。

## expert_seq 特征

`expert_seq` 由 case bank 预处理阶段生成：

```text
case_expert_seq.npy
case_expert_seq_feature_names.json
```

默认每个时间步包含：

| feature_name | 中文含义 | 来源 | 用途 |
| --- | --- | --- | --- |
| `soh_t` | 当前循环 SOH | `soh_seq` | 表示窗口内健康状态演化。 |
| `voltage_mean_t` | 平均电压 | cycle statistics | 表示电压平台水平。 |
| `current_abs_mean_t` | 平均绝对电流 | cycle statistics | 表示电流工况强度。 |
| `temp_mean_t` | 平均温度 | cycle statistics | 表示热环境。 |
| `delta_v_mean_t` | 充放电电压差均值 | Q-V summary | 表示整体极化 proxy。 |
| `delta_v_std_t` | 充放电电压差标准差 | Q-V summary | 表示极化沿容量轴的波动。 |
| `r_mean_t` | 近似极化/阻抗均值 | Q-V summary | 表示 R(Q) proxy 的整体水平。 |
| `r_std_t` | 近似极化/阻抗标准差 | Q-V summary | 表示 R(Q) proxy 的容量维波动。 |
| `delta_v_q95_t` | 充放电电压差高分位 | Q-V summary | 捕捉局部强极化。 |
| `r_q95_t` | 近似极化/阻抗高分位 | Q-V summary | 捕捉局部高 R(Q) proxy。 |
| `charge_c_rate_mean_t` | 充电倍率 proxy | operation summary | 表示充电强度。 |
| `discharge_c_rate_mean_t` | 放电倍率 proxy | operation summary | 表示放电强度。 |
| `temperature_max_t` | 最高温度 | operation summary | 捕捉高温暴露。 |
| `protocol_change_rate_t` | 协议变化率 proxy | operation summary | 表示窗口内工况变化程度。 |

若某个原始信号不可用，预处理会按特征注册表中的缺失策略填 0，并在 case-bank 日志中记录缺失比例。

## expert_context 特征

`expert_context` 是低维全局上下文，不包含黑箱 hidden states。当前上下文包括：

| group | 内容 | 用途 |
| --- | --- | --- |
| SOH state | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature` | 告诉专家当前健康状态和退化趋势。 |
| Q-V physics | `anchor_physics_features` | 表示窗口末端 DeltaV/R/partial-charge proxy。 |
| chemistry metadata | chemistry embedding | 告诉专家材料体系和工况域。 |
| retrieval diagnostics | `retrieval_confidence`, top-k distance mean/std | 告诉专家历史案例检索是否可靠。 |
| neighbor vote | top-k neighbor physical mode vote | 告诉专家历史邻居更偏向哪种退化模式。 |
| baseline summary | baseline mean/std/last step | 告诉专家基础预测的形状。 |
| future ops summary | future operation mean | 在已知未来工况设置下提供未来使用条件摘要。 |

## Router

专家权重由 `PhysicalDegradationRouter` 输出：

```text
expert_weights: [batch, num_experts]
expert_router_contributions: dict[group_name, tensor]
```

Router 使用 grouped additive logits。默认分组：

- `soh_state`
- `qv_polarization`
- `operation`
- `chemistry`
- `retrieval`
- `neighbor_vote`

每个 group 都有独立 contribution，因此可以在评估时解释某个样本为什么偏向某个专家。

## 训练和评估输出

训练损失以多步 SOH delta 的 MSE/Huber 为主，可加入专家负载均衡正则，避免少数专家长期占用全部权重。

评估 CSV 中专家相关字段：

| field | 含义 |
| --- | --- |
| `base_delta_h*` | 基础分支融合后的 delta SOH。 |
| `residual_h*` | 专家库输出的残差修正。 |
| `pred_delta_h*` | `base_delta_h* + residual_h*`。 |
| `pred_soh_h*` | `anchor_soh + pred_delta_h*`。 |
| `expert_weights_json` | 每个专家的样本级权重。 |
| `expert_router_contributions_json` | router 各输入分组对专家 logits 的贡献。 |

## 运行命令

训练：

```bash
python -m forecasting.train --config configs/battery_soh.yaml
```

few-shot 适配：

```bash
python -m forecasting.fewshot_adapt --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/best.pt
```

评估：

```bash
python -m forecasting.eval --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/best_adapted.pt --split target_query
```
