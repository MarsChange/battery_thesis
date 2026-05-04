# Expert Model Library

本文档说明当前项目中的小专家模型库。项目任务是少样本、跨工况电池 SOH 多步预测。专家模型只接收可解释的数值电池特征，不接收外部序列 embedding，不使用文本知识库，也不使用充电后静置电压恢复曲线。

## 角色定位

基础分支先给出未来多步 delta SOH 的基础预测：

```text
base_delta = fusion_base(fm_delta, rag_delta, pair_delta)
```

专家库只做残差修正：

```text
moe_residual = sum_m expert_weights[m] * residual_expert_m(...)
pred_delta = base_delta + moe_residual
pred_soh = anchor_soh + pred_delta
```

这里 `moe_residual` 表示专家库对基础预测的多步残差修正。专家学习目标是：

```text
residual_target = target_delta_soh - base_delta
```

因此专家不是完整预测器，不直接输出最终 SOH。

## 层次专家结构

默认配置见 [configs/battery_soh.yaml](/Users/marc/DeepScientist/battery_ts_rag/configs/battery_soh.yaml)。

第一层是 chemistry hard routing。模型直接使用 case metadata 中已知的 `chemistry_family` 选择分支，不训练 chemistry classifier：

| branch | 中文含义 | routing 依据 | 说明 |
| --- | --- | --- | --- |
| `LFP` | 磷酸铁锂分支 | `chemistry_family == "LFP"` | 只处理已标注为 LFP 的窗口。 |
| `NCM` | 镍钴锰分支 | `chemistry_family == "NCM"` | 只处理已标注为 NCM 的窗口。 |
| `NCA` | 镍钴铝分支 | `chemistry_family == "NCA"` | 只处理已标注为 NCA 的窗口。 |

第二层在选中的 chemistry branch 内，对 4 个 residual LSTM expert mode 做 soft routing：

| mode | 中文含义 | 设计用途 |
| --- | --- | --- |
| `slow_linear` | 慢速近线性退化专家 | 处理 SOH 下降平滑、近期斜率较小、曲率不明显、运行压力较低的窗口。 |
| `accelerating` | 加速退化专家 | 处理近期 SOH 斜率变陡或曲率显示加速退化的窗口。 |
| `high_polarization` | 高极化专家 | 处理 `R(Q)`、`r_mean`、`r_q95` 或电流/温度压力偏高的窗口。 |
| `curve_polarization_expert` | Q-V 极化曲线专家 | 处理 `DeltaV(Q)`、`delta_v_q95`、平台斜率或 `dq/dv` 峰值显示曲线极化异常的窗口。 |

总计 `3 chemistry branches * 4 expert modes = 12` 个 residual LSTM experts。不保留 `shared` 专家；通用预测由基础分支承担。

## 单个专家结构

单个专家类是 `ResidualLSTMExpert`，定义在 [forecasting/model.py](/Users/marc/DeepScientist/battery_ts_rag/forecasting/model.py)。

输入：

| input | shape | 含义 |
| --- | --- | --- |
| `expert_seq` | `[batch, lookback, 14]` | 窗口内逐 cycle 的低维数值特征序列。 |
| `expert_context` | `[batch, context_dim]` | 窗口级全局上下文，包括状态、物理 proxy、metadata embedding、检索可靠性和 baseline summary。 |
| `baseline_delta_detached` | `[batch, horizon]` | 基础预测的 detach 版本，防止专家训练反向改变 base model。 |
| `future_ops` | `[batch, horizon, 8]` | 可选未来工况；如果不可用则由 mask 控制。 |

结构：

```text
expert_seq -> Linear -> LSTM -> last hidden state
expert_context -> Linear
future_ops -> optional GRU summary
concat(last_hidden, context_embedding, future_embedding, baseline_delta_detached)
  -> MLP decoder
  -> residual_pred
```

## expert_seq 特征

`expert_seq` 由 case bank 预处理阶段生成：

```text
case_expert_seq.npy
case_expert_seq_feature_names.json
```

默认每个时间步包含 14 维：

| feature_name | 中文含义 | 来源 | 用途 |
| --- | --- | --- | --- |
| `soh_t` | 当前循环 SOH | `case_soh_seq` | 表示窗口内健康状态演化。 |
| `voltage_mean_t` | 平均电压 | cycle statistics | 表示电压工作点和平台水平。 |
| `current_abs_mean_t` | 平均绝对电流 | cycle statistics | 表示电流工况强度。 |
| `temp_mean_t` | 平均温度 | cycle statistics | 表示热环境。 |
| `delta_v_mean_t` | 充放电电压差均值 | Q-V summary | 表示整体极化 proxy。 |
| `delta_v_std_t` | 充放电电压差标准差 | Q-V summary | 表示极化沿容量轴的波动。 |
| `r_mean_t` | 近似极化/阻抗均值 | Q-V summary | 表示 `R(Q)` proxy 的整体水平。 |
| `r_std_t` | 近似极化/阻抗标准差 | Q-V summary | 表示 `R(Q)` proxy 的容量维波动。 |
| `delta_v_q95_t` | 充放电电压差高分位 | Q-V summary | 捕捉局部强极化。 |
| `r_q95_t` | 近似极化/阻抗高分位 | Q-V summary | 捕捉局部高 `R(Q)` proxy。 |
| `charge_c_rate_mean_t` | 充电倍率强度 proxy | operation summary | 当前实现由平均电流和容量近似得到，用于表示充电强度。 |
| `discharge_c_rate_mean_t` | 放电倍率强度 proxy | operation summary | 当前实现由平均电流和容量近似得到，用于表示放电强度。 |
| `temperature_max_t` | 最高温度 | operation summary | 捕捉高温暴露。 |
| `protocol_change_rate_t` | 协议变化率 proxy | operation summary | 表示相邻 cycle 工况强度变化。 |

若某个原始信号不可用，预处理按特征注册表中的缺失策略填 0，并在 case-bank 日志中记录缺失比例。

## expert_context 特征

`expert_context` 是低维全局上下文，不包含外部序列 embedding，也不包含充电后静置弛豫特征。

| group | 内容 | 用途 |
| --- | --- | --- |
| SOH state | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature` | 告诉专家当前健康状态和退化趋势。 |
| Q-V physics | `anchor_physics_features[0:12]` | 表示窗口末端 DeltaV/R/partial-charge proxy。 |
| metadata | `meta_embedding` | 编码 cell/domain/chemistry 等已知 metadata；chemistry 分支选择仍由 hard routing 决定。 |
| retrieval diagnostics | `retrieval_confidence`, top-k distance mean/std, feature availability, compatibility mean | 告诉专家历史案例检索是否可靠。 |
| baseline summary | `base_delta_mean`, `base_delta_std`, `base_delta_last` | 告诉专家基础预测的形状。 |
| future ops summary | future operation mean | 在已知未来工况设置下提供未来使用条件摘要。 |

## SemanticHierarchicalRouter

专家权重由 `SemanticHierarchicalRouter` 输出。它严格区分三层量：

| layer | 含义 | 输出 |
| --- | --- | --- |
| raw features | 命名清楚的状态、Q-V、partial charge、工况、检索和邻居投票特征 | 分组输入 |
| semantic concepts | 用平滑函数从 raw features 构造的语义概念分数 | `concept_*` |
| final expert weights | semantic prior 与小校准器结合后的专家权重 | `expert_weights` |

Router 输入分组：

| group | 具体输入 | 作用 |
| --- | --- | --- |
| `soh_state` | `anchor_soh`, `recent_soh_slope`, `recent_soh_curvature` | 判断健康状态、退化速度和加速趋势。 |
| `qv_polarization` | `delta_v_mean/std/q95`, `r_mean/std/q95`, `vc/vd_curve_slope_mean` | 判断极化强度、曲线形态和平台斜率。 |
| `partial_charge_shape` | `q_total`, `q_mean`, `q_std`, `dq_dv_peak_value` | 判断 partial charge 曲线形态和局部峰值。 |
| `operation_stress` | 电流、温度、CC/CV 时间、throughput/energy 变化和 protocol-change proxy | 判断运行压力和工况切换强度。 |
| `retrieval` | `retrieval_confidence`, top-k composite distance mean/std, top1 distance, availability, chemistry match ratio | 判断 RAG top-k 是否可靠。 |
| `neighbor_vote` | top-k reference semantic mode prior 加权投票 | 使用历史相似案例支持当前模式选择。 |

语义概念：

| concept | 中文含义 | 典型证据 |
| --- | --- | --- |
| `concept_slow_linear` | 慢速近线性退化 | slope/curvature 小，operation stress 低。 |
| `concept_accelerating` | 加速退化 | 负斜率增强、曲率增强、工况波动变大。 |
| `concept_high_polarization` | 高极化/高阻抗 proxy | `r_mean`, `r_q95`, 电流或温度偏高。 |
| `concept_curve_polarization` | Q-V 曲线极化异常 | `delta_v_std`, `delta_v_q95`, 曲线斜率或 `dq_dv_peak_value` 偏高。 |
| `concept_high_operation_stress` | 高工况压力 | 电流、温度、吞吐量或协议变化强。 |
| `concept_low_retrieval_reliability` | 检索低可信 | `retrieval_confidence` 低或 top-k 距离偏大。 |

Router 先得到 `semantic_prior`，再结合 top-k 邻居的 `rag_semantic_prior`，最后用小校准器输出 `expert_weights`。小校准器只做有限修正，不能替代语义先验。

## 训练和评估输出

训练流程分三阶段：

| stage | 训练内容 | 冻结内容 | 主要输出 |
| --- | --- | --- | --- |
| Stage A | general sequence branch、RAG prior branch、pairwise branch、base fusion router | semantic router 和 expert bank | `base_model_best.pt` |
| Stage B | 按 `cell_uid` 做 OOF baseline 生成 | 不训练模型 | `case_baseline_delta_oof.npy`, `case_residual_target_oof.npy` |
| Stage C | semantic router 和 chemistry-specific residual LSTM experts | base model 和 retrieval pipeline | `residual_experts_best.pt` |

评估 CSV 中专家相关字段：

| field | 含义 |
| --- | --- |
| `base_delta_h*` | 基础分支融合后的 delta SOH。 |
| `residual_h*` | 专家库输出的残差修正。 |
| `pred_delta_h*` | `base_delta_h* + residual_h*`。 |
| `pred_soh_h*` | `anchor_soh + pred_delta_h*`。 |
| `chemistry_branch` | hard routing 选中的 chemistry branch。 |
| `expert_weights_json` | 选中 chemistry branch 内 4 个 mode experts 的样本级权重。 |
| `dominant_expert` | 权重最大的 semantic mode。 |
| `router_contributions_json` | router 各输入分组对 mode logits 的贡献。 |
| `semantic_explanation_json` | chemistry branch、mode 权重、dominant mode 和语义证据。 |

## 运行命令

Stage A 训练基础模型：

```bash
python -m forecasting.train --config configs/battery_soh.yaml
```

Stage B 生成 OOF baseline：

```bash
python -m forecasting.generate_baseline_oof --config configs/battery_soh.yaml
```

Stage C 训练 semantic router 和 residual experts：

```bash
python -m forecasting.train_residual_experts --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/base_model_best.pt
```

few-shot 适配：

```bash
python -m forecasting.fewshot_adapt --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/residual_experts_best.pt
```

评估：

```bash
python -m forecasting.eval --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/best_adapted.pt --split target_query
```
