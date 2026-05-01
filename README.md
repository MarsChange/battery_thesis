# Battery SOH Forecasting

本项目是用于少样本、跨工况电池 SOH 多步预测的纯数值框架。核心对象是滑动窗口级 case，每个 case 包含历史 SOH、容量-电压曲线、DeltaV/R 极化 proxy、工况统计、metadata 和未来多步 SOH 标签。

本项目不使用 LLM，不使用文本知识库，也不使用外部时序基础模型嵌入。历史数据库是 numerical historical degradation case memory。

## 预测目标

所有监督标签统一为 delta SOH：

```text
target_delta_soh = future_soh - anchor_soh
pred_soh = anchor_soh + pred_delta_soh
```

其中 `anchor_soh` 是输入窗口最后一个循环的 SOH，`future_soh` 是未来 K 个循环的真实 SOH。

## 主线模块

当前预测主干由四部分组成：

- `case_bank`: 将电池循环数据切成窗口级监督样本，提取 SOH、Q-V、DeltaV/R、partial-charge、工况和 metadata 特征。
- `RAG retrieval`: 使用可解释数值距离从历史 case memory 中检索 top-k 相似窗口。
- `router`: 使用命名清楚的物理和工况特征输出专家权重。
- `LSTM experts`: 每个专家接收窗口级数值特征序列，输出未来 K 步 SOH delta 的专家预测或残差修正。

主方法检索和专家输入均不依赖外部序列 embedding。公开数据集中无法稳定提取的充电后静置电压恢复曲线不进入主线特征。

## RAG 检索特征配置

RAG 检索由 [configs/retrieval_features.yaml](/Users/marc/DeepScientist/battery_ts_rag/configs/retrieval_features.yaml) 控制。每个距离分量是一个布尔开关，`true` 表示参与 `composite_distance`，`false` 表示关闭。

默认启用的距离分量：

- `d_soh_state`: SOH 状态距离，比较 `anchor_soh`、近期斜率、曲率和退化阶段。
- `d_qv_shape`: 容量-电压曲线形状距离，比较 `Vd(Q)`、`DeltaV(Q)` 和 `R(Q)`。
- `d_physics`: 物理 proxy 距离，比较 DeltaV/R 统计量、曲线斜率和 partial-charge 统计量。
- `d_metadata`: metadata 与原始工况数值距离，比较 chemistry family、domain label、电压窗口，以及充电电流序列、放电电流序列、温度序列、归一化容量变化序列。

`d_operation` 保留为兼容旧 CSV / batch 字段的诊断输出，但主配置默认关闭，不参与 `composite_distance`。原先 `d_operation` 的 C-rate 分位数、切换率、时长和 throughput 细项不再作为独立检索分量。

`composite_distance` 是启用距离分量的加权和，数值越小表示 query 和 reference 越相似。`retrieval_confidence` 表示本次 top-k 检索是否可靠，数值越大越可信。

Stage-1 粗检索默认使用 `handcrafted_retrieval_embedding`，由 `anchor_soh`、SOH 斜率/曲率、Q-V summary、DeltaV/R proxy 和 metadata raw numeric summary 拼接得到。

## Router 输入

Router 的目标是用自然语言语义可解释的特征选择专家。默认输入分组：

- `soh_state`: `anchor_soh`、`recent_soh_slope`、`recent_soh_curvature`。
- `qv_polarization`: `delta_v_mean`、`delta_v_std`、`delta_v_q95`、`r_mean`、`r_std`、`r_q95`、Q-V 曲线斜率和平台形态统计。
- `operation`: 充电倍率、放电倍率、温度统计、`protocol_change_rate`。
- `chemistry`: 电池化学体系和 domain metadata。
- `retrieval`: `retrieval_confidence`、top-k 综合距离均值和离散度。
- `neighbor_vote`: top-k 历史邻居的物理模式投票结果。

Router 输出 `expert_weights`，每个权重对应一个 LSTM 专家。`expert_router_contributions` 会记录每个输入分组对专家 logits 的贡献，便于解释模型决策。

## Expert Library

默认专家列表定义在 [configs/battery_soh.yaml](/Users/marc/DeepScientist/battery_ts_rag/configs/battery_soh.yaml)：

- `LFP`
- `NCM`
- `NCA`
- `slow_linear`
- `accelerating`
- `high_polarization`
- `curve_polarization_expert`

不再保留 `shared` 专家。`curve_polarization_expert` 用 Q-V 曲线、`DeltaV(Q)` 和 `R(Q)` 表示极化强度，不依赖充电后静置曲线。

更详细说明见 [EXPERTS.md](/Users/marc/DeepScientist/battery_ts_rag/EXPERTS.md)。

## Feature Documentation

全项目统一特征说明由 [battery_data/feature_registry.py](/Users/marc/DeepScientist/battery_ts_rag/battery_data/feature_registry.py) 维护，并生成 [FEATURES.md](/Users/marc/DeepScientist/battery_ts_rag/FEATURES.md)。

每个特征都记录：

- 中文含义和英文含义。
- shape 和单位。
- 数据来源和提取方法。
- 在预测、检索、router 或 expert 中的用途。
- 缺失处理方式和默认启用状态。

## 运行顺序

构建 case bank：

```bash
python -m battery_data.build_case_bank --config configs/battery_soh.yaml
```

验证预处理特征：

```bash
python -m experiments.validate_preprocessing_features --config configs/battery_soh.yaml
```

验证 RAG 检索质量：

```bash
python -m experiments.validate_retrieval_quality --config configs/battery_soh.yaml --num_queries 100
```

训练预测模型：

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

## 输出路径

- Case-bank artifacts: `output/case_bank/`
- Feature validation: `output/experiments/feature_validation/<run_id>/`
- Retrieval validation: `output/experiments/retrieval_validation/<run_id>/`
- Forecasting checkpoints and evaluation: `output/forecasting/`

## 数据泄漏约束

- `target_query` 不得进入训练集。
- `target_query` 不得进入 RAG reference database。
- `target_query` 可以作为 query 用于检索质量诊断和最终评估。
- split 必须按 `cell_uid` 组织，不能把同一个电芯的窗口随机分到训练和测试两侧。
