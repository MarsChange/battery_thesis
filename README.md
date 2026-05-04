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
- `semantic hierarchical router`: 第一层使用已知 `chemistry_family` 做 hard routing，第二层用语义概念先验和小校准器输出 4 个模式专家权重。
- `residual LSTM experts`: 每个 chemistry branch 下有 4 个 LSTM residual experts，只输出 `moe_residual`，最终 `pred_delta = base_delta + moe_residual`。

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

Router 的目标是用语义可解释的数值特征选择专家。它分两层：

- 第一层：使用 metadata 中已知的 `chemistry_family` 直接 hard route 到 `LFP`、`NCM` 或 `NCA` branch，不训练 chemistry classifier。
- 第二层：在对应 chemistry branch 内，对 4 个 residual expert mode 做 soft routing。

第二层 `SemanticHierarchicalRouter` 默认输入分组：

- `soh_state`: `anchor_soh`、`recent_soh_slope`、`recent_soh_curvature`。
- `qv_polarization`: `delta_v_mean`、`delta_v_std`、`delta_v_q95`、`r_mean`、`r_std`、`r_q95`、Q-V 曲线斜率和平台形态统计。
- `partial_charge_shape`: `q_total`、`q_mean`、`q_std`、`dq_dv_peak_value`。
- `operation_stress`: 电流、温度、CC/CV 时长、throughput/energy 变化和 `protocol_change_rate`。
- `retrieval`: `retrieval_confidence`、top-k 综合距离均值/标准差、top-1 距离、特征可用率和 chemistry match ratio。
- `neighbor_vote`: top-k 历史邻居的 semantic mode vote。

Router 先构造 `concept_slow_linear`、`concept_accelerating`、`concept_high_polarization`、`concept_curve_polarization`、`concept_high_operation_stress`、`concept_low_retrieval_reliability`，再形成 semantic prior，最后用小校准器做有限修正。`expert_router_contributions` 和 `semantic_explanation_json` 会记录每个输入分组和语义概念对专家选择的影响。

## Expert Library

默认层次专家库定义在 [configs/battery_soh.yaml](/Users/marc/DeepScientist/battery_ts_rag/configs/battery_soh.yaml)：

第一层 chemistry branches：

- `LFP`
- `NCM`
- `NCA`

每个 chemistry branch 下都有 4 个 residual LSTM experts：

- `slow_linear`
- `accelerating`
- `high_polarization`
- `curve_polarization_expert`

总计 12 个 experts。所有 experts 都是 residual experts，不输出完整 SOH 预测。不再保留 `shared` 专家。`curve_polarization_expert` 用 Q-V 曲线、`DeltaV(Q)` 和 `R(Q)` 表示极化强度，不依赖充电后静置曲线。

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

Stage A 训练基础模型：

```bash
python -m forecasting.train --config configs/battery_soh.yaml
```

Stage B 生成按 cell_uid 划分的 OOF baseline：

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
