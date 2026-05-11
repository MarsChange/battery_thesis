# Battery SOH Forecasting

本项目是用于少样本、跨工况锂电池 SOH 多步预测的纯数值框架。样本单位是滑动窗口 case：过去 `lookback` 个 cycle 的数值特征用于预测未来 `horizon` 个 cycle 的 SOH。

本项目不使用 LLM，不使用文本知识库，不使用 TSFM / Chronos / foundation-model embedding，也不使用充电后静置弛豫特征。历史数据库是 numerical historical degradation case memory。

## 预测目标

所有监督标签统一为 delta SOH：

```text
target_delta_soh = future_soh - anchor_soh
pred_soh = anchor_soh + pred_delta_soh
```

其中 `anchor_soh` 是输入窗口最后一个循环的 SOH。

## 主线模块

- `case_bank`: 将每个电芯按 cycle 切成窗口级监督样本，并提取 SOH、放电 Q-V 窗口曲线、温度、电流、功率/能量 proxy 和 metadata。
- `RAG retrieval`: 使用可解释数值距离从历史 case memory 检索 top-k 相似窗口，不使用外部 embedding。
- `base model`: 融合普通数值时序分支、RAG prior 分支和 reference-conditioned pairwise 分支，输出 `base_delta`。
- `semantic router`: 第一层按已知 `chemistry_family` hard routing 到 LFP/NCM/NCA；第二层在对应 chemistry branch 内输出 4 个因子专家的动态权重。
- `factor residual experts`: 每个 chemistry branch 下有 4 个 attention-GRU residual experts，只输出 `moe_residual`，最终 `pred_delta = base_delta + moe_residual`。

## 特征工程

当前主线只使用四类可跨数据集稳定提取的数值特征：

- SOH 状态：`anchor_soh`、`soh_seq`、`recent_soh_slope`、`recent_soh_curvature`、`degradation_stage`。
- 放电 Q-V 窗口：MIT/HUST LFP 使用 2.8-3.6 V；TJU/XJTU NCM 和 TJU-NCA 使用 3.6-4.1 V；MIT 第一个容量标定 cycle 不用于 Q-V 特征。
- Q-V summary：`qv_dqdv_peak_value`、`qv_dqdv_peak_voltage`、`qv_dqdv_area`、`qv_capacity_span`。
- 工况/压力：温度均值/标准差/最大值、绝对电流均值/标准差/最大值、功率或能量 proxy、循环老化状态。

旧的 `DeltaV(Q)`、`R(Q)`、内阻 proxy 和 relaxation voltage 不进入当前主线预测、RAG、Router 或专家库。

## RAG 检索特征配置

RAG 检索由 [configs/retrieval_features.yaml](/Users/marc/DeepScientist/battery_ts_rag/configs/retrieval_features.yaml) 控制。每个距离分量是一个布尔开关，`true` 表示参与 `composite_distance`。

默认启用：

- `d_soh_state`: 比较 anchor SOH、近期斜率/曲率、退化阶段和 anchor 对齐的历史 SOH 形状。
- `d_qv_shape`: 比较指定放电电压窗口内的 `Qd(V)` 和 `dQ/dV(V)` 曲线形状。
- `d_physics`: 比较 dQ/dV 峰值、峰值电压、Q-V 面积和容量跨度。
- `d_metadata`: 比较 chemistry、domain、电压窗口，以及电流、温度、功率/能量 proxy、归一化容量变化的滑窗数值序列。

默认关闭：

- `d_operation`: 仅保留兼容旧输出，不作为独立检索分量。

Stage-1 粗检索固定使用 `handcrafted_retrieval_embedding`，由 SOH 状态、Q-V summary、物理 summary 和 metadata numeric summary 拼接得到。

## 专家库

第一层 chemistry branch：

- `LFP`
- `NCM`
- `NCA`

每个 chemistry branch 下有 4 个因子 residual experts：

- `high_temperature_expert`: 高温暴露因子。
- `high_current_expert`: 高绝对电流因子。
- `high_cycle_expert`: 低 SOH / 高循环老化因子。
- `high_power_expert`: 高功率或高能量吞吐因子。

总计 12 个 experts。没有 `shared` expert；通用预测由 base model 承担。更详细说明见 [EXPERTS.md](/Users/marc/DeepScientist/battery_ts_rag/EXPERTS.md)。

## Router 解释输出

Router 先从命名数值特征构造语义概念：

- `concept_high_temperature`
- `concept_high_current`
- `concept_high_cycle_aging`
- `concept_high_power`
- `concept_low_retrieval_reliability`

再结合 top-k reference 的语义投票和小校准器，输出逐 horizon 的 expert weights。评估与随机片段实验会保存 `semantic_explanation_json` 和 markdown 解释，供后续外部自然语言分析使用；当前代码本身不调用 LLM。

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

Stage B 生成按 `cell_uid` 划分的 OOF baseline：

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

随机片段预测：

```bash
python -m experiments.random_segment_soh_prediction --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/best_adapted.pt --split target_query --history-length 64 --future-length 64
```

随机片段实验会额外输出 `*_retrieval_topk_segments.png` 和 `*_retrieval_topk_segments.csv`，用于检查 query 与 top-k reference 在历史片段和未来片段上的形态是否可比。

## 输出路径

- Case-bank artifacts: `output/case_bank/`
- Feature validation: `output/experiments/feature_validation/<run_id>/`
- Retrieval validation: `output/experiments/retrieval_validation/<run_id>/`
- Forecasting checkpoints and evaluation: `output/forecasting/`
- Random segment prediction: `output/forecasting/figures/random_segments/`

## 数据泄漏约束

- `target_query` 不得进入训练集。
- `target_query` 不得进入 RAG reference database。
- `target_query` 可以作为 query 用于检索诊断和最终评估。
- split 必须按 `cell_uid` 组织，不能按窗口随机切分。
