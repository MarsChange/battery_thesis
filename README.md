
# Battery-thesis

## Battery SOH forecasting pipeline

本项目任务是少样本、跨工况的电池 SOH 多步预测。整套流程是纯数值方法：
- 不使用 LLM。
- 不使用机理文本知识库。
- 历史数据库是 numerical historical degradation case memory，而不是文本知识库。

历史 case memory 由以下数值信息构成：
- cycle statistics
- Q-indexed charge / discharge curve features
- partial-charge features
- relaxation features
- operation features
- optional time-series foundation model embeddings

The forecasting backbone combines:

- physics-inspired feature construction
- multi-stage historical case retrieval
- reference-conditioned trajectory difference learning
- an interpretable physical router driven by physics features, capacity-voltage curve features, relaxation features, operation features, chemistry metadata, and retrieval confidence
- chemistry-aware / degradation-shape experts
- multi-step `future_delta_soh` prediction, with `pred_soh = anchor_soh + pred_delta_soh`

Important constraints:

- `target_query` must never enter training.
- `target_query` must never enter the retrieval DB.
- You should inspect feature-validation and retrieval-validation results before deciding to train the final forecasting model.
- Missing partial-charge or relaxation features are kept via masks instead of dropping samples.

## RAG 检索特征配置

RAG 检索不是只用一个 embedding。检索逻辑由 [configs/rag_retrieval_features.yaml](configs/rag_retrieval_features.yaml) 控制，每个距离分量和每个底层特征都有独立的 `enabled: true/false` 开关。

核心距离分量如下，数值越小表示 query 和 reference 越相似：
- `d_soh_state`
- `d_qv_shape`
- `d_physics`
- `d_operation`
- `d_metadata`
- `d_tsfm`

`composite_distance` 是所有 `enabled=true` 的核心距离分量加权和。  
`retrieval_confidence` 表示这次 top-k 检索是否可靠，数值越大越可信。

`d_tsfm` 的作用：
- 它是时间序列基础模型嵌入距离。
- 它主要用于 Stage-1 粗检索。
- 它补充手工物理特征无法覆盖的整体时序形态。
- 它没有直接物理解释。
- 它不应主导最终检索。

如何关闭 `d_tsfm`：

```yaml
distance_components:
  d_tsfm:
    enabled: false
```

如何只用物理特征检索：
- 关闭 `d_tsfm`
- 开启 `d_soh_state`
- 开启 `d_qv_shape`
- 开启 `d_physics`
- 开启 `d_operation`
- 开启 `d_metadata`

推荐检索消融设置：
- `only_soh_state`
- `only_qv_shape`
- `only_physics`
- `no_tsfm`
- `only_tsfm`
- `full_retrieval`

Router 的可解释性来自命名清楚的物理启发特征、工况特征、检索置信度、chemistry metadata 和专家权重，而不是单纯依赖黑箱 embedding。

完整特征说明见 [FEATURES.md](FEATURES.md)。该文档由 `battery_data/feature_registry.py` 生成，避免 README、配置和代码中的特征语义漂移。

Command order:

1. Build case bank:

```bash
python -m battery_data.build_case_bank --config configs/battery_soh.yaml
```

2. Validate preprocessing features:

```bash
python -m experiments.validate_preprocessing_features --config configs/battery_soh.yaml
```

3. Validate retrieval quality:

```bash
python -m experiments.validate_retrieval_quality --config configs/battery_soh.yaml --num_queries 100
```

4. Train source model:

```bash
python -m forecasting.train --config configs/battery_soh.yaml
```

5. Few-shot adapt:

```bash
python -m forecasting.fewshot_adapt --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/best.pt
```

6. Evaluate:

```bash
python -m forecasting.eval --config configs/battery_soh.yaml --checkpoint output/forecasting/checkpoints/best_adapted.pt --split target_query
```

Outputs:

- Case-bank artifacts and preprocessing figures: `output/case_bank/`
- Feature-validation logs, metrics, figures, and `summary.md`: `output/experiments/feature_validation/<run_id>/`
- Retrieval-validation logs, diagnostics, figures, and `summary.md`: `output/experiments/retrieval_validation/<run_id>/`
- Forecasting checkpoints, logs, predictions, and evaluation figures: `output/forecasting/`

建议运行顺序：
1. 先构建 case bank。
2. 先看 feature validation 的结果。
3. 再看 retrieval validation 的结果。
4. 确认特征和检索都有效后，再训练 source model。
5. 然后做 few-shot 适配和最终评估。
