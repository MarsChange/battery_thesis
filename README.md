
# Battery-thesis

## Battery SOH forecasting pipeline

This pipeline is purely numerical. It does not use an LLM and it does not use a mechanism-text knowledge base. The historical memory is a numerical historical degradation case memory built from cycle statistics, Q-indexed charge/discharge curve features, partial-charge features, relaxation features, operation features, and optional time-series embeddings.

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
