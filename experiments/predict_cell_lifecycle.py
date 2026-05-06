"""Reconstruct one cell's lifecycle SOH curve from window-level forecasts.

The forecasting task is still multi-step SOH prediction on sliding windows:
`target_delta_soh = future_soh - anchor_soh` and
`pred_soh = anchor_soh + pred_delta_soh`.

This script is only an evaluation/visualization utility. It takes the
window-level prediction CSV produced by `forecasting.eval`, aggregates all
horizon predictions that point to the same physical cycle, and compares the
aggregated predicted SOH with the true lifecycle SOH for one target cell.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecasting.metrics import regression_metrics
from forecasting.train import load_config


def _read_case_rows(case_bank_dir: Path) -> pd.DataFrame:
    parquet_path = case_bank_dir / "case_rows.parquet"
    csv_path = case_bank_dir / "case_rows.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing case_rows.parquet/csv in {case_bank_dir}")


def _horizon_columns(frame: pd.DataFrame, prefix: str) -> List[str]:
    columns = [column for column in frame.columns if column.startswith(prefix)]
    return sorted(columns, key=lambda name: int(name.rsplit("h", 1)[1]))


def _select_default_cell(rows: pd.DataFrame, split: str) -> str:
    subset = rows[rows["split"].astype(str) == split]
    if subset.empty:
        raise ValueError(f"No cases found for split={split!r}.")
    return str(subset.groupby("cell_uid").size().sort_values(ascending=False).index[0])


def _collect_true_lifecycle(
    cell_rows: pd.DataFrame,
    soh_seq: np.ndarray,
    future_soh: np.ndarray,
) -> pd.DataFrame:
    values: Dict[int, list[float]] = defaultdict(list)
    for row_idx, row in cell_rows.iterrows():
        hist_start = int(row["cycle_idx_start"])
        hist = np.asarray(soh_seq[int(row_idx)], dtype=np.float32).reshape(-1)
        for offset, soh_value in enumerate(hist):
            if np.isfinite(soh_value):
                values[hist_start + offset].append(float(soh_value))
        future_start = int(row["target_cycle_idx_start"])
        future = np.asarray(future_soh[int(row_idx)], dtype=np.float32).reshape(-1)
        for offset, soh_value in enumerate(future):
            if np.isfinite(soh_value):
                values[future_start + offset].append(float(soh_value))
    records = [
        {"cycle_idx": cycle_idx, "true_soh": float(np.mean(cycle_values))}
        for cycle_idx, cycle_values in sorted(values.items())
        if cycle_values
    ]
    return pd.DataFrame(records)


def _collect_prediction_lifecycle(
    prediction_rows: pd.DataFrame,
    true_cols: List[str],
    pred_cols: List[str],
) -> pd.DataFrame:
    pred_values: Dict[int, list[float]] = defaultdict(list)
    true_values: Dict[int, list[float]] = defaultdict(list)
    source_cases: Dict[int, list[int]] = defaultdict(list)
    horizon_positions: Dict[int, list[int]] = defaultdict(list)
    for _, row in prediction_rows.iterrows():
        future_start = int(row["target_cycle_idx_start"]) if "target_cycle_idx_start" in row else int(row["window_end"]) + 1
        case_id = int(row["case_id"])
        for horizon_idx, (true_col, pred_col) in enumerate(zip(true_cols, pred_cols), start=1):
            cycle_idx = future_start + horizon_idx - 1
            pred_value = float(row[pred_col])
            true_value = float(row[true_col])
            if np.isfinite(pred_value) and np.isfinite(true_value):
                pred_values[cycle_idx].append(pred_value)
                true_values[cycle_idx].append(true_value)
                source_cases[cycle_idx].append(case_id)
                horizon_positions[cycle_idx].append(horizon_idx)
    records = []
    for cycle_idx in sorted(pred_values):
        preds = np.asarray(pred_values[cycle_idx], dtype=np.float32)
        trues = np.asarray(true_values[cycle_idx], dtype=np.float32)
        records.append(
            {
                "cycle_idx": int(cycle_idx),
                "pred_soh_mean": float(np.mean(preds)),
                "pred_soh_std": float(np.std(preds)),
                "true_soh_at_predicted_cycle": float(np.mean(trues)),
                "num_window_votes": int(len(preds)),
                "source_case_ids_json": json.dumps(source_cases[cycle_idx]),
                "horizon_positions_json": json.dumps(horizon_positions[cycle_idx]),
            }
        )
    return pd.DataFrame(records)


def _write_plot(
    truth: pd.DataFrame,
    prediction: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, (axis, err_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 7.4),
        dpi=180,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]},
    )
    axis.plot(
        truth["cycle_idx"],
        truth["true_soh"],
        color="#1f2933",
        linewidth=2.4,
        label="True SOH lifecycle",
        zorder=4,
    )
    if not prediction.empty:
        axis.plot(
            prediction["cycle_idx"],
            prediction["pred_soh_mean"],
            color="#d62728",
            linewidth=1.35,
            alpha=0.82,
            label="Aggregated predicted SOH",
            zorder=3,
        )
        if "pred_soh_std" in prediction:
            pred = prediction["pred_soh_mean"].to_numpy(dtype=np.float32)
            std = prediction["pred_soh_std"].to_numpy(dtype=np.float32)
            axis.fill_between(
                prediction["cycle_idx"],
                pred - std,
                pred + std,
                color="#d62728",
                alpha=0.12,
                label="Prediction std across windows",
                zorder=2,
            )
        if "true_soh_at_predicted_cycle" in prediction:
            abs_error = np.abs(
                prediction["pred_soh_mean"].to_numpy(dtype=np.float32)
                - prediction["true_soh_at_predicted_cycle"].to_numpy(dtype=np.float32)
            )
            err_axis.plot(
                prediction["cycle_idx"],
                abs_error,
                color="#7f1d1d",
                linewidth=1.1,
                label="Absolute error",
            )
            err_axis.fill_between(prediction["cycle_idx"], 0.0, abs_error, color="#7f1d1d", alpha=0.15)
    axis.set_title(title)
    axis.set_ylabel("SOH")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    err_axis.set_xlabel("Cycle index")
    err_axis.set_ylabel("|error|")
    err_axis.grid(True, alpha=0.25)
    err_axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _safe_json_loads(value: object, default: object) -> object:
    if not isinstance(value, str) or not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _expert_names_from_predictions(prediction_rows: pd.DataFrame) -> List[str]:
    if "semantic_explanation_json" in prediction_rows.columns:
        for value in prediction_rows["semantic_explanation_json"].tolist():
            payload = _safe_json_loads(value, {})
            if isinstance(payload, Mapping):
                mode_weights = payload.get("mode_weights", {})
                if isinstance(mode_weights, Mapping) and mode_weights:
                    return list(mode_weights.keys())
    return ["slow_linear", "accelerating", "high_polarization", "curve_polarization_expert"]


def _expert_weights_for_row(row: pd.Series, expert_names: List[str]) -> np.ndarray:
    payload = _safe_json_loads(row.get("semantic_explanation_json"), {})
    if isinstance(payload, Mapping) and isinstance(payload.get("mode_weights"), Mapping):
        mode_weights = payload["mode_weights"]
        return np.asarray([float(mode_weights.get(name, 0.0)) for name in expert_names], dtype=np.float32)
    weights = _safe_json_loads(row.get("expert_weights_json"), [])
    arr = np.asarray(weights, dtype=np.float32).reshape(-1)
    if arr.size < len(expert_names):
        padded = np.zeros(len(expert_names), dtype=np.float32)
        padded[: arr.size] = arr
        return padded
    return arr[: len(expert_names)].astype(np.float32)


def _collect_expert_weights(
    prediction_rows: pd.DataFrame,
    horizon: int,
    expert_names: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect expert weights by horizon step and by lifecycle cycle.

    The current model emits one expert-weight vector per query window. It does
    not emit a different router decision for each horizon. Therefore, each
    window's expert weights are attributed to every future step predicted from
    that window, then averaged by horizon or physical cycle.
    """

    by_horizon: Dict[int, list[np.ndarray]] = defaultdict(list)
    by_cycle: Dict[int, list[np.ndarray]] = defaultdict(list)
    for _, row in prediction_rows.iterrows():
        weights = _expert_weights_for_row(row, expert_names)
        future_start = int(row["target_cycle_idx_start"]) if "target_cycle_idx_start" in row else int(row["window_end"]) + 1
        for horizon_idx in range(1, horizon + 1):
            cycle_idx = future_start + horizon_idx - 1
            by_horizon[horizon_idx].append(weights)
            by_cycle[cycle_idx].append(weights)

    horizon_records = []
    for horizon_idx in sorted(by_horizon):
        matrix = np.stack(by_horizon[horizon_idx], axis=0)
        record = {"horizon": int(horizon_idx), "num_windows": int(matrix.shape[0])}
        for pos, name in enumerate(expert_names):
            record[name] = float(matrix[:, pos].mean())
        horizon_records.append(record)

    cycle_records = []
    for cycle_idx in sorted(by_cycle):
        matrix = np.stack(by_cycle[cycle_idx], axis=0)
        record = {"cycle_idx": int(cycle_idx), "num_window_votes": int(matrix.shape[0])}
        for pos, name in enumerate(expert_names):
            record[name] = float(matrix[:, pos].mean())
        cycle_records.append(record)
    return pd.DataFrame(horizon_records), pd.DataFrame(cycle_records)


def _write_expert_weight_plot(frame: pd.DataFrame, x_col: str, expert_names: List[str], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "slow_linear": "#2563eb",
        "accelerating": "#dc2626",
        "high_polarization": "#f59e0b",
        "curve_polarization_expert": "#16a34a",
    }
    figure, axis = plt.subplots(figsize=(11, 5.5), dpi=180)
    for name in expert_names:
        if name in frame.columns:
            axis.plot(
                frame[x_col],
                frame[name],
                linewidth=2.0,
                label=name,
                color=colors.get(name),
            )
    axis.set_title(title)
    axis.set_xlabel("Prediction horizon step" if x_col == "horizon" else "Cycle index")
    axis.set_ylabel("Expert weight")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _collect_semantic_summary(prediction_rows: pd.DataFrame, expert_names: List[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    concept_values: Dict[str, list[float]] = defaultdict(list)
    evidence_counts: Dict[str, int] = defaultdict(int)
    dominant_counts: Dict[str, int] = defaultdict(int)
    examples: list[dict[str, object]] = []
    if "semantic_explanation_json" not in prediction_rows.columns:
        return pd.DataFrame(), pd.DataFrame(), examples

    selected_positions = sorted({0, max(len(prediction_rows) // 2, 0), max(len(prediction_rows) - 1, 0)})
    for row_pos, (_, row) in enumerate(prediction_rows.iterrows()):
        payload = _safe_json_loads(row.get("semantic_explanation_json"), {})
        if not isinstance(payload, Mapping):
            continue
        dominant = str(payload.get("dominant_mode", row.get("dominant_expert", "unknown")))
        dominant_counts[dominant] += 1
        concepts = payload.get("semantic_concepts", {})
        if isinstance(concepts, Mapping):
            for key, value in concepts.items():
                try:
                    concept_values[str(key)].append(float(value))
                except Exception:
                    pass
        evidence = payload.get("semantic_evidence", {})
        if isinstance(evidence, Mapping):
            for key in evidence:
                evidence_counts[str(key)] += 1
        if row_pos in selected_positions:
            examples.append(
                {
                    "case_id": int(row["case_id"]),
                    "window_end": int(row["window_end"]),
                    "target_cycle_idx_start": int(row.get("target_cycle_idx_start", int(row["window_end"]) + 1)),
                    "dominant_mode": dominant,
                    "mode_weights": payload.get("mode_weights", {}),
                    "semantic_concepts": concepts,
                    "semantic_evidence": evidence,
                }
            )

    concept_summary = pd.DataFrame(
        [
            {
                "semantic_concept": key,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            for key, values in sorted(concept_values.items())
            if values
        ]
    )
    dominant_summary = pd.DataFrame(
        [
            {
                "semantic_item": key,
                "count": int(value),
                "ratio": float(value / max(len(prediction_rows), 1)),
            }
            for key, value in sorted({**dominant_counts, **{f"evidence:{k}": v for k, v in evidence_counts.items()}}.items())
        ]
    )
    return concept_summary, dominant_summary, examples


def _write_semantic_markdown(
    path: Path,
    *,
    cell_uid: str,
    metrics: Mapping[str, object],
    expert_names: List[str],
    horizon_weights: pd.DataFrame,
    concept_summary: pd.DataFrame,
    dominant_summary: pd.DataFrame,
    examples: list[dict[str, object]],
) -> None:
    def _markdown_table(frame: pd.DataFrame, columns: List[str], floatfmt: str = ".4f") -> str:
        if frame.empty:
            return "_No data available._"
        subset = frame[columns].copy()
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = []
        for _, row in subset.iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, (float, np.floating)):
                    values.append(format(float(value), floatfmt))
                else:
                    values.append(str(value))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join([header, separator, *rows])

    lines = [
        f"# Semantic router explanation for {cell_uid}",
        "",
        "## Cell",
        f"- Dataset: {metrics.get('source_dataset')}",
        f"- Chemistry branch: {metrics.get('chemistry_family')}",
        f"- Domain: {metrics.get('domain_label')}",
        f"- Lifecycle MAE: {float(metrics.get('mae', float('nan'))):.6f}",
        f"- Lifecycle RMSE: {float(metrics.get('rmse', float('nan'))):.6f}",
        "",
        "## Expert Semantics",
        "- `slow_linear`: slow and near-linear degradation residual correction.",
        "- `accelerating`: accelerated degradation or knee-like residual correction.",
        "- `high_polarization`: correction driven by elevated resistance or operation stress.",
        "- `curve_polarization_expert`: correction driven by Q-V curve shape and local polarization signals.",
        "",
        "## Mean Expert Weights By Prediction Horizon",
    ]
    if not horizon_weights.empty:
        lines.append(_markdown_table(horizon_weights, ["horizon", *expert_names]))
    lines.extend(["", "## Semantic Concept Summary"])
    if not concept_summary.empty:
        lines.append(_markdown_table(concept_summary, concept_summary.columns.tolist()))
    lines.extend(["", "## Dominant Mode And Evidence Frequency"])
    if not dominant_summary.empty:
        lines.append(_markdown_table(dominant_summary, dominant_summary.columns.tolist()))
    lines.extend(["", "## Example Window-Level Text Semantics"])
    for example in examples:
        lines.append(f"### case_id={example['case_id']} | window_end={example['window_end']} | target_start={example['target_cycle_idx_start']}")
        lines.append(f"- Dominant mode: `{example['dominant_mode']}`")
        lines.append(f"- Mode weights: `{json.dumps(example['mode_weights'], ensure_ascii=False)}`")
        evidence = example.get("semantic_evidence", {})
        if isinstance(evidence, Mapping):
            for key, text in evidence.items():
                lines.append(f"- {key}: {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_lifecycle_prediction(
    config_path: str | Path,
    predictions_csv: str | Path | None = None,
    cell_uid: str | None = None,
    split: str = "target_query",
    output_dir: str | Path | None = None,
) -> Dict[str, object]:
    cfg = load_config(str(config_path))
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    model_output_dir = Path(cfg.get("model_output_dir", "output/forecasting"))
    predictions_path = Path(predictions_csv) if predictions_csv else model_output_dir / f"predictions_{split}.csv"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {predictions_path}")

    rows = _read_case_rows(case_bank_dir).sort_values("case_id").reset_index(drop=True)
    predictions = pd.read_csv(predictions_path)
    if "target_cycle_idx_start" not in predictions.columns:
        predictions = predictions.merge(
            rows[["case_id", "target_cycle_idx_start", "target_cycle_idx_end"]],
            on="case_id",
            how="left",
        )
    if cell_uid is None:
        cell_uid = _select_default_cell(rows, split)

    cell_rows = rows[(rows["cell_uid"].astype(str) == str(cell_uid)) & (rows["split"].astype(str) == split)].copy()
    if cell_rows.empty:
        raise ValueError(f"No {split} cases found for cell_uid={cell_uid!r}.")
    prediction_rows = predictions[predictions["cell_uid"].astype(str) == str(cell_uid)].copy()
    if prediction_rows.empty:
        raise ValueError(f"No prediction rows found for cell_uid={cell_uid!r} in {predictions_path}.")

    soh_seq = np.load(case_bank_dir / "case_soh_seq.npy")
    future_soh = np.load(case_bank_dir / "case_future_soh.npy")
    truth = _collect_true_lifecycle(cell_rows, soh_seq=soh_seq, future_soh=future_soh)
    true_cols = _horizon_columns(prediction_rows, "true_soh_h")
    pred_cols = _horizon_columns(prediction_rows, "pred_soh_h")
    horizon = len(pred_cols)
    prediction = _collect_prediction_lifecycle(prediction_rows, true_cols=true_cols, pred_cols=pred_cols)

    merged = prediction.merge(
        truth.rename(columns={"true_soh": "true_soh_from_lifecycle"}),
        on="cycle_idx",
        how="left",
    )
    pred_arr = merged["pred_soh_mean"].to_numpy(dtype=np.float32)
    true_arr = merged["true_soh_at_predicted_cycle"].to_numpy(dtype=np.float32)
    metrics = regression_metrics(pred_arr[:, None], true_arr[:, None]) if len(merged) else {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    absolute_error = np.abs(pred_arr - true_arr) if len(merged) else np.zeros(0, dtype=np.float32)
    metrics.update(
        {
            "cell_uid": str(cell_uid),
            "split": str(split),
            "source_dataset": str(cell_rows["source_dataset"].iloc[0]),
            "chemistry_family": str(cell_rows["chemistry_family"].iloc[0]),
            "domain_label": str(cell_rows["domain_label"].iloc[0]),
            "num_windows": int(len(prediction_rows)),
            "num_true_cycles": int(len(truth)),
            "num_predicted_cycles": int(len(prediction)),
            "prediction_cycle_coverage_ratio": float(len(prediction) / max(len(truth), 1)),
            "max_abs_error": float(np.max(absolute_error)) if absolute_error.size else float("nan"),
            "mean_window_votes_per_predicted_cycle": float(prediction["num_window_votes"].mean()) if not prediction.empty else 0.0,
        }
    )

    output_root = Path(output_dir) if output_dir else model_output_dir / "figures" / "lifecycle"
    output_root.mkdir(parents=True, exist_ok=True)
    safe_cell = str(cell_uid).replace("/", "_").replace(":", "_")
    curve_csv = output_root / f"{safe_cell}_lifecycle_prediction.csv"
    metrics_json = output_root / f"{safe_cell}_lifecycle_metrics.json"
    figure_path = output_root / f"{safe_cell}_lifecycle_prediction.png"
    expert_horizon_csv = output_root / f"{safe_cell}_expert_weights_by_horizon.csv"
    expert_cycle_csv = output_root / f"{safe_cell}_expert_weights_by_cycle.csv"
    expert_horizon_figure = output_root / f"{safe_cell}_expert_weights_by_horizon.png"
    expert_cycle_figure = output_root / f"{safe_cell}_expert_weights_by_cycle.png"
    semantic_concepts_csv = output_root / f"{safe_cell}_semantic_concepts_summary.csv"
    semantic_items_csv = output_root / f"{safe_cell}_semantic_items_summary.csv"
    semantic_markdown = output_root / f"{safe_cell}_semantic_explanation.md"
    merged.to_csv(curve_csv, index=False)
    metrics_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=True))
    _write_plot(
        truth=truth,
        prediction=prediction,
        output_path=figure_path,
        title=f"Lifecycle SOH prediction | {cell_uid} | {metrics['chemistry_family']} | {metrics['source_dataset']}",
    )
    expert_names = _expert_names_from_predictions(prediction_rows)
    horizon_weights, cycle_weights = _collect_expert_weights(prediction_rows, horizon=horizon, expert_names=expert_names)
    horizon_weights.to_csv(expert_horizon_csv, index=False)
    cycle_weights.to_csv(expert_cycle_csv, index=False)
    _write_expert_weight_plot(
        horizon_weights,
        x_col="horizon",
        expert_names=expert_names,
        output_path=expert_horizon_figure,
        title=f"Expert weights by prediction step | {cell_uid}",
    )
    _write_expert_weight_plot(
        cycle_weights,
        x_col="cycle_idx",
        expert_names=expert_names,
        output_path=expert_cycle_figure,
        title=f"Expert weights along lifecycle | {cell_uid}",
    )
    concept_summary, dominant_summary, semantic_examples = _collect_semantic_summary(prediction_rows, expert_names)
    concept_summary.to_csv(semantic_concepts_csv, index=False)
    dominant_summary.to_csv(semantic_items_csv, index=False)
    _write_semantic_markdown(
        semantic_markdown,
        cell_uid=str(cell_uid),
        metrics=metrics,
        expert_names=expert_names,
        horizon_weights=horizon_weights,
        concept_summary=concept_summary,
        dominant_summary=dominant_summary,
        examples=semantic_examples,
    )
    return {
        "cell_uid": str(cell_uid),
        "figure_path": str(figure_path),
        "curve_csv": str(curve_csv),
        "metrics_json": str(metrics_json),
        "expert_weights_by_horizon_figure": str(expert_horizon_figure),
        "expert_weights_by_cycle_figure": str(expert_cycle_figure),
        "expert_weights_by_horizon_csv": str(expert_horizon_csv),
        "expert_weights_by_cycle_csv": str(expert_cycle_csv),
        "semantic_explanation_markdown": str(semantic_markdown),
        "semantic_concepts_csv": str(semantic_concepts_csv),
        "semantic_items_csv": str(semantic_items_csv),
        "metrics": metrics,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot one cell's lifecycle SOH prediction from window forecasts")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--predictions-csv", type=str, default=None)
    parser.add_argument("--cell-uid", type=str, default=None)
    parser.add_argument("--split", type=str, default="target_query")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args(argv)
    result = run_lifecycle_prediction(
        config_path=args.config,
        predictions_csv=args.predictions_csv,
        cell_uid=args.cell_uid,
        split=args.split,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
