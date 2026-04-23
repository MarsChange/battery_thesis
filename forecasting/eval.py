from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from forecasting.data import BatterySOHForecastDataset
from forecasting.metrics import horizon_metrics, regression_metrics
from forecasting.model import BatterySOHForecaster
from forecasting.train import load_config, move_batch_to_device, resolve_device
from forecasting.visualization import plot_group_bar, plot_horizon_error, plot_weight_heatmap
from retrieval.multistage_retriever import component_matrix_to_named_list


def _group_metrics(frame: pd.DataFrame, pred_cols: List[str], true_cols: List[str], group_col: str) -> pd.DataFrame:
    rows = []
    for group, sub in frame.groupby(group_col):
        pred = sub[pred_cols].to_numpy(dtype=np.float32)
        true = sub[true_cols].to_numpy(dtype=np.float32)
        metrics = regression_metrics(pred, true)
        rows.append({group_col: group, **metrics})
    return pd.DataFrame(rows)


def evaluate(cfg: Dict[str, object], checkpoint_path: str | Path, split: str | None = None) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    test_splits = [split] if split is not None else list(cfg.get("eval", {}).get("test_splits", ["target_query"]))
    dataset = BatterySOHForecastDataset(
        case_bank_dir=cfg.get("output_dir", "output/case_bank"),
        splits=test_splits,
        retrieval_cfg=dict(cfg.get("retrieval", {})),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("train", {}).get("batch_size", 64)), shuffle=False)

    model = BatterySOHForecaster(**checkpoint["model_init"])
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    model.to(device)
    model.eval()

    output_dir = Path(cfg.get("model_output_dir", "output/forecasting"))
    figure_dir = output_dir / "figures" / "evaluation"
    figure_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    expert_rows = []
    fusion_rows = []
    router_rows = []
    retrieval_rows = []

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            batch_size = outputs["pred_delta"].shape[0]
            for idx in range(batch_size):
                case_id = int(batch["query"]["case_id"][idx])
                row = dataset.case_rows.iloc[dataset.case_rows.index[dataset.case_rows["case_id"] == case_id][0]]
                pred_delta = outputs["pred_delta"][idx].detach().cpu().numpy()
                pred_soh = outputs["pred_soh"][idx].detach().cpu().numpy()
                true_delta = batch["query"]["target_delta_soh"][idx].detach().cpu().numpy()
                true_soh = batch["query"]["target_soh"][idx].detach().cpu().numpy()
                record = {
                    "case_id": case_id,
                    "cell_uid": row["cell_uid"],
                    "split": row["split"],
                    "chemistry_family": row["chemistry_family"],
                    "domain_label": row["domain_label"],
                    "window_start": int(row["window_start"]),
                    "window_end": int(row["window_end"]),
                    "anchor_soh": float(row["anchor_soh"]),
                    "expert_weights_json": json.dumps(outputs["expert_weights"][idx].detach().cpu().numpy().astype(float).tolist()),
                    "fusion_weights_json": json.dumps(outputs["fusion_weights"][idx].detach().cpu().numpy().astype(float).tolist()),
                    "expert_router_contributions_json": json.dumps({k: v[idx].detach().cpu().numpy().astype(float).tolist() for k, v in outputs["expert_router_contributions"].items()}),
                    "fusion_router_contributions_json": json.dumps({k: v[idx].detach().cpu().numpy().astype(float).tolist() for k, v in outputs["fusion_router_contributions"].items()}),
                    "topk_neighbor_case_ids_json": json.dumps(batch["retrieval"]["neighbor_case_ids"][idx].detach().cpu().numpy().astype(int).tolist()),
                    "topk_component_distances_json": json.dumps(
                        component_matrix_to_named_list(batch["retrieval"]["component_distances"][idx].detach().cpu().numpy().astype(float))
                    ),
                    "topk_composite_distance_json": json.dumps(batch["retrieval"]["composite_distance"][idx].detach().cpu().numpy().astype(float).tolist()),
                    "retrieval_confidence": float(outputs["retrieval_confidence"][idx].detach().cpu().item()),
                }
                for h in range(len(pred_delta)):
                    record[f"true_soh_h{h+1}"] = float(true_soh[h])
                    record[f"pred_soh_h{h+1}"] = float(pred_soh[h])
                    record[f"true_delta_h{h+1}"] = float(true_delta[h])
                    record[f"pred_delta_h{h+1}"] = float(pred_delta[h])
                    record[f"fm_delta_h{h+1}"] = float(outputs["fm_delta"][idx, h].detach().cpu().item())
                    record[f"rag_delta_h{h+1}"] = float(outputs["rag_delta"][idx, h].detach().cpu().item())
                    record[f"pair_delta_h{h+1}"] = float(outputs["pair_delta"][idx, h].detach().cpu().item())
                    record[f"moe_delta_h{h+1}"] = float(outputs["moe_delta"][idx, h].detach().cpu().item())
                prediction_rows.append(record)

                expert_rows.append({"case_id": case_id, **{f"expert_{i+1}": float(v) for i, v in enumerate(outputs["expert_weights"][idx].detach().cpu().numpy())}})
                fusion_rows.append({"case_id": case_id, **{f"branch_{i+1}": float(v) for i, v in enumerate(outputs["fusion_weights"][idx].detach().cpu().numpy())}})
                for group_name, value in outputs["expert_router_contributions"].items():
                    router_rows.append({"case_id": case_id, "router": "expert", "group": group_name, "values_json": json.dumps(value[idx].detach().cpu().numpy().astype(float).tolist())})
                for group_name, value in outputs["fusion_router_contributions"].items():
                    router_rows.append({"case_id": case_id, "router": "fusion", "group": group_name, "values_json": json.dumps(value[idx].detach().cpu().numpy().astype(float).tolist())})
                retrieval_rows.append(
                    {
                        "case_id": case_id,
                        "neighbor_case_ids_json": json.dumps(batch["retrieval"]["neighbor_case_ids"][idx].detach().cpu().numpy().astype(int).tolist()),
                        "component_distances_json": json.dumps(
                            component_matrix_to_named_list(batch["retrieval"]["component_distances"][idx].detach().cpu().numpy().astype(float))
                        ),
                        "composite_distance_json": json.dumps(batch["retrieval"]["composite_distance"][idx].detach().cpu().numpy().astype(float).tolist()),
                        "retrieval_confidence": float(outputs["retrieval_confidence"][idx].detach().cpu().item()),
                    }
                )

    predictions = pd.DataFrame(prediction_rows)
    pred_cols = [col for col in predictions.columns if col.startswith("pred_delta_h")]
    true_cols = [col for col in predictions.columns if col.startswith("true_delta_h")]
    overall_metrics = regression_metrics(predictions[pred_cols].to_numpy(dtype=np.float32), predictions[true_cols].to_numpy(dtype=np.float32))
    horizon = horizon_metrics(predictions[pred_cols].to_numpy(dtype=np.float32), predictions[true_cols].to_numpy(dtype=np.float32))

    predictions.to_csv(output_dir / f"predictions_{'-'.join(test_splits)}.csv", index=False)
    (output_dir / "metrics_overall.json").write_text(json.dumps(overall_metrics, indent=2, ensure_ascii=True))
    pd.DataFrame({"horizon": np.arange(1, len(horizon["mae"]) + 1), "mae": horizon["mae"], "rmse": horizon["rmse"], "mape": horizon["mape"]}).to_csv(output_dir / "metrics_by_horizon.csv", index=False)
    _group_metrics(predictions, pred_cols, true_cols, "chemistry_family").to_csv(output_dir / "metrics_by_chemistry.csv", index=False)
    _group_metrics(predictions, pred_cols, true_cols, "domain_label").to_csv(output_dir / "metrics_by_domain.csv", index=False)
    pd.DataFrame(expert_rows).to_csv(output_dir / "expert_weights_summary.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(output_dir / "fusion_weights_summary.csv", index=False)
    pd.DataFrame(router_rows).to_csv(output_dir / "router_contributions.csv", index=False)
    pd.DataFrame(retrieval_rows).to_csv(output_dir / "retrieval_case_report.csv", index=False)

    plot_horizon_error(horizon["mae"], "Horizon-wise error", figure_dir / "horizon_wise_error.png")
    plot_group_bar(pd.read_csv(output_dir / "metrics_by_chemistry.csv"), "mae", "chemistry_family", "Error by chemistry", figure_dir / "error_by_chemistry.png")
    plot_group_bar(pd.read_csv(output_dir / "metrics_by_domain.csv"), "mae", "domain_label", "Error by domain", figure_dir / "error_by_domain.png")

    expert_frame = pd.DataFrame(expert_rows).drop(columns=["case_id"])
    plot_weight_heatmap(expert_frame.to_numpy(dtype=np.float32), [str(v) for v in predictions["chemistry_family"].tolist()], expert_frame.columns.tolist(), "Expert weights by chemistry samples", figure_dir / "expert_weight_heatmap_by_chemistry.png")
    plot_weight_heatmap(expert_frame.to_numpy(dtype=np.float32), [str(v) for v in predictions["domain_label"].tolist()], expert_frame.columns.tolist(), "Expert weights by stage samples", figure_dir / "expert_weight_heatmap_by_stage.png")

    # Simple prediction examples by chemistry / domain.
    for group_col, file_name in [("chemistry_family", "prediction_examples_by_chemistry.png"), ("domain_label", "prediction_examples_by_domain.png")]:
        figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
        shown = 0
        for _, row in predictions.groupby(group_col).head(1).iterrows():
            pred_series = [row[f"pred_soh_h{i+1}"] for i in range(len(horizon["mae"]))]
            true_series = [row[f"true_soh_h{i+1}"] for i in range(len(horizon["mae"]))]
            axis.plot(np.arange(1, len(pred_series) + 1), true_series, linewidth=1.8, label=f"{row[group_col]} true")
            axis.plot(np.arange(1, len(pred_series) + 1), pred_series, linewidth=1.4, linestyle="--", label=f"{row[group_col]} pred")
            shown += 1
            if shown >= 4:
                break
        axis.set_title(file_name.replace(".png", "").replace("_", " ").title())
        axis.set_xlabel("Horizon")
        axis.set_ylabel("SOH")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncols=2)
        figure.tight_layout()
        figure.savefig(figure_dir / file_name, bbox_inches="tight")
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4.5), dpi=180)
    conf = predictions["retrieval_confidence"].to_numpy(dtype=np.float32)
    mae_per_case = np.abs(predictions[pred_cols].to_numpy(dtype=np.float32) - predictions[true_cols].to_numpy(dtype=np.float32)).mean(axis=1)
    fusion_frame = pd.DataFrame(fusion_rows)
    if "branch_2" in fusion_frame.columns:
        axis.scatter(conf, fusion_frame["branch_2"].to_numpy(dtype=np.float32), s=16, alpha=0.7)
    axis.set_title("Fusion weight by retrieval confidence")
    axis.set_xlabel("Retrieval confidence")
    axis.set_ylabel("RAG branch weight")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "fusion_weight_by_retrieval_confidence.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4.5), dpi=180)
    axis.scatter(conf, mae_per_case, s=16, alpha=0.7)
    axis.set_title("Retrieval confidence vs error")
    axis.set_xlabel("Retrieval confidence")
    axis.set_ylabel("Mean absolute error")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "retrieval_confidence_vs_error.png", bbox_inches="tight")
    plt.close(figure)

    return {"output_dir": str(output_dir), "metrics_overall_path": str(output_dir / "metrics_overall.json")}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate battery SOH forecasting model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = evaluate(cfg, args.checkpoint, split=args.split)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
