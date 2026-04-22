from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from battery_data.build_case_bank import build_case_bank, load_config
from experiments.plotting_utils import ensure_dir, save_bar, save_boxplot, save_heatmap
from forecasting.metrics import horizon_metrics, regression_metrics
from retrieval.multistage_retriever import MultiStageBatteryRetriever
from retrieval.physics_distance import operation_distance, physics_feature_distance, qv_map_distance
from retrieval.retrieval_diagnostics import (
    nearest1_prediction,
    rag_only_prediction,
    save_component_distance_bar,
    save_metadata_table,
    save_partial_relax_overlay,
    save_qv_overlay,
    save_soh_history_figure,
    summarize_retrieval_result,
)


def _retrieved_distance_summary(query_idx: int, result, retriever: MultiStageBatteryRetriever) -> Dict[str, float]:
    valid = np.flatnonzero(result.retrieval_mask > 0)
    if len(valid) == 0:
        return {
            "retrieved_anchor_soh_difference": np.nan,
            "retrieved_recent_slope_difference": np.nan,
            "retrieved_qv_distance": np.nan,
            "retrieved_physics_distance": np.nan,
            "retrieved_operation_distance": np.nan,
            "retrieved_future_delta_distance": np.nan,
        }
    ref_ids = result.neighbor_case_ids[valid].astype(int)
    ref_rows = retriever.case_rows.iloc[[retriever.case_id_to_index[int(case_id)] for case_id in ref_ids]]
    query_row = retriever.case_rows.iloc[query_idx]
    q_op = retriever._operation_summary(query_idx)
    q_future = retriever.arrays["future_delta"][query_idx]
    q_qv = retriever.arrays["qv_maps"][query_idx, -1]
    q_qv_mask = retriever.arrays["qv_masks"][query_idx, -1]
    q_phy = retriever.arrays["physics_features"][query_idx, -1]
    q_phy_mask = retriever.arrays["physics_feature_masks"][query_idx, -1]
    return {
        "retrieved_anchor_soh_difference": float(np.mean(np.abs(ref_rows["anchor_soh"].to_numpy(dtype=np.float32) - float(query_row["anchor_soh"])))),
        "retrieved_recent_slope_difference": float(np.mean(np.abs(ref_rows["recent_soh_slope"].to_numpy(dtype=np.float32) - float(query_row["recent_soh_slope"])))),
        "retrieved_qv_distance": float(
            np.mean(
                [
                    qv_map_distance(q_qv, retriever.arrays["qv_maps"][ref_idx, -1], q_qv_mask, retriever.arrays["qv_masks"][ref_idx, -1], retriever.qv_channel_weights)
                    for ref_idx in [retriever.case_id_to_index[int(case_id)] for case_id in ref_ids]
                ]
            )
        ),
        "retrieved_physics_distance": float(
            np.mean(
                [
                    physics_feature_distance(q_phy, retriever.arrays["physics_features"][ref_idx, -1], q_phy_mask, retriever.arrays["physics_feature_masks"][ref_idx, -1])
                    for ref_idx in [retriever.case_id_to_index[int(case_id)] for case_id in ref_ids]
                ]
            )
        ),
        "retrieved_operation_distance": float(
            np.mean(
                [
                    operation_distance(q_op, retriever._operation_summary(ref_idx))
                    for ref_idx in [retriever.case_id_to_index[int(case_id)] for case_id in ref_ids]
                ]
            )
        ),
        "retrieved_future_delta_distance": float(np.mean([np.abs(retriever.arrays["future_delta"][retriever.case_id_to_index[int(case_id)]] - q_future).mean() for case_id in ref_ids])),
    }


def _random_distance_summary(query_idx: int, sampled_ids: np.ndarray, retriever: MultiStageBatteryRetriever) -> Dict[str, float]:
    if len(sampled_ids) == 0:
        return {
            "random_anchor_soh_difference": np.nan,
            "random_recent_slope_difference": np.nan,
            "random_qv_distance": np.nan,
            "random_physics_distance": np.nan,
            "random_operation_distance": np.nan,
            "random_future_delta_distance": np.nan,
        }
    query_row = retriever.case_rows.iloc[query_idx]
    q_op = retriever._operation_summary(query_idx)
    q_future = retriever.arrays["future_delta"][query_idx]
    q_qv = retriever.arrays["qv_maps"][query_idx, -1]
    q_qv_mask = retriever.arrays["qv_masks"][query_idx, -1]
    q_phy = retriever.arrays["physics_features"][query_idx, -1]
    q_phy_mask = retriever.arrays["physics_feature_masks"][query_idx, -1]
    ref_indices = [retriever.case_id_to_index[int(case_id)] for case_id in sampled_ids.tolist()]
    ref_rows = retriever.case_rows.iloc[ref_indices]
    return {
        "random_anchor_soh_difference": float(np.mean(np.abs(ref_rows["anchor_soh"].to_numpy(dtype=np.float32) - float(query_row["anchor_soh"])))),
        "random_recent_slope_difference": float(np.mean(np.abs(ref_rows["recent_soh_slope"].to_numpy(dtype=np.float32) - float(query_row["recent_soh_slope"])))),
        "random_qv_distance": float(np.mean([qv_map_distance(q_qv, retriever.arrays["qv_maps"][ref_idx, -1], q_qv_mask, retriever.arrays["qv_masks"][ref_idx, -1], retriever.qv_channel_weights) for ref_idx in ref_indices])),
        "random_physics_distance": float(np.mean([physics_feature_distance(q_phy, retriever.arrays["physics_features"][ref_idx, -1], q_phy_mask, retriever.arrays["physics_feature_masks"][ref_idx, -1]) for ref_idx in ref_indices])),
        "random_operation_distance": float(np.mean([operation_distance(q_op, retriever._operation_summary(ref_idx)) for ref_idx in ref_indices])),
        "random_future_delta_distance": float(np.mean([np.abs(retriever.arrays["future_delta"][ref_idx] - q_future).mean() for ref_idx in ref_indices])),
    }


def _random_rag_prediction(sampled_ids: np.ndarray, retriever: MultiStageBatteryRetriever) -> np.ndarray:
    sampled_ids = np.asarray(sampled_ids, dtype=np.int64)
    if sampled_ids.size == 0:
        return np.zeros(retriever.arrays["future_delta"].shape[1], dtype=np.float32)
    ref_future = retriever.arrays["future_delta"][sampled_ids]
    return np.asarray(ref_future.mean(axis=0), dtype=np.float32)


def _chemistry_filtered_rag_prediction(query_row: pd.Series, result, retriever: MultiStageBatteryRetriever) -> tuple[np.ndarray, bool]:
    valid = np.flatnonzero(result.retrieval_mask > 0)
    if valid.size == 0:
        return np.zeros(retriever.arrays["future_delta"].shape[1], dtype=np.float32), False
    keep = []
    for pos in valid.tolist():
        case_id = int(result.neighbor_case_ids[pos])
        ref_row = retriever.case_rows.iloc[retriever.case_id_to_index[case_id]]
        if str(ref_row["chemistry_family"]) == str(query_row["chemistry_family"]):
            keep.append(pos)
    if not keep:
        return rag_only_prediction(result), False
    keep = np.asarray(keep, dtype=np.int64)
    weights = np.asarray(result.retrieval_alpha[keep], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-6)
    pred = np.sum(result.neighbor_future_delta_soh[keep] * weights[:, None], axis=0)
    return np.asarray(pred, dtype=np.float32), True


def _ensure_case_bank(cfg: Dict[str, object]) -> Path:
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    if not (case_bank_dir / "case_rows.parquet").exists() and not (case_bank_dir / "case_rows.csv").exists():
        build_case_bank(cfg)
    return case_bank_dir


def validate_retrieval_quality(cfg: Dict[str, object], num_queries: int = 100) -> Dict[str, object]:
    case_bank_dir = _ensure_case_bank(cfg)
    parquet_path = case_bank_dir / "case_rows.parquet"
    csv_path = case_bank_dir / "case_rows.csv"
    if parquet_path.exists():
        try:
            rows = pd.read_parquet(parquet_path)
        except Exception:
            rows = pd.read_csv(csv_path)
    else:
        rows = pd.read_csv(csv_path)
    rows = rows.sort_values("case_id").reset_index(drop=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(cfg.get("experiments", {}).get("retrieval_validation", {}).get("output_dir", "output/experiments/retrieval_validation")) / run_id)
    figure_dir = ensure_dir(output_dir / "figures")
    try:
        import yaml

        (output_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        (output_dir / "run_config.yaml").write_text(json.dumps(cfg, indent=2, ensure_ascii=True))
    (output_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id, "case_bank_dir": str(case_bank_dir)}, indent=2, ensure_ascii=True))
    (output_dir / "experiment.log").write_text("retrieval validation started\n")

    retrieval_cfg = dict(cfg.get("retrieval", {}))
    retriever = MultiStageBatteryRetriever(
        case_bank_dir=case_bank_dir,
        db_splits=list(retrieval_cfg.get("db_splits", ["source_train"])),
        metric=str(retrieval_cfg.get("metric", "cosine")),
        stage1_embedding_name=str(retrieval_cfg.get("stage1_embedding", "tsfm")),
        top_m=int(retrieval_cfg.get("top_m", 200)),
        top_k=int(retrieval_cfg.get("top_k", 8)),
        rerank_weights=dict(retrieval_cfg.get("rerank_weights", {})),
        hard_filter={"same_cell_policy": retrieval_cfg.get("same_cell_policy", "exclude")},
        mmr={
            "use_mmr": bool(retrieval_cfg.get("use_mmr", True)),
            "mmr_lambda": float(retrieval_cfg.get("mmr_lambda", 0.75)),
            "max_neighbors_per_cell": int(retrieval_cfg.get("max_neighbors_per_cell", 2)),
        },
        retrieval_temperature=float(retrieval_cfg.get("retrieval_temperature", 0.1)),
        qv_channel_weights=dict(retrieval_cfg.get("qv_channel_weights", {})) if retrieval_cfg.get("qv_channel_weights") else None,
    )

    query_rows = rows[rows["split"].isin(["source_val", "target_query"])].copy()
    if len(query_rows) > num_queries:
        query_rows = query_rows.iloc[:num_queries].copy()

    diag_rows = []
    component_rows = []
    random_rows = []
    rag_rows = []
    method_predictions: Dict[str, List[np.ndarray]] = {"rag": [], "random_rag": [], "nearest1": [], "chemistry_filtered_rag": []}
    method_targets: Dict[str, List[np.ndarray]] = {"rag": [], "random_rag": [], "nearest1": [], "chemistry_filtered_rag": []}

    for _, query_row in query_rows.iterrows():
        case_id = int(query_row["case_id"])
        query_idx = retriever.case_id_to_index[case_id]
        result = retriever.retrieve(case_id)
        diag = summarize_retrieval_result(query_row, result, retriever)
        diag_rows.append(diag)

        valid = np.flatnonzero(result.retrieval_mask > 0)
        for rank, pos in enumerate(valid.tolist(), start=1):
            row = {"query_case_id": case_id, "neighbor_rank": rank}
            for comp_idx, comp_name in enumerate(result.explain["component_names"]):
                row[comp_name] = float(result.component_distances[pos, comp_idx])
            component_rows.append(row)

        candidates = retriever._hard_filter_candidate_ids(query_idx, retriever.db_case_ids)
        rng = np.random.RandomState(case_id)
        if len(candidates) > 0:
            sample_size = max(1, min(max(len(valid), 1), len(candidates)))
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            query_future = retriever.arrays["future_delta"][query_idx]
            rag_pred = rag_only_prediction(result)
            random_rag_pred = _random_rag_prediction(np.asarray(sampled, dtype=np.int64), retriever)
            nearest1_pred = nearest1_prediction(result)
            chemistry_rag_pred, chemistry_rag_available = _chemistry_filtered_rag_prediction(query_row, result, retriever)
            retrieved_summary = _retrieved_distance_summary(query_idx, result, retriever)
            random_summary = _random_distance_summary(query_idx, np.asarray(sampled, dtype=np.int64), retriever)
            rag_rows.append(
                {
                    "query_case_id": case_id,
                    "split": query_row["split"],
                    "chemistry_family": query_row["chemistry_family"],
                    "domain_label": query_row["domain_label"],
                    "rag_mae": float(np.abs(rag_pred - query_future).mean()),
                    "random_rag_mae": float(np.abs(random_rag_pred - query_future).mean()),
                    "nearest1_mae": float(np.abs(nearest1_pred - query_future).mean()),
                    "chemistry_filtered_rag_mae": float(np.abs(chemistry_rag_pred - query_future).mean()),
                    "rag_last_mae": float(np.abs(rag_pred[-1] - query_future[-1])),
                    "random_rag_last_mae": float(np.abs(random_rag_pred[-1] - query_future[-1])),
                    "nearest1_last_mae": float(np.abs(nearest1_pred[-1] - query_future[-1])),
                    "chemistry_filtered_rag_last_mae": float(np.abs(chemistry_rag_pred[-1] - query_future[-1])),
                    "chemistry_filtered_available": bool(chemistry_rag_available),
                }
            )
            method_predictions["rag"].append(rag_pred)
            method_targets["rag"].append(query_future)
            method_predictions["random_rag"].append(random_rag_pred)
            method_targets["random_rag"].append(query_future)
            method_predictions["nearest1"].append(nearest1_pred)
            method_targets["nearest1"].append(query_future)
            if chemistry_rag_available:
                method_predictions["chemistry_filtered_rag"].append(chemistry_rag_pred)
                method_targets["chemistry_filtered_rag"].append(query_future)
            random_rows.append(
                {
                    "query_case_id": case_id,
                    "query_split": query_row["split"],
                    "query_chemistry": query_row["chemistry_family"],
                    "query_domain": query_row["domain_label"],
                    "retrieved_rag_last_horizon_error": float(np.abs(rag_pred[-1] - query_future[-1])),
                    "random_rag_last_horizon_error": float(np.abs(random_rag_pred[-1] - query_future[-1])),
                    **retrieved_summary,
                    **random_summary,
                }
            )

        if len(diag_rows) <= int(cfg.get("experiments", {}).get("retrieval_validation", {}).get("num_plot_queries", 20)):
            query_dir = ensure_dir(figure_dir / f"query_{case_id}")
            save_soh_history_figure(query_idx, query_row, result, retriever, query_dir / "soh_history_and_topk_future.png")
            save_qv_overlay(query_idx, result, retriever, query_dir / "qv_map_overlay.png")
            save_partial_relax_overlay(query_idx, result, retriever, query_dir / "partial_charge_relaxation_overlay.png")
            save_component_distance_bar(result, query_dir / "retrieval_component_distance_bar.png")
            save_metadata_table(query_row, result, retriever, query_dir / "retrieval_metadata_table.csv")

    diagnostics_df = pd.DataFrame(diag_rows)
    diagnostics_df.to_csv(output_dir / "retrieval_diagnostics.csv", index=False)
    pd.DataFrame(component_rows).to_csv(output_dir / "retrieval_component_distances.csv", index=False)
    diagnostics_df.groupby("split").mean(numeric_only=True).reset_index().to_csv(output_dir / "retrieval_summary_by_split.csv", index=False)
    diagnostics_df.groupby("chemistry_family").mean(numeric_only=True).reset_index().to_csv(output_dir / "retrieval_summary_by_chemistry.csv", index=False)
    diagnostics_df.groupby("domain_label").mean(numeric_only=True).reset_index().to_csv(output_dir / "retrieval_summary_by_domain.csv", index=False)

    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(output_dir / "retrieval_vs_random_metrics.csv", index=False)
    rag_df = pd.DataFrame(rag_rows)
    rag_df.to_csv(output_dir / "rag_only_metrics.csv", index=False)
    if len(rag_df):
        horizon_rows = []
        for method, preds in method_predictions.items():
            if not preds:
                continue
            pred_arr = np.stack(preds).astype(np.float32)
            target_arr = np.stack(method_targets[method]).astype(np.float32)
            horizon_mae = np.abs(pred_arr - target_arr).mean(axis=0)
            for horizon_idx, value in enumerate(horizon_mae.tolist(), start=1):
                horizon_rows.append({"method": method, "horizon": horizon_idx, "mae": float(value)})
        horizon_frame = pd.DataFrame(horizon_rows)
        horizon_frame.to_csv(output_dir / "rag_only_metrics_by_horizon.csv", index=False)

        method_metric_cols = {
            "rag": "rag_mae",
            "random_rag": "random_rag_mae",
            "nearest1": "nearest1_mae",
            "chemistry_filtered_rag": "chemistry_filtered_rag_mae",
        }
        chemistry_rows = []
        domain_rows = []
        for method, metric_col in method_metric_cols.items():
            if metric_col not in rag_df.columns:
                continue
            chem_group = rag_df.groupby("chemistry_family")[metric_col].mean()
            chemistry_rows.extend(
                [{"chemistry_family": str(name), "method": method, "mae": float(value)} for name, value in chem_group.items()]
            )
            domain_group = rag_df.groupby("domain_label")[metric_col].mean()
            domain_rows.extend(
                [{"domain_label": str(name), "method": method, "mae": float(value)} for name, value in domain_group.items()]
            )
        pd.DataFrame(chemistry_rows).to_csv(output_dir / "rag_only_metrics_by_chemistry.csv", index=False)
        pd.DataFrame(domain_rows).to_csv(output_dir / "rag_only_metrics_by_domain.csv", index=False)
    else:
        pd.DataFrame().to_csv(output_dir / "rag_only_metrics_by_horizon.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "rag_only_metrics_by_chemistry.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "rag_only_metrics_by_domain.csv", index=False)

    if len(diagnostics_df):
        split_conf = diagnostics_df.groupby("split")["retrieval_confidence"].mean().to_dict()
        save_bar({str(k): float(v) for k, v in split_conf.items()}, "Retrieval confidence by split", "Confidence", figure_dir / "retrieval_vs_random_distance_boxplot.png")
        grouped_chem = {str(name): values["topk_mean_composite_distance"].tolist() for name, values in diagnostics_df.groupby("chemistry_family")}
        grouped_domain = {str(name): values["topk_mean_composite_distance"].tolist() for name, values in diagnostics_df.groupby("domain_label")}
        save_boxplot(grouped_chem, "Retrieval quality by chemistry", "Composite distance", figure_dir / "retrieval_vs_random_by_chemistry.png")
        save_boxplot(grouped_domain, "Retrieval quality by domain", "Composite distance", figure_dir / "retrieval_vs_random_by_domain.png")
        if len(rag_df):
            save_bar(
                {
                    "rag": float(rag_df["rag_mae"].mean()),
                    "random_rag": float(rag_df["random_rag_mae"].mean()),
                    "nearest1": float(rag_df["nearest1_mae"].mean()),
                    "chemistry_filtered_rag": float(rag_df["chemistry_filtered_rag_mae"].mean()),
                },
                "RAG-only vs baselines MAE",
                "MAE",
                figure_dir / "rag_only_vs_random_mae.png",
            )
            if len(horizon_frame):
                horizon_pivot = horizon_frame.pivot(index="method", columns="horizon", values="mae").fillna(0.0)
                save_heatmap(
                    horizon_pivot.to_numpy(dtype=np.float32),
                    horizon_pivot.index.tolist(),
                    [f"h{int(col)}" for col in horizon_pivot.columns.tolist()],
                    "RAG-only horizon-wise MAE",
                    figure_dir / "rag_only_horizon_wise_mae.png",
                    cmap="magma",
                )
            save_boxplot(
                {
                    "rag_last": rag_df["rag_last_mae"].tolist(),
                    "random_rag_last": rag_df["random_rag_last_mae"].tolist(),
                    "nearest1_last": rag_df["nearest1_last_mae"].tolist(),
                    "chem_filtered_last": rag_df["chemistry_filtered_rag_last_mae"].tolist(),
                },
                "Retrieved vs random future-delta error",
                "Last-horizon MAE",
                figure_dir / "retrieval_vs_random_future_delta_error.png",
            )
        if len(random_df):
            save_boxplot(
                {
                    "retrieved_qv": random_df["retrieved_qv_distance"].tolist(),
                    "random_qv": random_df["random_qv_distance"].tolist(),
                    "retrieved_physics": random_df["retrieved_physics_distance"].tolist(),
                    "random_physics": random_df["random_physics_distance"].tolist(),
                },
                "Retrieved vs random distance comparison",
                "Distance",
                figure_dir / "retrieval_vs_random_distance_boxplot.png",
            )

    weak_chem = diagnostics_df.groupby("chemistry_family")["topk_mean_composite_distance"].mean().sort_values(ascending=False).head(3).index.tolist() if len(diagnostics_df) else []
    weak_domain = diagnostics_df.groupby("domain_label")["topk_mean_composite_distance"].mean().sort_values(ascending=False).head(3).index.tolist() if len(diagnostics_df) else []
    discriminative_components = (
        pd.DataFrame(component_rows).drop(columns=["query_case_id", "neighbor_rank"]).mean().sort_values(ascending=False).head(3).index.tolist()
        if component_rows
        else []
    )
    summary_lines = [
        "# Summary",
        f"- 检索 query 数: {len(query_rows)}",
        f"- 检索是否优于 random baseline: {bool(len(random_df) > 0 and random_df['retrieved_qv_distance'].mean() < random_df['random_qv_distance'].mean() and random_df['retrieved_physics_distance'].mean() < random_df['random_physics_distance'].mean())}",
        f"- 区分度较强的 component distance: {discriminative_components}",
        f"- 检索较弱的 chemistry: {weak_chem}",
        f"- 检索较弱的 domain: {weak_domain}",
        f"- RAG-only 是否具有预测参考价值: {bool(len(rag_df) > 0 and float(rag_df['rag_mae'].mean()) < float(rag_df['random_rag_mae'].mean()))}",
        "- top-k 检索案例可视化已导出到 figures/query_<case_id>/",
        "- rerank weights 建议: 若 random_qv / random_physics 仍接近 retrieved，可提高 qv_map 或 physics_features 权重。",
        f"- chemistry-filtered RAG 可用比例: {float(rag_df['chemistry_filtered_available'].mean()) if len(rag_df) else 0.0}",
        f"- retrieval_confidence 平均值: {float(diagnostics_df['retrieval_confidence'].mean()) if len(diagnostics_df) else 0.0}",
        f"- same cell rate 平均值: {float(diagnostics_df['topk_same_cell_rate'].mean()) if len(diagnostics_df) else 0.0}",
        "- target_query 仅作为 query 诊断，未进入 DB",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))
    return {"output_dir": str(output_dir), "run_id": run_id}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval quality for battery SOH")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=100)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = validate_retrieval_quality(cfg, num_queries=args.num_queries)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
