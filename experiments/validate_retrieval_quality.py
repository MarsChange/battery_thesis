"""experiments.validate_retrieval_quality

运行 RAG 检索质量验证实验，检查 top-k 历史案例是否比随机案例更相似、
是否能为 `future_delta_soh` 提供更好的数值参考，并输出命名距离分量。
"""

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
from retrieval.multistage_retriever import COMPONENT_NAMES, MultiStageBatteryRetriever
from retrieval.retrieval_diagnostics import (
    nearest1_prediction,
    persistence_delta_prediction,
    rag_only_prediction,
    save_component_distance_bar,
    save_metadata_table,
    save_partial_relax_overlay,
    save_qv_overlay,
    save_soh_history_figure,
    summarize_retrieval_result,
)


def _delta_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    error = np.asarray(pred, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error * error)))
    last_horizon_mae = float(np.mean(np.abs(error[:, -1]))) if error.ndim == 2 and error.shape[1] > 0 else 0.0
    return {
        "MAE": mae,
        "RMSE": rmse,
        "last_horizon_MAE": last_horizon_mae,
    }


def _ensure_case_bank(cfg: Dict[str, object]) -> Path:
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    if not (case_bank_dir / "case_rows.parquet").exists() and not (case_bank_dir / "case_rows.csv").exists():
        build_case_bank(cfg)
    return case_bank_dir


def _component_row(
    query_row: pd.Series,
    neighbor_rank: int,
    neighbor_row: pd.Series,
    bundle: Dict[str, float],
    retrieval_confidence: float,
) -> Dict[str, object]:
    return {
        "query_case_id": int(query_row["case_id"]),
        "neighbor_rank": int(neighbor_rank),
        "neighbor_case_id": int(neighbor_row["case_id"]),
        "query_cell_uid": str(query_row["cell_uid"]),
        "neighbor_cell_uid": str(neighbor_row["cell_uid"]),
        "query_chemistry_family": str(query_row["chemistry_family"]),
        "neighbor_chemistry_family": str(neighbor_row["chemistry_family"]),
        "query_domain_label": str(query_row["domain_label"]),
        "neighbor_domain_label": str(neighbor_row["domain_label"]),
        "d_soh_state": float(bundle["d_soh_state"]),
        "d_qv_shape": float(bundle["d_qv_shape"]),
        "d_physics": float(bundle["d_physics"]),
        "d_operation": float(bundle["d_operation"]),
        "d_metadata": float(bundle["d_metadata"]),
        "d_tsfm": float(bundle["d_tsfm"]),
        "composite_distance": float(bundle["composite_distance"]),
        "retrieval_confidence": float(retrieval_confidence),
    }


def _mean_bundle(bundles: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    if not bundles:
        return {
            f"{prefix}_mean_d_soh_state": np.nan,
            f"{prefix}_mean_d_qv_shape": np.nan,
            f"{prefix}_mean_d_physics": np.nan,
            f"{prefix}_mean_d_operation": np.nan,
            f"{prefix}_mean_d_metadata": np.nan,
            f"{prefix}_mean_d_tsfm": np.nan,
            f"{prefix}_mean_composite_distance": np.nan,
        }
    frame = pd.DataFrame(bundles)
    return {
        f"{prefix}_mean_d_soh_state": float(frame["d_soh_state"].mean()),
        f"{prefix}_mean_d_qv_shape": float(frame["d_qv_shape"].mean()),
        f"{prefix}_mean_d_physics": float(frame["d_physics"].mean()),
        f"{prefix}_mean_d_operation": float(frame["d_operation"].mean()),
        f"{prefix}_mean_d_metadata": float(frame["d_metadata"].mean()),
        f"{prefix}_mean_d_tsfm": float(frame["d_tsfm"].mean()),
        f"{prefix}_mean_composite_distance": float(frame["composite_distance"].mean()),
    }


def _random_rag_prediction(sampled_case_ids: np.ndarray, retriever: MultiStageBatteryRetriever) -> np.ndarray:
    sampled_case_ids = np.asarray(sampled_case_ids, dtype=np.int64)
    if sampled_case_ids.size == 0:
        return np.zeros(retriever.arrays["future_delta"].shape[1], dtype=np.float32)
    row_indices = [retriever.case_id_to_index[int(case_id)] for case_id in sampled_case_ids.tolist()]
    return np.asarray(retriever.arrays["future_delta"][row_indices].mean(axis=0), dtype=np.float32)


def _chemistry_filtered_prediction(query_row: pd.Series, result, retriever: MultiStageBatteryRetriever) -> tuple[np.ndarray, bool]:
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


def validate_retrieval_quality(cfg: Dict[str, object], num_queries: int = 100) -> Dict[str, object]:
    case_bank_dir = _ensure_case_bank(cfg)
    if (case_bank_dir / "case_rows.parquet").exists():
        try:
            rows = pd.read_parquet(case_bank_dir / "case_rows.parquet")
        except Exception:
            rows = pd.read_csv(case_bank_dir / "case_rows.csv")
    else:
        rows = pd.read_csv(case_bank_dir / "case_rows.csv")
    rows = rows.sort_values("case_id").reset_index(drop=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(cfg.get("experiments", {}).get("retrieval_validation", {}).get("output_dir", "output/experiments/retrieval_validation"))
    output_dir = ensure_dir(output_root / run_id)
    figure_dir = ensure_dir(output_dir / "figures")

    retrieval_cfg = dict(cfg.get("retrieval", {}))
    retrieval_config_path = str(retrieval_cfg.get("retrieval_feature_config_path", "configs/rag_retrieval_features.yaml"))
    retriever = MultiStageBatteryRetriever(
        case_bank_dir=case_bank_dir,
        retrieval_config_path=retrieval_config_path,
        db_splits=list(retrieval_cfg.get("db_splits", ["source_train"])),
        query_splits=["source_val", "target_query"],
        top_m=int(retrieval_cfg.get("top_m", 200)),
        top_k=int(retrieval_cfg.get("top_k", 8)),
        same_cell_policy=str(retrieval_cfg.get("same_cell_policy", "exclude")),
        allow_cross_chemistry=bool(retrieval_cfg.get("allow_cross_chemistry", True)),
        metric=str(retrieval_cfg.get("metric", "cosine")),
        stage1_embedding_name=str(retrieval_cfg.get("stage1_embedding", "tsfm")),
        retrieval_temperature=float(retrieval_cfg.get("retrieval_temperature", 0.1)),
        mmr={
            "use_mmr": bool(retrieval_cfg.get("use_mmr", True)),
            "mmr_lambda": float(retrieval_cfg.get("mmr_lambda", 0.75)),
            "max_neighbors_per_cell": int(retrieval_cfg.get("max_neighbors_per_cell", 2)),
            "max_neighbors_per_domain": retrieval_cfg.get("max_neighbors_per_domain"),
        },
    )

    try:
        import yaml

        (output_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        if bool(retriever.rag_config.get("logging", {}).get("save_yaml_snapshot_with_each_run", True)):
            (output_dir / "rag_retrieval_features.yaml").write_text(Path(retrieval_config_path).read_text())
    except Exception:
        (output_dir / "run_config.yaml").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "case_bank_dir": str(case_bank_dir),
                "retrieval_config_path": retrieval_config_path,
                "db_splits": retriever.db_splits,
                "query_splits": retriever.query_splits,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    (output_dir / "experiment.log").write_text("retrieval validation started\n", encoding="utf-8")

    query_rows = rows[rows["split"].isin(["source_val", "target_query"])].copy()
    if len(query_rows) > num_queries:
        query_rows = query_rows.iloc[:num_queries].copy()

    component_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    random_compare_rows: List[Dict[str, object]] = []
    method_predictions: Dict[str, List[np.ndarray]] = {
        "rag_only": [],
        "nearest1": [],
        "random_rag": [],
        "persistence": [],
        "chemistry_filtered_rag": [],
    }
    method_targets: Dict[str, List[np.ndarray]] = {name: [] for name in method_predictions}
    method_query_ids: Dict[str, List[int]] = {name: [] for name in method_predictions}

    num_plot_queries = int(cfg.get("experiments", {}).get("retrieval_validation", {}).get("num_plot_queries", 20))

    for plot_rank, (_, query_row) in enumerate(query_rows.iterrows(), start=1):
        case_id = int(query_row["case_id"])
        query_idx = retriever.case_id_to_index[case_id]
        result = retriever.retrieve(case_id)
        query_summary = summarize_retrieval_result(query_row, result, retriever)
        summary_rows.append(query_summary)

        valid = np.flatnonzero(result.retrieval_mask > 0)
        retrieved_bundles = []
        for neighbor_rank, pos in enumerate(valid.tolist(), start=1):
            neighbor_case_id = int(result.neighbor_case_ids[pos])
            neighbor_row = retriever.case_rows.iloc[retriever.case_id_to_index[neighbor_case_id]]
            bundle = {
                "d_soh_state": float(result.d_soh_state[pos]),
                "d_qv_shape": float(result.d_qv_shape[pos]),
                "d_physics": float(result.d_physics[pos]),
                "d_operation": float(result.d_operation[pos]),
                "d_metadata": float(result.d_metadata[pos]),
                "d_tsfm": float(result.d_tsfm[pos]),
                "composite_distance": float(result.composite_distance[pos]),
            }
            retrieved_bundles.append(bundle)
            component_rows.append(_component_row(query_row, neighbor_rank, neighbor_row, bundle, result.retrieval_confidence))

        candidate_case_ids = retriever._hard_filter_candidate_ids(query_idx, retriever.db_case_ids)
        rng = np.random.RandomState(case_id)
        sample_size = max(1, min(len(valid) if len(valid) else retriever.top_k, len(candidate_case_ids))) if len(candidate_case_ids) else 0
        random_case_ids = rng.choice(candidate_case_ids, size=sample_size, replace=False) if sample_size > 0 else np.zeros(0, dtype=np.int64)
        random_bundles = []
        for random_case_id in np.asarray(random_case_ids, dtype=np.int64).tolist():
            ref_idx = retriever.case_id_to_index[int(random_case_id)]
            random_bundles.append(retriever._distance_bundle(query_idx, ref_idx))

        target_delta = np.asarray(retriever.arrays["future_delta"][query_idx], dtype=np.float32)
        rag_pred = rag_only_prediction(result)
        nearest1_pred = nearest1_prediction(result)
        random_rag_pred = _random_rag_prediction(random_case_ids, retriever)
        persistence_pred = persistence_delta_prediction(target_delta.shape[0])
        chemistry_pred, chemistry_available = _chemistry_filtered_prediction(query_row, result, retriever)

        for method_name, prediction, available in [
            ("rag_only", rag_pred, True),
            ("nearest1", nearest1_pred, True),
            ("random_rag", random_rag_pred, True),
            ("persistence", persistence_pred, True),
            ("chemistry_filtered_rag", chemistry_pred, chemistry_available),
        ]:
            if not available:
                continue
            method_predictions[method_name].append(np.asarray(prediction, dtype=np.float32))
            method_targets[method_name].append(target_delta)
            method_query_ids[method_name].append(case_id)

        random_compare_rows.append(
            {
                "query_case_id": case_id,
                "split": str(query_row["split"]),
                "chemistry_family": str(query_row["chemistry_family"]),
                "domain_label": str(query_row["domain_label"]),
                **_mean_bundle(retrieved_bundles, "retrieved"),
                **_mean_bundle(random_bundles, "random"),
                "retrieved_rag_only_mae": float(np.mean(np.abs(rag_pred - target_delta))),
                "random_rag_only_mae": float(np.mean(np.abs(random_rag_pred - target_delta))),
            }
        )

        if plot_rank <= num_plot_queries:
            query_dir = ensure_dir(figure_dir / f"query_{case_id}")
            save_soh_history_figure(query_idx, query_row, result, retriever, query_dir / "soh_history_and_topk_future.png")
            save_qv_overlay(query_idx, result, retriever, query_dir / "qv_map_overlay.png")
            save_partial_relax_overlay(query_idx, result, retriever, query_dir / "partial_charge_relaxation_overlay.png")
            save_component_distance_bar(result, query_dir / "retrieval_component_distance_bar.png")
            save_metadata_table(query_row, result, retriever, query_dir / "retrieval_metadata_table.csv")

    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(output_dir / "retrieval_component_distances.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "retrieval_summary_by_query.csv", index=False)
    summary_df.to_csv(output_dir / "retrieval_diagnostics.csv", index=False)
    summary_df.groupby("split").mean(numeric_only=True).reset_index().to_csv(output_dir / "retrieval_summary_by_split.csv", index=False)
    summary_df.groupby("chemistry_family").mean(numeric_only=True).reset_index().to_csv(output_dir / "retrieval_summary_by_chemistry.csv", index=False)
    summary_df.groupby("domain_label").mean(numeric_only=True).reset_index().to_csv(output_dir / "retrieval_summary_by_domain.csv", index=False)

    compare_df = pd.DataFrame(random_compare_rows)
    compare_df.to_csv(output_dir / "retrieval_vs_random_metrics.csv", index=False)

    rag_metric_rows = []
    horizon_rows = []
    chemistry_rows = []
    domain_rows = []
    for method_name, preds in method_predictions.items():
        if not preds:
            continue
        pred_arr = np.stack(preds).astype(np.float32)
        target_arr = np.stack(method_targets[method_name]).astype(np.float32)
        metrics = _delta_metrics(pred_arr, target_arr)
        rag_metric_rows.append(
            {
                "method": method_name,
                **metrics,
                "num_queries": int(pred_arr.shape[0]),
            }
        )
        horizon_mae = np.abs(pred_arr - target_arr).mean(axis=0)
        for horizon_idx, value in enumerate(horizon_mae.tolist(), start=1):
            horizon_rows.append({"method": method_name, "horizon": horizon_idx, "horizon_wise_MAE": float(value)})

    rag_metrics_df = pd.DataFrame(rag_metric_rows)
    rag_metrics_df.to_csv(output_dir / "rag_only_metrics.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(output_dir / "rag_only_metrics_by_horizon.csv", index=False)

    if len(query_rows):
        query_meta = query_rows[["case_id", "chemistry_family", "domain_label"]].rename(columns={"case_id": "query_case_id"})
        for method_name, preds in method_predictions.items():
            if not preds:
                continue
            pred_arr = np.stack(preds).astype(np.float32)
            target_arr = np.stack(method_targets[method_name]).astype(np.float32)
            case_ids = np.asarray(method_query_ids[method_name], dtype=np.int64)
            method_frame = pd.DataFrame(
                {
                    "query_case_id": case_ids,
                    "method": method_name,
                    "mae": np.abs(pred_arr - target_arr).mean(axis=1),
                }
            ).merge(query_meta, on="query_case_id", how="left")
            chemistry_rows.extend(
                method_frame.groupby("chemistry_family")["mae"].mean().reset_index().assign(method=method_name).to_dict(orient="records")
            )
            domain_rows.extend(
                method_frame.groupby("domain_label")["mae"].mean().reset_index().assign(method=method_name).to_dict(orient="records")
            )
    pd.DataFrame(chemistry_rows).to_csv(output_dir / "rag_only_metrics_by_chemistry.csv", index=False)
    pd.DataFrame(domain_rows).to_csv(output_dir / "rag_only_metrics_by_domain.csv", index=False)

    if len(compare_df):
        save_boxplot(
            {
                "retrieved_composite": compare_df["retrieved_mean_composite_distance"].tolist(),
                "random_composite": compare_df["random_mean_composite_distance"].tolist(),
                "retrieved_qv": compare_df["retrieved_mean_d_qv_shape"].tolist(),
                "random_qv": compare_df["random_mean_d_qv_shape"].tolist(),
            },
            "Retrieved top-k vs random top-k distance",
            "Distance (lower is better)",
            figure_dir / "retrieval_vs_random_distance_boxplot.png",
        )
        save_boxplot(
            {
                "retrieved_rag_only_mae": compare_df["retrieved_rag_only_mae"].tolist(),
                "random_rag_only_mae": compare_df["random_rag_only_mae"].tolist(),
            },
            "RAG-only prediction error vs random references",
            "MAE",
            figure_dir / "retrieval_vs_random_future_delta_error.png",
        )
        save_boxplot(
            {str(name): sub["retrieved_mean_composite_distance"].tolist() for name, sub in compare_df.groupby("chemistry_family")},
            "Retrieved top-k quality by chemistry",
            "Mean composite distance",
            figure_dir / "retrieval_vs_random_by_chemistry.png",
        )
        save_boxplot(
            {str(name): sub["retrieved_mean_composite_distance"].tolist() for name, sub in compare_df.groupby("domain_label")},
            "Retrieved top-k quality by domain",
            "Mean composite distance",
            figure_dir / "retrieval_vs_random_by_domain.png",
        )

    if len(rag_metrics_df):
        save_bar(
            {str(row["method"]): float(row["MAE"]) for _, row in rag_metrics_df.iterrows()},
            "RAG-only and baseline MAE",
            "MAE",
            figure_dir / "rag_only_vs_random_mae.png",
        )
    horizon_df = pd.DataFrame(horizon_rows)
    if len(horizon_df):
        pivot = horizon_df.pivot(index="method", columns="horizon", values="horizon_wise_MAE").fillna(0.0)
        save_heatmap(
            pivot.to_numpy(dtype=np.float32),
            pivot.index.tolist(),
            [f"h{int(col)}" for col in pivot.columns.tolist()],
            "RAG-only horizon-wise MAE",
            figure_dir / "rag_only_horizon_wise_mae.png",
            cmap="magma",
        )

    enabled_components = [
        name
        for name in COMPONENT_NAMES
        if bool(dict(retriever.rag_config.get("distance_components", {}).get(name, {}) or {}).get("enabled", False))
    ]
    enabled_feature_summary = {}
    for component_name in enabled_components:
        features_cfg = dict(retriever.rag_config.get("distance_components", {}).get(component_name, {}).get("features", {}) or {})
        enabled_feature_summary[component_name] = [
            feature_name
            for feature_name, feature_cfg in features_cfg.items()
            if bool(dict(feature_cfg or {}).get("enabled", False))
        ]
    d_tsfm_cfg = dict(retriever.rag_config.get("distance_components", {}).get("d_tsfm", {}) or {})
    d_tsfm_features = dict(d_tsfm_cfg.get("features", {}) or {})
    d_tsfm_enabled = bool(d_tsfm_cfg.get("enabled", False))
    coarse_tsfm = bool(dict(d_tsfm_features.get("tsfm_stage1_coarse_retrieval", {}) or {}).get("enabled", True))
    rerank_tsfm = bool(dict(d_tsfm_features.get("tsfm_in_final_rerank", {}) or {}).get("enabled", True))
    retrieved_better_than_random = bool(
        len(compare_df)
        and float(compare_df["retrieved_mean_composite_distance"].mean()) < float(compare_df["random_mean_composite_distance"].mean())
    )
    rag_better_than_random = bool(
        len(compare_df)
        and float(compare_df["retrieved_rag_only_mae"].mean()) < float(compare_df["random_rag_only_mae"].mean())
    )
    rag_better_than_persistence = False
    if len(rag_metrics_df) and "persistence" in rag_metrics_df["method"].tolist() and "rag_only" in rag_metrics_df["method"].tolist():
        rag_better_than_persistence = float(rag_metrics_df.loc[rag_metrics_df["method"] == "rag_only", "MAE"].iloc[0]) < float(
            rag_metrics_df.loc[rag_metrics_df["method"] == "persistence", "MAE"].iloc[0]
        )

    weak_chemistry = (
        summary_df.groupby("chemistry_family")["topk_mean_composite_distance"].mean().sort_values(ascending=False).head(3).index.tolist()
        if len(summary_df)
        else []
    )
    weak_domain = (
        summary_df.groupby("domain_label")["topk_mean_composite_distance"].mean().sort_values(ascending=False).head(3).index.tolist()
        if len(summary_df)
        else []
    )
    suggestion = "维持当前权重。"
    if len(compare_df):
        if float(compare_df["retrieved_mean_d_qv_shape"].mean()) >= float(compare_df["random_mean_d_qv_shape"].mean()):
            suggestion = "提高 d_qv_shape 权重或增强 qv 曲线摘要特征。"
        elif float(compare_df["retrieved_mean_d_physics"].mean()) >= float(compare_df["random_mean_d_physics"].mean()):
            suggestion = "提高 d_physics 权重或检查 partial charge / relaxation 特征可用率。"

    summary_lines = [
        "# 检索质量验证总结",
        "",
        f"- 当前启用的 RAG 检索特征分量：{enabled_components}",
        f"- 当前启用的底层检索特征：{enabled_feature_summary}",
        f"- composite_distance 由以下距离分量组成：{enabled_components}",
        f"- d_tsfm 是否启用：{d_tsfm_enabled}",
        f"- d_tsfm 是否用于 Stage-1 粗检索：{coarse_tsfm}",
        f"- d_tsfm 是否参与最终 rerank：{rerank_tsfm}",
        f"- retrieved top-k 是否比 random top-k 更相似：{retrieved_better_than_random}",
        f"- RAG-only 是否优于 random RAG：{rag_better_than_random}",
        f"- RAG-only 是否优于 persistence baseline：{rag_better_than_persistence}",
        f"- 检索较弱的 chemistry：{weak_chemistry}",
        f"- 检索较弱的 domain：{weak_domain}",
        f"- 是否建议调整权重：{suggestion}",
        "- target_query 仅作为 query 参与评估，不进入 reference database。",
        "- 每个 top-k neighbor 的命名距离分量已保存到 retrieval_component_distances.csv。",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return {"output_dir": str(output_dir), "run_id": run_id}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate retrieval quality for battery SOH forecasting")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_queries", type=int, default=100)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = validate_retrieval_quality(cfg, num_queries=args.num_queries)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
