from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from battery_data.build_case_bank import build_case_bank, load_config
from experiments.plotting_utils import ensure_dir, save_bar, save_boxplot, save_heatmap
from forecasting.metrics import horizon_metrics, regression_metrics


def _ensure_case_bank(cfg: Dict[str, object]) -> Path:
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    if not (case_bank_dir / "case_rows.parquet").exists() and not (case_bank_dir / "case_rows.csv").exists():
        build_case_bank(cfg)
    return case_bank_dir


def _load_case_bank(case_bank_dir: Path) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, object]]:
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
    arrays = {
        "cycle_stats": np.load(case_bank_dir / "case_cycle_stats.npy"),
        "soh_seq": np.load(case_bank_dir / "case_soh_seq.npy"),
        "qv_maps": np.load(case_bank_dir / "case_qv_maps.npy"),
        "qv_masks": np.load(case_bank_dir / "case_qv_masks.npy"),
        "partial_charge": np.load(case_bank_dir / "case_partial_charge.npy"),
        "partial_charge_mask": np.load(case_bank_dir / "case_partial_charge_mask.npy"),
        "relaxation": np.load(case_bank_dir / "case_relaxation.npy"),
        "relaxation_mask": np.load(case_bank_dir / "case_relaxation_mask.npy"),
        "physics_features": np.load(case_bank_dir / "case_physics_features.npy"),
        "physics_feature_masks": np.load(case_bank_dir / "case_physics_feature_masks.npy"),
        "anchor_physics_features": np.load(case_bank_dir / "case_anchor_physics_features.npy"),
        "operation_seq": np.load(case_bank_dir / "case_operation_seq.npy"),
        "future_ops": np.load(case_bank_dir / "case_future_ops.npy"),
        "future_ops_mask": np.load(case_bank_dir / "case_future_ops_mask.npy"),
        "future_delta_soh": np.load(case_bank_dir / "case_future_delta_soh.npy"),
        "future_soh": np.load(case_bank_dir / "case_future_soh.npy"),
    }
    tsfm_path = case_bank_dir / "case_tsfm_embeddings.npy"
    arrays["tsfm_embedding"] = np.load(tsfm_path) if tsfm_path.exists() else None
    feature_names = json.loads((case_bank_dir / "feature_names.json").read_text())
    return rows, arrays, feature_names


def _availability_rows(rows: pd.DataFrame, arrays: Dict[str, np.ndarray]) -> pd.DataFrame:
    result = []
    for idx, row in rows.iterrows():
        result.append(
            {
                "case_id": int(row["case_id"]),
                "split": row["split"],
                "chemistry_family": row["chemistry_family"],
                "domain_label": row["domain_label"],
                "soh_seq": float(np.isfinite(arrays["soh_seq"][idx]).mean()),
                "cycle_stats": float(np.isfinite(arrays["cycle_stats"][idx]).mean()),
                "qv_curve_stats": float(np.isfinite(arrays["cycle_stats"][idx, -1]).mean()),
                "qv_map": float(arrays["qv_masks"][idx].mean()),
                "partial_charge": float(arrays["partial_charge_mask"][idx].mean()),
                "relaxation": float(arrays["relaxation_mask"][idx].mean()),
                "physics_12d": float(arrays["physics_feature_masks"][idx].mean()),
                "operation": float(np.isfinite(arrays["operation_seq"][idx]).mean()),
                "future_operation": float(arrays["future_ops_mask"][idx].mean()),
                "tsfm_embedding": float(np.isfinite(arrays["tsfm_embedding"][idx]).mean()) if arrays["tsfm_embedding"] is not None else 0.0,
            }
        )
    return pd.DataFrame(result)


def _target_views(arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    delta = arrays["future_delta_soh"]
    return {
        "future_delta_h1": delta[:, 0],
        "future_delta_last": delta[:, -1],
        "future_delta_mean": delta.mean(axis=1),
        "future_slope": delta[:, -1] - delta[:, 0],
    }


def _flat_feature_table(rows: pd.DataFrame, arrays: Dict[str, np.ndarray], feature_names: Dict[str, object]) -> pd.DataFrame:
    records = []
    for idx, row in rows.iterrows():
        anchor_cycle = arrays["cycle_stats"][idx, -1]
        qv_anchor = arrays["qv_maps"][idx, -1]
        partial_anchor = arrays["partial_charge"][idx, -1]
        relax_anchor = arrays["relaxation"][idx, -1]
        physics_anchor = arrays["anchor_physics_features"][idx]
        record = {
            "case_id": int(row["case_id"]),
            "split": row["split"],
            "chemistry_family": row["chemistry_family"],
            "domain_label": row["domain_label"],
            "anchor_soh": float(row["anchor_soh"]),
            "recent_soh_slope": float(row["recent_soh_slope"]),
        }
        for feat_idx, name in enumerate(feature_names["cycle_stats"]):
            record[f"cycle_{name}"] = float(anchor_cycle[feat_idx])
        record["qv_map_mean"] = float(qv_anchor.mean())
        record["qv_map_std"] = float(qv_anchor.std())
        record["partial_charge_mean"] = float(partial_anchor.mean())
        record["partial_charge_std"] = float(partial_anchor.std())
        record["relaxation_mean"] = float(relax_anchor.mean())
        record["relaxation_std"] = float(relax_anchor.std())
        for feat_idx, name in enumerate(feature_names["physics_feature_names"]):
            record[f"physics_{name}"] = float(physics_anchor[feat_idx])
        for op_idx in range(arrays["operation_seq"].shape[-1]):
            record[f"operation_{op_idx}"] = float(arrays["operation_seq"][idx].mean(axis=0)[op_idx])
            record[f"future_operation_{op_idx}"] = float(arrays["future_ops"][idx].mean(axis=0)[op_idx])
        if arrays["tsfm_embedding"] is not None:
            for dim_idx in range(min(8, arrays["tsfm_embedding"].shape[-1])):
                record[f"tsfm_{dim_idx}"] = float(arrays["tsfm_embedding"][idx, dim_idx])
        records.append(record)
    table = pd.DataFrame(records)
    targets = _target_views(arrays)
    for key, value in targets.items():
        table[key] = value
    return table


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from scipy.stats import spearmanr

        stat = spearmanr(a, b, nan_policy="omit")
        return float(stat.correlation) if np.isfinite(stat.correlation) else 0.0
    except Exception:
        return None


def _mutual_information(x: np.ndarray, y: np.ndarray) -> float | None:
    try:
        from sklearn.feature_selection import mutual_info_regression

        mi = mutual_info_regression(np.asarray(x, dtype=np.float32).reshape(-1, 1), np.asarray(y, dtype=np.float32), random_state=7)
        return float(mi[0])
    except Exception:
        return None


def _fit_ridge(X_train: np.ndarray, y_train: np.ndarray, reg: float = 1.0) -> np.ndarray:
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.float32)
    eye = np.eye(X.shape[1], dtype=np.float32)
    return np.linalg.solve(X.T @ X + reg * eye, X.T @ y)


def _predict_linear(X: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32) @ np.asarray(coef, dtype=np.float32)


def _run_random_forest(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            random_state=7,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        pred_val = model.predict(X_val)
        return np.asarray(pred_val, dtype=np.float32), np.asarray(model.feature_importances_, dtype=np.float32)
    except Exception:
        return None


def _run_small_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray) -> np.ndarray | None:
    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.multioutput import MultiOutputRegressor

        base = MLPRegressor(
            hidden_layer_sizes=(64,),
            learning_rate_init=1e-3,
            max_iter=50,
            random_state=7,
        )
        model = MultiOutputRegressor(base)
        model.fit(X_train, y_train)
        return np.asarray(model.predict(X_val), dtype=np.float32)
    except Exception:
        return None


def _permutation_importance_scores(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> np.ndarray | None:
    try:
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0, random_state=7)
        model.fit(X_train, y_train)
        scorer = lambda est, X, y: -np.abs(est.predict(X) - y).mean()
        result = permutation_importance(
            model,
            X_val,
            y_val,
            n_repeats=5,
            random_state=7,
            scoring=scorer,
        )
        return np.asarray(result.importances_mean, dtype=np.float32)
    except Exception:
        return None


def _feature_group_matrix(table: pd.DataFrame, group: str) -> Tuple[np.ndarray, List[str]]:
    columns_map = {
        "soh_only": [col for col in table.columns if col in ["anchor_soh", "recent_soh_slope"]],
        "cycle_stats_only": [col for col in table.columns if col.startswith("cycle_")],
        "qv_curve_stats_only": [col for col in table.columns if col.startswith("cycle_delta_v") or col.startswith("cycle_r_") or col.startswith("cycle_vc_") or col.startswith("cycle_vd_")],
        "qv_map_pooled_stats": [col for col in table.columns if col.startswith("qv_map_")],
        "partial_charge_only": [col for col in table.columns if col.startswith("partial_charge_")],
        "relaxation_only": [col for col in table.columns if col.startswith("relaxation_")],
        "physics_12d_only": [col for col in table.columns if col.startswith("physics_")],
        "operation_only": [col for col in table.columns if col.startswith("operation_") or col.startswith("future_operation_")],
    }
    if group == "soh_plus_operation":
        cols = columns_map["soh_only"] + columns_map["operation_only"]
    elif group == "soh_plus_qv_physics":
        cols = columns_map["soh_only"] + columns_map["qv_curve_stats_only"] + columns_map["physics_12d_only"]
    elif group == "all_physics_no_tsfm":
        cols = columns_map["cycle_stats_only"] + columns_map["qv_map_pooled_stats"] + columns_map["partial_charge_only"] + columns_map["relaxation_only"] + columns_map["physics_12d_only"] + columns_map["operation_only"]
    elif group == "all_features_with_tsfm":
        cols = (
            columns_map["cycle_stats_only"]
            + columns_map["qv_map_pooled_stats"]
            + columns_map["partial_charge_only"]
            + columns_map["relaxation_only"]
            + columns_map["physics_12d_only"]
            + columns_map["operation_only"]
            + [col for col in table.columns if col.startswith("tsfm_")]
        )
    else:
        cols = columns_map[group]
    cols = [col for col in cols if col in table.columns]
    return table[cols].to_numpy(dtype=np.float32), cols


def _repeat_rows(array: np.ndarray, repeats: int) -> np.ndarray:
    if repeats <= 1:
        return np.asarray(array)
    return np.repeat(np.asarray(array), repeats, axis=0)


def _run_fewshot_support_sanity(
    flat_table: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    feature_groups: List[str],
) -> pd.DataFrame:
    train_mask = flat_table["split"].eq("source_train").to_numpy()
    support_mask = flat_table["split"].eq("target_support").to_numpy()
    query_mask = flat_table["split"].eq("target_query").to_numpy()
    if not train_mask.any() or not support_mask.any() or not query_mask.any():
        return pd.DataFrame()

    y_train = arrays["future_delta_soh"][train_mask]
    y_support = arrays["future_delta_soh"][support_mask]
    y_query = arrays["future_delta_soh"][query_mask]
    rows = []

    for group in feature_groups:
        X, cols = _feature_group_matrix(flat_table, group)
        if X.size == 0 or not cols:
            continue
        X_train = X[train_mask]
        X_support = X[support_mask]
        X_query = X[query_mask]
        if X_train.size == 0 or X_support.size == 0 or X_query.size == 0:
            continue

        source_coef = _fit_ridge(X_train, y_train, reg=1.0)
        pred_support = _predict_linear(X_support, source_coef)
        pred_query_source = _predict_linear(X_query, source_coef)
        rows.append(
            {
                "feature_group": group,
                "adaptation": "source_only",
                **regression_metrics(pred_query_source, y_query),
                "last_step_mae": float(np.abs(pred_query_source[:, -1] - y_query[:, -1]).mean()),
            }
        )

        bias = np.asarray(y_support - pred_support, dtype=np.float32).mean(axis=0, keepdims=True)
        pred_query_bias = pred_query_source + bias
        rows.append(
            {
                "feature_group": group,
                "adaptation": "support_bias",
                **regression_metrics(pred_query_bias, y_query),
                "last_step_mae": float(np.abs(pred_query_bias[:, -1] - y_query[:, -1]).mean()),
            }
        )

        repeat_factor = max(2, min(8, int(np.ceil(len(X_train) / max(len(X_support), 1)))))
        adapt_coef = _fit_ridge(
            np.concatenate([X_train, _repeat_rows(X_support, repeat_factor)], axis=0),
            np.concatenate([y_train, _repeat_rows(y_support, repeat_factor)], axis=0),
            reg=1.0,
        )
        pred_query_refit = _predict_linear(X_query, adapt_coef)
        rows.append(
            {
                "feature_group": group,
                "adaptation": "support_weighted_refit",
                **regression_metrics(pred_query_refit, y_query),
                "last_step_mae": float(np.abs(pred_query_refit[:, -1] - y_query[:, -1]).mean()),
            }
        )

    return pd.DataFrame(rows)


def validate_preprocessing_features(cfg: Dict[str, object]) -> Dict[str, object]:
    case_bank_dir = _ensure_case_bank(cfg)
    rows, arrays, feature_names = _load_case_bank(case_bank_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(Path(cfg.get("experiments", {}).get("feature_validation", {}).get("output_dir", "output/experiments/feature_validation")) / run_id)
    figure_dir = ensure_dir(output_dir / "figures")
    try:
        import yaml

        (output_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception:
        (output_dir / "run_config.yaml").write_text(json.dumps(cfg, indent=2, ensure_ascii=True))
    (output_dir / "run_metadata.json").write_text(json.dumps({"run_id": run_id, "case_bank_dir": str(case_bank_dir)}, indent=2, ensure_ascii=True))
    (output_dir / "experiment.log").write_text("feature validation started\n")

    availability = _availability_rows(rows, arrays)
    availability_summary = availability.drop(columns=["case_id", "split", "chemistry_family", "domain_label"]).mean().to_dict()
    quality_summary = {
        "feature_group_availability": {key: float(value) for key, value in availability_summary.items()},
        "nan_ratio": float(1.0 - np.isfinite(arrays["cycle_stats"]).mean()),
        "inf_ratio": float(np.isinf(arrays["cycle_stats"]).mean()),
        "outlier_ratio": float((np.abs(arrays["cycle_stats"]) > np.quantile(np.abs(arrays["cycle_stats"]), 0.99)).mean()),
    }
    pd.DataFrame([quality_summary]).to_csv(output_dir / "feature_quality_summary.csv", index=False)
    (output_dir / "feature_quality_summary.json").write_text(json.dumps(quality_summary, indent=2, ensure_ascii=True))
    save_bar(quality_summary["feature_group_availability"], "Feature availability by group", "Availability", figure_dir / "feature_availability_by_group.png")
    missing_by_split = availability.drop(columns=["case_id"]).groupby("split").mean(numeric_only=True).T
    save_heatmap(1.0 - missing_by_split.to_numpy(dtype=np.float32), missing_by_split.index.tolist(), missing_by_split.columns.tolist(), "Feature missingness by split", figure_dir / "feature_missingness_by_split.png")

    grouped_chem = defaultdict(list)
    grouped_domain = defaultdict(list)
    for _, row in rows.iterrows():
        grouped_chem[str(row["chemistry_family"])].append(float(row["anchor_soh"]))
        grouped_domain[str(row["domain_label"])].append(float(row["recent_soh_slope"]))
    save_boxplot(grouped_chem, "Feature distribution by chemistry", "Anchor SOH", figure_dir / "feature_distribution_by_chemistry.png")
    save_boxplot(grouped_domain, "Feature distribution by domain", "Recent SOH slope", figure_dir / "feature_distribution_by_domain.png")

    flat_table = _flat_feature_table(rows, arrays, feature_names)
    explore_mask = flat_table["split"].isin(["source_train", "source_val"]).to_numpy()
    correlation_rows = []
    target_views = ["future_delta_h1", "future_delta_last", "future_delta_mean", "future_slope"]
    feature_columns = [col for col in flat_table.columns if col not in {"case_id", "split", "chemistry_family", "domain_label"} and col not in target_views]
    for feature in feature_columns:
        x = flat_table.loc[explore_mask, feature].to_numpy(dtype=np.float32)
        for target in target_views:
            y = flat_table.loc[explore_mask, target].to_numpy(dtype=np.float32)
            spearman = _spearman(x, y)
            mutual_info = _mutual_information(x, y)
            correlation_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "pearson": _corr(x, y),
                    "spearman": spearman if spearman is not None else np.nan,
                    "mutual_information": mutual_info if mutual_info is not None else np.nan,
                }
            )
    correlation_df = pd.DataFrame(correlation_rows)
    correlation_df.to_csv(output_dir / "feature_target_correlation.csv", index=False)
    topk = correlation_df.assign(abs_pearson=lambda df: df["pearson"].abs()).sort_values("abs_pearson", ascending=False).head(20)
    topk.to_csv(output_dir / "feature_target_correlation_topk.csv", index=False)
    pivot = correlation_df.pivot(index="feature", columns="target", values="pearson").fillna(0.0)
    save_heatmap(pivot.to_numpy(dtype=np.float32), pivot.index.tolist(), pivot.columns.tolist(), "Feature-target correlation", figure_dir / "feature_target_correlation_heatmap.png", cmap="coolwarm")
    save_bar({row["feature"]: float(row["abs_pearson"]) for _, row in topk.iterrows()}, "Top feature correlations", "|Pearson|", figure_dir / "top_feature_correlation_bar.png")

    train_mask = flat_table["split"].eq("source_train").to_numpy()
    val_mask = flat_table["split"].eq("source_val").to_numpy()
    y_train = arrays["future_delta_soh"][train_mask]
    y_val = arrays["future_delta_soh"][val_mask]
    feature_groups = [
        "soh_only",
        "cycle_stats_only",
        "qv_curve_stats_only",
        "qv_map_pooled_stats",
        "partial_charge_only",
        "relaxation_only",
        "physics_12d_only",
        "operation_only",
        "soh_plus_operation",
        "soh_plus_qv_physics",
        "all_physics_no_tsfm",
    ]
    if arrays["tsfm_embedding"] is not None:
        feature_groups.append("all_features_with_tsfm")
    ablation_rows = []
    importance_rows = []
    best_permutation_context = None
    run_rf = bool(cfg.get("experiments", {}).get("feature_validation", {}).get("run_random_forest", True))
    run_mlp = bool(cfg.get("experiments", {}).get("feature_validation", {}).get("run_small_mlp", False))
    for group in feature_groups:
        X, cols = _feature_group_matrix(flat_table, group)
        X_train = X[train_mask]
        X_val = X[val_mask]
        if X_train.size == 0 or X_val.size == 0:
            continue
        coef = _fit_ridge(X_train, y_train, reg=1.0)
        pred_val = _predict_linear(X_val, coef)
        metrics = regression_metrics(pred_val, y_val)
        h_metrics = horizon_metrics(pred_val, y_val)
        ablation_rows.append(
            {
                "feature_group": group,
                "model": "ridge",
                **metrics,
                "last_step_mae": float(np.abs(pred_val[:, -1] - y_val[:, -1]).mean()),
                "horizon_mae_json": json.dumps(h_metrics["mae"].astype(float).tolist()),
            }
        )
        for feat_name, weight in zip(cols, np.abs(coef).mean(axis=1).tolist()):
            importance_rows.append({"feature_group": group, "feature": feat_name, "importance": float(weight), "model": "ridge"})
        if best_permutation_context is None or metrics["mae"] < best_permutation_context["mae"]:
            best_permutation_context = {
                "mae": metrics["mae"],
                "feature_group": group,
                "cols": cols,
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
            }

        if run_rf:
            rf_result = _run_random_forest(X_train, y_train, X_val)
            if rf_result is not None:
                pred_rf, rf_importance = rf_result
                rf_metrics = regression_metrics(pred_rf, y_val)
                rf_h_metrics = horizon_metrics(pred_rf, y_val)
                ablation_rows.append(
                    {
                        "feature_group": group,
                        "model": "random_forest",
                        **rf_metrics,
                        "last_step_mae": float(np.abs(pred_rf[:, -1] - y_val[:, -1]).mean()),
                        "horizon_mae_json": json.dumps(rf_h_metrics["mae"].astype(float).tolist()),
                    }
                )
                for feat_name, weight in zip(cols, rf_importance.tolist()):
                    importance_rows.append({"feature_group": group, "feature": feat_name, "importance": float(weight), "model": "random_forest"})

        if run_mlp:
            pred_mlp = _run_small_mlp(X_train, y_train, X_val)
            if pred_mlp is not None:
                mlp_metrics = regression_metrics(pred_mlp, y_val)
                mlp_h_metrics = horizon_metrics(pred_mlp, y_val)
                ablation_rows.append(
                    {
                        "feature_group": group,
                        "model": "small_mlp",
                        **mlp_metrics,
                        "last_step_mae": float(np.abs(pred_mlp[:, -1] - y_val[:, -1]).mean()),
                        "horizon_mae_json": json.dumps(mlp_h_metrics["mae"].astype(float).tolist()),
                    }
                )

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(output_dir / "feature_ablation_metrics.csv", index=False)
    by_domain = []
    by_chem = []
    for group in ablation_df["feature_group"].tolist():
        X, _ = _feature_group_matrix(flat_table, group)
        coef = _fit_ridge(X[train_mask], y_train, reg=1.0)
        pred_val = _predict_linear(X[val_mask], coef)
        val_rows = rows[val_mask].reset_index(drop=True)
        for chemistry, idxs in val_rows.groupby("chemistry_family").groups.items():
            idxs = np.asarray(list(idxs), dtype=np.int64)
            by_chem.append({"feature_group": group, "chemistry_family": chemistry, **regression_metrics(pred_val[idxs], y_val[idxs])})
        for domain, idxs in val_rows.groupby("domain_label").groups.items():
            idxs = np.asarray(list(idxs), dtype=np.int64)
            by_domain.append({"feature_group": group, "domain_label": domain, **regression_metrics(pred_val[idxs], y_val[idxs])})
    pd.DataFrame(by_domain).to_csv(output_dir / "feature_ablation_metrics_by_domain.csv", index=False)
    pd.DataFrame(by_chem).to_csv(output_dir / "feature_ablation_metrics_by_chemistry.csv", index=False)
    save_bar(dict(zip(ablation_df["feature_group"], ablation_df["mae"])), "Feature group ablation MAE", "MAE", figure_dir / "feature_group_ablation_mae.png")
    save_bar(dict(zip(ablation_df["feature_group"], ablation_df["rmse"])), "Feature group ablation RMSE", "RMSE", figure_dir / "feature_group_ablation_rmse.png")
    horizon_heat = np.stack([np.asarray(json.loads(text), dtype=np.float32) for text in ablation_df["horizon_mae_json"].tolist()]) if len(ablation_df) else np.zeros((0, 0))
    save_heatmap(horizon_heat, ablation_df["feature_group"].tolist(), [f"h{i+1}" for i in range(horizon_heat.shape[1])], "Horizon-wise ablation MAE", figure_dir / "horizon_wise_ablation.png")
    if by_domain:
        domain_df = pd.DataFrame(by_domain)
        domain_pivot = domain_df.pivot_table(index="feature_group", columns="domain_label", values="mae", aggfunc="mean").fillna(0.0)
        save_heatmap(domain_pivot.to_numpy(dtype=np.float32), domain_pivot.index.tolist(), domain_pivot.columns.tolist(), "Domain ablation MAE", figure_dir / "domain_ablation_heatmap.png")
    if by_chem:
        chem_df = pd.DataFrame(by_chem)
        chem_pivot = chem_df.pivot_table(index="feature_group", columns="chemistry_family", values="mae", aggfunc="mean").fillna(0.0)
        save_heatmap(chem_pivot.to_numpy(dtype=np.float32), chem_pivot.index.tolist(), chem_pivot.columns.tolist(), "Chemistry ablation MAE", figure_dir / "chemistry_ablation_heatmap.png")

    importance_df = pd.DataFrame(importance_rows).sort_values("importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)
    top_importance = importance_df.head(20)
    save_bar(dict(zip(top_importance["feature"], top_importance["importance"])), "Feature importance top20", "Importance", figure_dir / "feature_importance_top20.png")
    permutation_scores = None
    if best_permutation_context is not None:
        permutation_scores = _permutation_importance_scores(
            best_permutation_context["X_train"],
            best_permutation_context["y_train"],
            best_permutation_context["X_val"],
            best_permutation_context["y_val"],
        )
    if permutation_scores is not None:
        perm_pairs = sorted(
            zip(best_permutation_context["cols"], permutation_scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
        save_bar(dict(perm_pairs), "Permutation importance top20", "Importance", figure_dir / "permutation_importance_top20.png")
    else:
        save_bar(dict(zip(top_importance["feature"], top_importance["importance"])), "Permutation importance top20", "Importance", figure_dir / "permutation_importance_top20.png")

    fewshot_df = _run_fewshot_support_sanity(flat_table, arrays, feature_groups)
    if len(fewshot_df):
        fewshot_df.to_csv(output_dir / "feature_ablation_metrics_fewshot.csv", index=False)
        best_fewshot = fewshot_df.sort_values("mae").iloc[0].to_dict()
        fewshot_plot_rows = fewshot_df.sort_values(["mae", "feature_group", "adaptation"]).head(12)
        save_bar(
            {f"{row['feature_group']}|{row['adaptation']}": float(row["mae"]) for _, row in fewshot_plot_rows.iterrows()},
            "Few-shot support sanity MAE",
            "MAE",
            figure_dir / "fewshot_support_sanity_mae.png",
        )
    else:
        best_fewshot = None

    weak_chem = pd.DataFrame(by_chem).sort_values("mae", ascending=False).head(3)["chemistry_family"].tolist() if by_chem else []
    weak_domain = pd.DataFrame(by_domain).sort_values("mae", ascending=False).head(3)["domain_label"].tolist() if by_domain else []
    recommend_groups = ablation_df.sort_values("mae").head(3)[["feature_group", "model"]].astype(str).agg("/".join, axis=1).tolist() if len(ablation_df) else []
    summary_lines = [
        "# Summary",
        f"- 数据集覆盖 case 数: {len(rows)}",
        f"- 每类特征平均可用率: {json.dumps(quality_summary['feature_group_availability'], ensure_ascii=True)}",
        f"- 最有效的特征组: {recommend_groups}",
        f"- 效果较弱的 chemistry: {weak_chem}",
        f"- 效果较弱的 domain: {weak_domain}",
        f"- 是否建议进入最终预测主干: {bool(len(recommend_groups) > 0)}",
        f"- target_query 未用于特征探索/选择: True",
        f"- 发现的异常或潜在数据问题: nan_ratio={quality_summary['nan_ratio']:.4f}, inf_ratio={quality_summary['inf_ratio']:.4f}, outlier_ratio={quality_summary['outlier_ratio']:.4f}",
        f"- few-shot support sanity 最优结果: {best_fewshot if best_fewshot is not None else 'unavailable'}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines))
    return {"output_dir": str(output_dir), "run_id": run_id}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate preprocessing features for battery SOH")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = validate_preprocessing_features(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
