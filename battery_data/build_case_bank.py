from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_data.canonicalize import assign_cell_uids, combine_canonical_cycles, load_enabled_cells
from battery_data.case_schema import CaseSample
from battery_data.curve_features import QV_CHANNEL_NAMES, extract_q_indexed_feature_map, plot_qv_feature_map
from battery_data.domain_labeling import build_domain_label
from battery_data.features import augment_cycle_feature_frame, recent_delta_mean
from battery_data.physical_features import (
    aggregate_window_physics_features,
    compute_physics_features,
    extract_partial_charge_curve,
    extract_relaxation_curve,
    plot_partial_charge_and_relaxation,
)
from battery_data.schema import DEFAULT_TOKEN_FEATURES
from battery_data.splits import assert_no_split_leakage, build_split_manifest


DEFAULT_QV_CURVE_STATS = [
    "delta_v_mean",
    "delta_v_std",
    "delta_v_max",
    "r_mean",
    "r_std",
    "r_q95",
    "vc_slope_mean",
    "vd_slope_mean",
    "ic_mean",
    "id_mean",
    "v_charge_mean",
    "v_discharge_mean",
]

DEFAULT_OPERATION_FEATURES = [
    "current_abs_mean",
    "temp_mean",
    "cc_time",
    "cv_time",
    "charge_throughput_delta_1",
    "discharge_throughput_delta_1",
    "energy_charge_delta_1",
    "energy_discharge_delta_1",
]


def load_config(path: str) -> dict:
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def _save_case_rows(rows_df: pd.DataFrame, output_dir: Path) -> None:
    parquet_path = output_dir / "case_rows.parquet"
    csv_path = output_dir / "case_rows.csv"
    try:
        rows_df.to_parquet(parquet_path, index=False)
    except Exception:
        rows_df.to_csv(csv_path, index=False)


def _find_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lowered = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return str(lowered[candidate.lower()])
    return None


def _safe_numeric(df: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None or column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").astype("float64")


def _load_raw_cycle_tables(source_dataset: str, file_path: str) -> Dict[int, pd.DataFrame]:
    path = Path(file_path)
    if not path.exists():
        return {}

    if source_dataset == "hust":
        raw_df = pd.read_csv(path)
        grouped = raw_df.groupby("cycle_index", sort=True)
        result = {}
        for cycle_idx, grp in grouped:
            result[int(cycle_idx)] = pd.DataFrame(
                {
                    "time": pd.to_numeric(grp.get("Time (s)"), errors="coerce"),
                    "voltage": pd.to_numeric(grp.get("Voltage (V)"), errors="coerce"),
                    "current": pd.to_numeric(grp.get("Current (mA)"), errors="coerce") / 1000.0,
                    "capacity": pd.to_numeric(grp.get("Capacity (mAh)"), errors="coerce") / 1000.0,
                    "temperature": np.nan,
                    "step": grp.get("Status"),
                }
            )
        return result

    if source_dataset == "tju":
        raw_df = pd.read_csv(path)
        grouped = raw_df.groupby("cycle number", sort=True)
        result = {}
        for cycle_idx, grp in grouped:
            current = pd.to_numeric(grp.get("<I>/mA"), errors="coerce") / 1000.0
            charge_capacity = pd.to_numeric(grp.get("Q charge/mA.h"), errors="coerce") / 1000.0
            discharge_capacity = pd.to_numeric(grp.get("Q discharge/mA.h"), errors="coerce") / 1000.0
            capacity = np.where(current.to_numpy(dtype=np.float64) >= 0, charge_capacity, discharge_capacity)
            result[int(float(cycle_idx))] = pd.DataFrame(
                {
                    "time": pd.to_numeric(grp.get("time/s"), errors="coerce"),
                    "voltage": pd.to_numeric(grp.get("Ecell/V"), errors="coerce"),
                    "current": current,
                    "capacity": capacity,
                    "temperature": np.nan,
                    "step": np.where(current > 0, "charge", "discharge"),
                }
            )
        return result

    if source_dataset == "xjtu":
        if path.name.endswith("_summary.csv"):
            path = path.with_name(path.name.replace("_summary.csv", "_data.csv"))
        if not path.exists():
            return {}
        raw_df = pd.read_csv(path)
        grouped = raw_df.groupby("cycle_index", sort=True)
        result = {}
        for cycle_idx, grp in grouped:
            result[int(cycle_idx)] = pd.DataFrame(
                {
                    "time": pd.to_numeric(grp.get("relative_time_min"), errors="coerce"),
                    "voltage": pd.to_numeric(grp.get("voltage_V"), errors="coerce"),
                    "current": pd.to_numeric(grp.get("current_A"), errors="coerce"),
                    "capacity": pd.to_numeric(grp.get("capacity_Ah"), errors="coerce"),
                    "temperature": pd.to_numeric(grp.get("temperature_C"), errors="coerce"),
                    "step": grp.get("description"),
                }
            )
        return result

    if source_dataset == "mit":
        if path.name.endswith("_structure_summary.csv"):
            path = path.with_name(path.name.replace("_structure_summary.csv", "_structure_cycles_interpolated.csv"))
        if not path.exists():
            return {}
        raw_df = pd.read_csv(path)
        grouped = raw_df.groupby("cycle_index", sort=True)
        result = {}
        for cycle_idx, grp in grouped:
            current = pd.to_numeric(grp.get("current"), errors="coerce")
            charge_capacity = pd.to_numeric(grp.get("charge_capacity"), errors="coerce")
            discharge_capacity = pd.to_numeric(grp.get("discharge_capacity"), errors="coerce")
            capacity = np.where(current.to_numpy(dtype=np.float64) >= 0, charge_capacity, discharge_capacity)
            result[int(cycle_idx)] = pd.DataFrame(
                {
                    "time": np.arange(len(grp), dtype=np.float64),
                    "voltage": pd.to_numeric(grp.get("voltage"), errors="coerce"),
                    "current": current,
                    "capacity": capacity,
                    "temperature": pd.to_numeric(grp.get("temperature"), errors="coerce"),
                    "step": grp.get("step_type"),
                }
            )
        return result

    return {}


def _degradation_stage(anchor_soh: float) -> str:
    if anchor_soh >= 0.95:
        return "fresh"
    if anchor_soh >= 0.9:
        return "early"
    if anchor_soh >= 0.8:
        return "mid"
    if anchor_soh >= 0.7:
        return "late"
    return "end"


def _recent_curvature(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 3:
        return 0.0
    return float(np.diff(arr, n=2).mean())


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return float(value)


def _collect_cycle_feature_names(cfg: Dict[str, object]) -> List[str]:
    token_features = list(cfg.get("memory", {}).get("token_features", DEFAULT_TOKEN_FEATURES))
    cycle_features = [name for name in token_features if name != "soh"]
    for name in DEFAULT_QV_CURVE_STATS:
        if name not in cycle_features:
            cycle_features.append(name)
    return cycle_features


def _select_operation_features(frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    columns = [name for name in DEFAULT_OPERATION_FEATURES if name in frame.columns]
    values = []
    for name in DEFAULT_OPERATION_FEATURES:
        if name in frame.columns:
            series = pd.to_numeric(frame[name], errors="coerce").fillna(0.0).astype("float32")
        else:
            series = pd.Series([0.0] * len(frame), dtype="float32")
        values.append(series.to_numpy(dtype=np.float32))
    return np.stack(values, axis=-1), list(DEFAULT_OPERATION_FEATURES)


def _build_cycle_feature_table(
    cell_cycles: pd.DataFrame,
    cycle_features: List[str],
    rolling_window: int,
    spectral_window: int,
    spectral_columns: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    frame = augment_cycle_feature_frame(
        cell_cycles,
        rolling_window=rolling_window,
        spectral_window=spectral_window,
        spectral_columns=spectral_columns,
    )
    operation_seq, operation_names = _select_operation_features(frame)
    for name in cycle_features:
        if name not in frame.columns:
            frame[name] = 0.0
    return frame, operation_seq, operation_names


def _window_feature_summary(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return np.concatenate(
        [
            np.nan_to_num(arr.mean(axis=0), nan=0.0),
            np.nan_to_num(arr.std(axis=0), nan=0.0),
            np.nan_to_num(arr[-1], nan=0.0),
        ],
        axis=0,
    ).astype(np.float32)


def _fit_normalization_stats(rows: pd.DataFrame, arrays: Dict[str, np.ndarray]) -> Dict[str, object]:
    source_idx = rows.index[rows["split"] == "source_train"].to_numpy(dtype=np.int64)
    stats: Dict[str, object] = {"fit_split": "source_train", "num_fit_cases": int(len(source_idx))}
    for name, values in arrays.items():
        if len(source_idx) == 0:
            arr = np.asarray(values, dtype=np.float32).reshape(-1)
        else:
            arr = np.asarray(values[source_idx], dtype=np.float32).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            stats[name] = {"mean": 0.0, "std": 1.0, "q05": 0.0, "q50": 0.0, "q95": 0.0}
        else:
            stats[name] = {
                "mean": float(finite.mean()),
                "std": float(finite.std() if finite.std() > 1e-8 else 1.0),
                "q05": float(np.quantile(finite, 0.05)),
                "q50": float(np.quantile(finite, 0.50)),
                "q95": float(np.quantile(finite, 0.95)),
            }
    return stats


def _hashable_missing_summary(case: CaseSample) -> Dict[str, float]:
    return {
        "qv_missing_ratio": float(1.0 - np.asarray(case.qv_masks, dtype=np.float32).mean()),
        "partial_charge_missing_ratio": float(1.0 - np.asarray(case.partial_charge_masks, dtype=np.float32).mean()),
        "relaxation_missing_ratio": float(1.0 - np.asarray(case.relaxation_masks, dtype=np.float32).mean()),
        "physics_feature_missing_ratio": float(1.0 - np.asarray(case.physics_feature_masks, dtype=np.float32).mean()),
        "future_ops_missing_ratio": float(1.0 - np.asarray(case.future_operation_mask, dtype=np.float32).mean()) if np.asarray(case.future_operation_mask).size else 1.0,
    }


def _bar_plot(data: Dict[str, float], title: str, xlabel: str, ylabel: str, save_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    labels = list(data.keys())
    values = [float(data[label]) for label in labels]
    axis.bar(np.arange(len(labels)), values, color="#3b82f6")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def _distribution_plot(
    grouped_values: Dict[str, List[float]],
    title: str,
    ylabel: str,
    save_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    labels = list(grouped_values.keys())
    values = [grouped_values[label] if grouped_values[label] else [0.0] for label in labels]
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def _heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, save_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(max(6, 0.35 * len(col_labels)), max(3, 0.45 * len(row_labels))), dpi=180)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Missing ratio")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def _save_case_bank_figures(case_rows: pd.DataFrame, arrays: Dict[str, np.ndarray], feature_names: Dict[str, object], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    availability_by_dataset = {}
    availability_by_chem = {}
    availability_by_domain = {}

    for group_col, output in [
        ("source_dataset", availability_by_dataset),
        ("chemistry_family", availability_by_chem),
        ("domain_label", availability_by_domain),
    ]:
        grouped = case_rows.groupby(group_col)
        for key, group in grouped:
            idx = group.index.to_numpy(dtype=np.int64)
            output[str(key)] = float(arrays["case_physics_feature_masks.npy"][idx].mean()) if len(idx) else 0.0

    _bar_plot(availability_by_dataset, "Feature availability by dataset", "Dataset", "Availability", figure_dir / "feature_availability_by_dataset.png")
    _bar_plot(availability_by_chem, "Feature availability by chemistry", "Chemistry", "Availability", figure_dir / "feature_availability_by_chemistry.png")
    _bar_plot(availability_by_domain, "Feature availability by domain", "Domain", "Availability", figure_dir / "feature_availability_by_domain.png")

    soh_grouped = defaultdict(list)
    slope_grouped = defaultdict(list)
    for _, row in case_rows.iterrows():
        soh_grouped[str(row["source_dataset"])].append(float(row["anchor_soh"]))
        slope_grouped[str(row["domain_label"])].append(float(row["recent_soh_slope"]))
    _distribution_plot(soh_grouped, "Anchor SOH distribution by dataset", "Anchor SOH", figure_dir / "soh_distribution_by_dataset.png")
    _distribution_plot(slope_grouped, "Recent SOH slope distribution by domain", "Recent SOH slope", figure_dir / "soh_slope_distribution_by_domain.png")

    physics_mask = arrays["case_physics_feature_masks.npy"]
    chemistry_values = case_rows["chemistry_family"].astype(str).tolist()
    chemistries = list(dict.fromkeys(chemistry_values))
    matrix = np.zeros((len(chemistries), physics_mask.shape[-1]), dtype=np.float32)
    for row_idx, chemistry in enumerate(chemistries):
        idx = case_rows.index[case_rows["chemistry_family"].astype(str) == chemistry].to_numpy(dtype=np.int64)
        if len(idx):
            matrix[row_idx] = 1.0 - physics_mask[idx].mean(axis=(0, 1))
    _heatmap(matrix, chemistries, feature_names["physics_feature_names"], "Physics feature missingness by chemistry", figure_dir / "physics_feature_missingness_heatmap.png")

    plotted = False
    for chemistry in chemistries:
        idx = case_rows.index[case_rows["chemistry_family"].astype(str) == chemistry].to_numpy(dtype=np.int64)
        if len(idx) == 0:
            continue
        case_idx = int(idx[0])
        q_grid = np.linspace(0.0, 1.0, arrays["case_qv_maps.npy"].shape[-1], dtype=np.float32)
        plot_qv_feature_map(
            q_grid=q_grid,
            qv_map=arrays["case_qv_maps.npy"][case_idx, -1],
            qv_mask=arrays["case_qv_masks.npy"][case_idx, -1],
            save_path=figure_dir / "example_qv_maps_by_chemistry.png",
            title=f"Example anchor Q-indexed map | chemistry={chemistry}",
        )
        plotted = True
        break
    if not plotted:
        plot_qv_feature_map(
            q_grid=np.linspace(0.0, 1.0, arrays["case_qv_maps.npy"].shape[-1], dtype=np.float32),
            qv_map=np.zeros_like(arrays["case_qv_maps.npy"][0, -1]),
            qv_mask=np.zeros_like(arrays["case_qv_masks.npy"][0, -1]),
            save_path=figure_dir / "example_qv_maps_by_chemistry.png",
            title="Example anchor Q-indexed map",
        )

    example_idx = 0
    plot_partial_charge_and_relaxation(
        partial_charge_curve=arrays["case_partial_charge.npy"][example_idx, -1],
        partial_charge_mask=bool(arrays["case_partial_charge_mask.npy"][example_idx, -1]),
        relaxation_curve=arrays["case_relaxation.npy"][example_idx, -1],
        relaxation_mask=bool(arrays["case_relaxation_mask.npy"][example_idx, -1]),
        save_path=figure_dir / "example_partial_charge_relaxation.png",
        title="Example anchor partial-charge / relaxation features",
    )


def _optional_window_encoder(cfg: Dict[str, object]):
    model_cfg = cfg.get("model", {})
    if not bool(model_cfg.get("use_tsfm_embedding", True)):
        return None
    if "encoder" not in cfg:
        return None
    try:
        from battery_data.cli_build_memory_bank import build_encoder_from_config

        return build_encoder_from_config(cfg)
    except Exception as exc:
        print(f"[CaseBank] Skip tsfm embedding build: {exc}", flush=True)
        return None


def build_case_bank(cfg: Dict[str, object]) -> Dict[str, object]:
    output_dir = Path(cfg.get("output_dir", "output/case_bank"))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures" / "preprocessing_overview"
    figure_dir.mkdir(parents=True, exist_ok=True)

    cells = assign_cell_uids(load_enabled_cells(cfg), prefix=str(cfg.get("cell_uid_prefix", "cell")))
    if not cells:
        raise ValueError("No cells were loaded; cannot build case bank.")

    split_manifest = build_split_manifest(cells, cfg.get("split", {}), cfg.get("domain_labeling"))
    assert_no_split_leakage(split_manifest)
    manifest_map = split_manifest.set_index("cell_uid").to_dict(orient="index")
    canonical_cycles = combine_canonical_cycles(cells)

    feature_cfg = cfg.get("features", {})
    memory_cfg = cfg.get("memory", {})
    lookback = int(memory_cfg.get("lookback_length", 32))
    horizon = int(memory_cfg.get("prediction_length", 8))
    stride = int(memory_cfg.get("stride", 1))
    q_grid_size = int(feature_cfg.get("q_grid_size", 100))
    partial_charge_segments = int(feature_cfg.get("partial_charge_segments", 50))
    relaxation_points = int(feature_cfg.get("relaxation_points", 30))
    rolling_window = int(feature_cfg.get("rolling_median_window", 5))
    relax_time_max = float(feature_cfg.get("relaxation_time_max_minutes", 30))
    current_rest_threshold = float(feature_cfg.get("current_rest_threshold", 0.02))
    future_ops_mode = str(feature_cfg.get("future_ops_mode", "known"))

    derived_cfg = memory_cfg.get("derived_features", {})
    spectral_window = int(derived_cfg.get("spectral_window", 16))
    spectral_columns = list(derived_cfg.get("spectral_columns", ["voltage_mean", "temp_mean", "current_mean"]))
    cycle_feature_names = _collect_cycle_feature_names(cfg)

    cases: List[CaseSample] = []
    case_rows: List[Dict[str, object]] = []
    q_grid = np.linspace(0.0, 1.0, q_grid_size, dtype=np.float32)
    raw_tables = {cell.source_info.get("cell_uid", cell.cycles["cell_uid"].iloc[0]): _load_raw_cycle_tables(cell.source_dataset, cell.file_path) for cell in cells}

    for cell in cells:
        cell_uid = str(cell.cycles["cell_uid"].iloc[0])
        cell_cycles = cell.cycles.sort_values("cycle_idx").reset_index(drop=True)
        split_row = manifest_map[cell_uid]
        split = str(split_row["split"])
        raw_cycle_map = raw_tables.get(cell_uid, {})
        feature_frame, operation_seq_full, operation_names = _build_cycle_feature_table(
            cell_cycles,
            cycle_features=cycle_feature_names,
            rolling_window=int(derived_cfg.get("rolling_window", 5)),
            spectral_window=spectral_window,
            spectral_columns=spectral_columns,
        )
        soh_values = pd.to_numeric(feature_frame["soh"], errors="coerce").to_numpy(dtype=np.float32)
        capacity_values = pd.to_numeric(feature_frame["capacity"], errors="coerce").to_numpy(dtype=np.float32)
        throughput_values = pd.to_numeric(feature_frame.get("charge_throughput"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if not np.isfinite(throughput_values).any():
            throughput_values = pd.to_numeric(feature_frame.get("discharge_throughput"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

        qv_maps_all = []
        qv_masks_all = []
        partial_curves_all = []
        partial_masks_all = []
        relaxation_curves_all = []
        relaxation_masks_all = []
        physics_features_all = []
        physics_masks_all = []
        curve_stats_rows = []
        availability_rows = []

        for _, row in feature_frame.iterrows():
            cycle_idx = int(row["cycle_idx"])
            raw_cycle_df = raw_cycle_map.get(cycle_idx)
            if raw_cycle_df is None or raw_cycle_df.empty:
                qv_result = {
                    "qv_map": np.zeros((6, q_grid_size), dtype=np.float32),
                    "qv_mask": np.zeros(6, dtype=np.float32),
                    "q_grid": q_grid,
                    "curve_stats": {name: 0.0 for name in DEFAULT_QV_CURVE_STATS},
                }
                partial_result = {
                    "partial_charge_curve": np.zeros(partial_charge_segments, dtype=np.float32),
                    "partial_charge_mask": False,
                    "partial_charge_stats": {},
                }
                relax_result = {
                    "relaxation_curve": np.zeros(relaxation_points, dtype=np.float32),
                    "relaxation_mask": False,
                    "relaxation_stats": {},
                }
            else:
                qv_result = extract_q_indexed_feature_map(
                    raw_cycle_df,
                    q_grid_size=q_grid_size,
                    rolling_median_window=rolling_window,
                )
                partial_result = extract_partial_charge_curve(
                    raw_cycle_df,
                    voltage_grid_size=partial_charge_segments,
                )
                relax_result = extract_relaxation_curve(
                    raw_cycle_df,
                    relax_points=relaxation_points,
                    relax_time_max_minutes=relax_time_max,
                    current_rest_threshold=current_rest_threshold,
                )
            physics_result = compute_physics_features(
                partial_charge_curve=partial_result["partial_charge_curve"],
                partial_charge_mask=bool(partial_result["partial_charge_mask"]),
                relaxation_curve=relax_result["relaxation_curve"],
                relaxation_mask=bool(relax_result["relaxation_mask"]),
                qv_curve_stats=qv_result["curve_stats"],
            )
            qv_maps_all.append(np.asarray(qv_result["qv_map"], dtype=np.float32))
            qv_masks_all.append(np.asarray(qv_result["qv_mask"], dtype=np.float32))
            partial_curves_all.append(np.asarray(partial_result["partial_charge_curve"], dtype=np.float32))
            partial_masks_all.append(float(bool(partial_result["partial_charge_mask"])))
            relaxation_curves_all.append(np.asarray(relax_result["relaxation_curve"], dtype=np.float32))
            relaxation_masks_all.append(float(bool(relax_result["relaxation_mask"])))
            physics_features_all.append(np.asarray(physics_result["physics_features"], dtype=np.float32))
            physics_masks_all.append(np.asarray(physics_result["physics_feature_mask"], dtype=np.float32))
            curve_stats_rows.append(qv_result["curve_stats"])
            availability_rows.append(
                {
                    "qv_available": float(np.asarray(qv_result["qv_mask"]).mean() > 0),
                    "partial_charge_available": float(bool(partial_result["partial_charge_mask"])),
                    "relaxation_available": float(bool(relax_result["relaxation_mask"])),
                }
            )

        curve_stats_frame = pd.DataFrame(curve_stats_rows)
        for name in DEFAULT_QV_CURVE_STATS:
            feature_frame[name] = curve_stats_frame.get(name, 0.0)
        cycle_stats_matrix = np.stack(
            [
                pd.to_numeric(feature_frame.get(name, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
                for name in cycle_feature_names
            ],
            axis=-1,
        )
        qv_maps_all = np.stack(qv_maps_all).astype(np.float32)
        qv_masks_all = np.stack(qv_masks_all).astype(np.float32)
        partial_curves_all = np.stack(partial_curves_all).astype(np.float32)
        partial_masks_all = np.asarray(partial_masks_all, dtype=np.float32)
        relaxation_curves_all = np.stack(relaxation_curves_all).astype(np.float32)
        relaxation_masks_all = np.asarray(relaxation_masks_all, dtype=np.float32)
        physics_features_all = np.stack(physics_features_all).astype(np.float32)
        physics_masks_all = np.stack(physics_masks_all).astype(np.float32)

        max_start = len(cell_cycles) - lookback - horizon + 1
        if max_start <= 0:
            continue

        for start in range(0, max_start, stride):
            window_end = start + lookback
            target_end = window_end + horizon
            anchor_idx = window_end - 1
            future_soh = soh_values[window_end:target_end]
            if np.isnan(soh_values[anchor_idx]) or np.isnan(future_soh).any():
                continue
            target_delta_soh = future_soh - soh_values[anchor_idx]
            if not np.isfinite(target_delta_soh).all():
                continue

            window_physics = aggregate_window_physics_features(
                physics_features_all[start:window_end],
                physics_masks_all[start:window_end],
            )
            future_ops = operation_seq_full[window_end:target_end].copy()
            future_ops_mask = np.isfinite(future_ops).astype(np.float32)
            if future_ops_mode == "repeat_last":
                future_ops = np.repeat(operation_seq_full[anchor_idx : anchor_idx + 1], horizon, axis=0).astype(np.float32)
                future_ops_mask = np.ones_like(future_ops, dtype=np.float32)
            elif future_ops_mode == "none":
                future_ops = np.zeros((horizon, operation_seq_full.shape[-1]), dtype=np.float32)
                future_ops_mask = np.zeros_like(future_ops, dtype=np.float32)

            recent_soh_window = soh_values[start:window_end]
            recent_slope = recent_delta_mean(recent_soh_window, min(5, len(recent_soh_window) - 1))
            recent_curvature = _recent_curvature(recent_soh_window)
            throughput_recent = float(throughput_values[anchor_idx] - throughput_values[max(start, anchor_idx - min(5, lookback - 1))])
            anchor_row = cell_cycles.iloc[anchor_idx]
            chemistry = str(anchor_row.get("chemistry_family") or "Unknown")
            domain_label = build_domain_label(anchor_row.to_dict(), cfg.get("domain_labeling")) or str(split_row["domain_label"])

            missing_summary = {
                "qv": (1.0 - qv_masks_all[start:window_end].mean(axis=0)).astype(float).tolist(),
                "partial_charge": float(1.0 - partial_masks_all[start:window_end].mean()),
                "relaxation": float(1.0 - relaxation_masks_all[start:window_end].mean()),
                "physics_features": (1.0 - physics_masks_all[start:window_end].mean(axis=0)).astype(float).tolist(),
                "future_ops": float(1.0 - future_ops_mask.mean()) if future_ops_mask.size else 1.0,
            }

            case = CaseSample(
                case_id=len(cases),
                cell_uid=cell_uid,
                source_dataset=str(anchor_row.get("source_dataset") or split_row["source_dataset"]),
                raw_cell_id=str(split_row["raw_cell_id"]),
                split=split,
                domain_label=domain_label,
                window_start=start,
                window_end=window_end,
                target_start=window_end,
                target_end=target_end,
                cycle_idx_start=int(cell_cycles.iloc[start]["cycle_idx"]),
                cycle_idx_end=int(cell_cycles.iloc[anchor_idx]["cycle_idx"]),
                target_cycle_idx_start=int(cell_cycles.iloc[window_end]["cycle_idx"]),
                target_cycle_idx_end=int(cell_cycles.iloc[target_end - 1]["cycle_idx"]),
                chemistry_family=chemistry,
                temperature_bucket=str(anchor_row.get("temperature_bucket") or "unknown"),
                charge_rate_bucket=str(anchor_row.get("charge_rate_bucket") or "unknown"),
                discharge_policy_family=str(anchor_row.get("discharge_policy_family") or "unknown"),
                nominal_capacity_bucket=str(anchor_row.get("nominal_capacity_bucket") or "unknown"),
                voltage_window_bucket=str(anchor_row.get("voltage_window_bucket") or "unknown"),
                full_or_partial=str(anchor_row.get("full_or_partial") or "unknown"),
                anchor_soh=float(soh_values[anchor_idx]),
                anchor_capacity=float(capacity_values[anchor_idx]) if np.isfinite(capacity_values[anchor_idx]) else 0.0,
                recent_soh_slope=float(recent_slope),
                recent_soh_curvature=float(recent_curvature),
                throughput_recent=float(throughput_recent),
                degradation_stage=_degradation_stage(float(soh_values[anchor_idx])),
                target_delta_soh=target_delta_soh.astype(np.float32),
                target_soh=future_soh.astype(np.float32),
                cycle_stats=cycle_stats_matrix[start:window_end].astype(np.float32),
                soh_seq=recent_soh_window.astype(np.float32),
                qv_maps=qv_maps_all[start:window_end].astype(np.float32),
                qv_masks=qv_masks_all[start:window_end].astype(np.float32),
                partial_charge_curves=partial_curves_all[start:window_end].astype(np.float32),
                partial_charge_masks=partial_masks_all[start:window_end].astype(np.float32),
                relaxation_curves=relaxation_curves_all[start:window_end].astype(np.float32),
                relaxation_masks=relaxation_masks_all[start:window_end].astype(np.float32),
                physics_features=physics_features_all[start:window_end].astype(np.float32),
                physics_feature_masks=physics_masks_all[start:window_end].astype(np.float32),
                anchor_physics_features=window_physics["anchor_physics_features"].astype(np.float32),
                operation_seq=operation_seq_full[start:window_end].astype(np.float32),
                future_operation_seq=np.nan_to_num(future_ops, nan=0.0).astype(np.float32),
                future_operation_mask=np.nan_to_num(future_ops_mask, nan=0.0).astype(np.float32),
                tsfm_embedding=None,
                metadata={
                    "q_grid": q_grid.tolist(),
                    "split_manifest_domain": split_row["domain_label"],
                    "raw_file_path": cell.file_path,
                },
                feature_names={
                    "cycle_stats": cycle_feature_names,
                    "operation": operation_names,
                    "physics_feature_names": compute_physics_features(
                        np.zeros(partial_charge_segments, dtype=np.float32),
                        False,
                        np.zeros(relaxation_points, dtype=np.float32),
                        False,
                        {},
                    )["physics_feature_names"],
                    "qv_channels": QV_CHANNEL_NAMES,
                },
                missing_mask=missing_summary,
            )
            row = case.to_row_dict()
            row.update(_hashable_missing_summary(case))
            cases.append(case)
            case_rows.append(row)

    if not cases:
        raise ValueError("No case samples were built; check dataset size and config.")

    rows_df = pd.DataFrame(case_rows).sort_values("case_id").reset_index(drop=True)

    encoder = _optional_window_encoder(cfg)
    tsfm_embeddings = None
    if encoder is not None:
        try:
            encoder_input = np.stack(
                [
                    np.concatenate([case.soh_seq[:, None], case.cycle_stats], axis=-1).astype(np.float32)
                    for case in cases
                ]
            )
            tsfm_embeddings = encoder.encode(encoder_input).astype(np.float32)
            for case_idx, case in enumerate(cases):
                case.tsfm_embedding = tsfm_embeddings[case_idx]
            rows_df["has_tsfm_embedding"] = True
        except Exception as exc:
            print(f"[CaseBank] Skip tsfm embedding encoding: {exc}", flush=True)
            tsfm_embeddings = None

    arrays = {
        "case_cycle_stats.npy": np.stack([case.cycle_stats for case in cases]).astype(np.float32),
        "case_soh_seq.npy": np.stack([case.soh_seq for case in cases]).astype(np.float32),
        "case_qv_maps.npy": np.stack([case.qv_maps for case in cases]).astype(np.float32),
        "case_qv_masks.npy": np.stack([case.qv_masks for case in cases]).astype(np.float32),
        "case_partial_charge.npy": np.stack([case.partial_charge_curves for case in cases]).astype(np.float32),
        "case_partial_charge_mask.npy": np.stack([case.partial_charge_masks for case in cases]).astype(np.float32),
        "case_relaxation.npy": np.stack([case.relaxation_curves for case in cases]).astype(np.float32),
        "case_relaxation_mask.npy": np.stack([case.relaxation_masks for case in cases]).astype(np.float32),
        "case_physics_features.npy": np.stack([case.physics_features for case in cases]).astype(np.float32),
        "case_physics_feature_masks.npy": np.stack([case.physics_feature_masks for case in cases]).astype(np.float32),
        "case_anchor_physics_features.npy": np.stack([case.anchor_physics_features for case in cases]).astype(np.float32),
        "case_operation_seq.npy": np.stack([case.operation_seq for case in cases]).astype(np.float32),
        "case_future_ops.npy": np.stack([case.future_operation_seq for case in cases]).astype(np.float32),
        "case_future_ops_mask.npy": np.stack([case.future_operation_mask for case in cases]).astype(np.float32),
        "case_future_delta_soh.npy": np.stack([case.target_delta_soh for case in cases]).astype(np.float32),
        "case_future_soh.npy": np.stack([case.target_soh for case in cases]).astype(np.float32),
    }
    if tsfm_embeddings is not None:
        arrays["case_tsfm_embeddings.npy"] = tsfm_embeddings.astype(np.float32)

    _save_case_rows(rows_df, output_dir)
    for name, values in arrays.items():
        np.save(output_dir / name, values)

    normalization_stats = _fit_normalization_stats(
        rows_df,
        {
            "cycle_stats": arrays["case_cycle_stats.npy"],
            "soh_seq": arrays["case_soh_seq.npy"],
            "physics_features": arrays["case_physics_features.npy"],
            "anchor_physics_features": arrays["case_anchor_physics_features.npy"],
            "operation_seq": arrays["case_operation_seq.npy"],
            "future_ops": arrays["case_future_ops.npy"],
            **({"tsfm_embeddings": tsfm_embeddings} if tsfm_embeddings is not None else {}),
        },
    )
    (output_dir / "normalization_stats.json").write_text(json.dumps(normalization_stats, indent=2, ensure_ascii=True))

    feature_names = {
        "cycle_stats": cycle_feature_names,
        "qv_channels": QV_CHANNEL_NAMES,
        "physics_feature_names": cases[0].feature_names["physics_feature_names"],
        "operation": DEFAULT_OPERATION_FEATURES,
        "future_operation": DEFAULT_OPERATION_FEATURES,
    }
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2, ensure_ascii=True))

    build_log = {
        "total_cases": int(len(rows_df)),
        "cases_by_split": {k: int(v) for k, v in rows_df["split"].value_counts().to_dict().items()},
        "cases_by_dataset": {k: int(v) for k, v in rows_df["source_dataset"].value_counts().to_dict().items()},
        "cases_by_chemistry": {k: int(v) for k, v in rows_df["chemistry_family"].value_counts().to_dict().items()},
        "qv_map_availability": float(arrays["case_qv_masks.npy"].mean()),
        "partial_charge_availability": float(arrays["case_partial_charge_mask.npy"].mean()),
        "relaxation_availability": float(arrays["case_relaxation_mask.npy"].mean()),
        "physics_feature_availability": float(arrays["case_physics_feature_masks.npy"].mean()),
        "future_ops_availability": float(arrays["case_future_ops_mask.npy"].mean()),
        "missingness_summary": {
            "qv": (1.0 - arrays["case_qv_masks.npy"].mean(axis=(0, 1))).astype(float).tolist(),
            "partial_charge": float(1.0 - arrays["case_partial_charge_mask.npy"].mean()),
            "relaxation": float(1.0 - arrays["case_relaxation_mask.npy"].mean()),
            "physics_features": (1.0 - arrays["case_physics_feature_masks.npy"].mean(axis=(0, 1))).astype(float).tolist(),
            "future_ops": float(1.0 - arrays["case_future_ops_mask.npy"].mean()),
        },
        "feature_stats": normalization_stats,
    }
    (output_dir / "case_bank_build_log.json").write_text(json.dumps(build_log, indent=2, ensure_ascii=True))
    pd.DataFrame([{"key": key, "value_json": json.dumps(value, ensure_ascii=True)} for key, value in build_log.items()]).to_csv(
        output_dir / "case_bank_build_log.csv",
        index=False,
    )

    _save_case_bank_figures(rows_df, arrays, feature_names, figure_dir)

    return {
        "output_dir": str(output_dir),
        "num_cases": int(len(rows_df)),
        "splits": build_log["cases_by_split"],
        "has_tsfm_embeddings": bool(tsfm_embeddings is not None),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build battery SOH forecasting case bank")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = build_case_bank(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
