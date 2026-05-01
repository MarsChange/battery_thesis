"""experiments.run_subset_rag_retrieval

使用真实电池数据子集构建一个可控的 cross-cell RAG 检索数据库，并对多组检索
特征设置做并排对照。

当前脚本默认对以下三组检索设置同时运行：
1. `state_metadata`：只使用 SOH 状态距离和元信息距离。
2. `state_metadata_qv`：在 `state_metadata` 基础上加入 Q-V 曲线形状距离。
3. `state_metadata_qv_physics`：在 `state_metadata_qv` 基础上进一步加入物理启发特征距离。

实验目的：
- 检查 query 来自数据库内部、但禁止 same-cell retrieval 时，top-k 能否稳定返回其他电池；
- 对比不同特征组合下 top-1 的相似性是否改善；
- 特别关注 HUST 的 LFP 电池是否存在 “top-1 局部状态最接近，但 top-2 更像整条 SOH / dQ-dV” 的情况。

输出结构：
- `case_bank/`：子集 case bank；
- `settings/<setting_name>/`：每个检索设置单独的 retrieval config、csv、summary 和图像；
- 根目录下的 `comparison_*.csv`、`comparison_summary.md` 和 `figures/comparison/`：聚合对照结果。
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from battery_data.adapters.hust import load_hust_cells
from battery_data.adapters.tju import load_tju_cells
from battery_data.build_case_bank import _load_raw_cycle_tables, build_case_bank_from_cells, load_config
from battery_data.canonicalize import assign_cell_uids
from battery_data.domain_labeling import build_domain_label
from battery_data.schema import CanonicalCell
from retrieval.multistage_retriever import MultiStageBatteryRetriever, RetrievalResult
from retrieval.physics_distance import compute_retrieval_confidence, normalized_l2


DEFAULT_OUTPUT_DIR = Path("tests/artifacts/subset_rag_cross_cell_retrieval")

RETRIEVAL_SETTING_PRESETS: dict[str, dict[str, object]] = {
    "state_metadata": {
        "description": "只使用 SOH 状态距离和元信息距离。适合做最简基线，但对同 chemistry 内部排序约束较弱。",
        "enabled_components": ["d_soh_state", "d_metadata"],
        "weights": {"d_soh_state": 0.70, "d_metadata": 0.30},
    },
    "state_metadata_qv": {
        "description": "在 SOH 状态和元信息之外加入 Q-V 曲线形状距离，用于区分局部状态相似但曲线形态不同的电池。",
        "enabled_components": ["d_soh_state", "d_qv_shape", "d_metadata"],
        "weights": {"d_soh_state": 0.50, "d_qv_shape": 0.30, "d_metadata": 0.20},
    },
    "state_metadata_qv_physics": {
        "description": "进一步加入 DeltaV/R 与 partial charge 物理 proxy 距离，用于在 Q-V 之外补充退化状态信息。",
        "enabled_components": ["d_soh_state", "d_qv_shape", "d_physics", "d_metadata"],
        "weights": {"d_soh_state": 0.40, "d_qv_shape": 0.30, "d_physics": 0.20, "d_metadata": 0.10},
    },
    "state_metadata_qv_physics_hust_hv_dqdv": {
        "description": "HUST/LFP 特殊处理：在 state_metadata_qv_physics 基础上，对 HUST query 使用 3.2V 之后的 dQ/dV 曲线距离参与最终重排序。",
        "enabled_components": ["d_soh_state", "d_qv_shape", "d_physics", "d_metadata"],
        "weights": {"d_soh_state": 0.35, "d_qv_shape": 0.25, "d_physics": 0.20, "d_metadata": 0.20},
        "hust_high_voltage_dqdv": True,
        "hust_high_voltage_min_v": 3.2,
        "hust_high_voltage_weight": 0.45,
    },
}


def _folder_alias(cell: CanonicalCell) -> str:
    """把原始数据来源映射为实验里稳定可读的数据集标签。"""

    if cell.source_dataset == "hust":
        return "hust_lfp"
    folder = str(cell.source_info.get("folder", ""))
    if folder == "Dataset_1_NCA_battery":
        return "tju_dataset_1_nca"
    if folder == "Dataset_2_NCM_battery":
        return "tju_dataset_2_ncm"
    return f"{cell.source_dataset}_unknown"


def _representative_row(cell: CanonicalCell) -> pd.Series:
    return cell.cycles.iloc[min(len(cell.cycles) - 1, 0)]


def _sample_cells(cells: Sequence[CanonicalCell], count: int, rng: np.random.RandomState) -> list[CanonicalCell]:
    ordered = sorted(cells, key=lambda item: item.raw_cell_id)
    if count <= 0 or len(ordered) <= count:
        return list(ordered)
    chosen_idx = np.sort(rng.choice(len(ordered), size=count, replace=False))
    return [ordered[int(idx)] for idx in chosen_idx.tolist()]


def _prepare_reference_cells(
    cfg: Dict[str, object],
    *,
    max_hust_cells: int,
    max_nca_cells: int,
    max_ncm_cells: int,
) -> tuple[list[CanonicalCell], dict[str, object]]:
    """从真实数据中抽样构建 reference database。"""

    rng = np.random.RandomState(int(cfg.get("random_seed", 7)))
    hust_cfg = dict(cfg.get("datasets", {}).get("hust", {}) or {})
    tju_cfg = dict(cfg.get("datasets", {}).get("tju", {}) or {})
    hust_cfg["max_cells"] = max_hust_cells

    hust_cells = load_hust_cells(hust_cfg)
    tju_cells = load_tju_cells(tju_cfg)

    tju_dataset_1 = [cell for cell in tju_cells if str(cell.source_info.get("folder", "")) == "Dataset_1_NCA_battery"]
    tju_dataset_2 = [cell for cell in tju_cells if str(cell.source_info.get("folder", "")) == "Dataset_2_NCM_battery"]

    selected_cells = []
    selected_cells.extend(_sample_cells(hust_cells, max_hust_cells, rng))
    selected_cells.extend(_sample_cells(tju_dataset_1, max_nca_cells, rng))
    selected_cells.extend(_sample_cells(tju_dataset_2, max_ncm_cells, rng))

    all_cells = assign_cell_uids(selected_cells, prefix="subset")
    metadata = {
        "num_hust_reference_cells": int(sum(1 for cell in all_cells if _folder_alias(cell) == "hust_lfp")),
        "num_nca_reference_cells": int(sum(1 for cell in all_cells if _folder_alias(cell) == "tju_dataset_1_nca")),
        "num_ncm_reference_cells": int(sum(1 for cell in all_cells if _folder_alias(cell) == "tju_dataset_2_ncm")),
        "db_cell_uids": sorted(str(cell.cycles["cell_uid"].iloc[0]) for cell in all_cells),
        "query_selection_policy": "random source_train cells, latest window per selected cell",
    }
    return all_cells, metadata


def _build_manifest(cells: Sequence[CanonicalCell]) -> pd.DataFrame:
    """为子集实验构建 split manifest。所有 cell 都进入 source_train。"""

    rows = []
    for cell in cells:
        representative = _representative_row(cell)
        chemistry_family = str(representative.get("chemistry_family") or "Unknown")
        rows.append(
            {
                "cell_uid": str(representative["cell_uid"]),
                "source_dataset": _folder_alias(cell),
                "raw_cell_id": cell.raw_cell_id,
                "file_path": cell.file_path,
                "n_cycles": int(len(cell.cycles)),
                "domain_label": build_domain_label({"chemistry_family": chemistry_family}, {"mode": "chemistry_only"}),
                "split": "source_train",
            }
        )
    return pd.DataFrame(rows).sort_values(["source_dataset", "raw_cell_id"]).reset_index(drop=True)


def _make_subset_cfg(cfg: Dict[str, object], output_dir: Path) -> Dict[str, object]:
    """生成只用于本次子集实验的 case bank 配置。"""

    subset_cfg = json.loads(json.dumps(cfg))
    subset_cfg["output_dir"] = str(output_dir / "case_bank")
    subset_cfg["adapter_cache"] = True
    subset_cfg["model"] = dict(subset_cfg.get("model", {}) or {})
    subset_cfg["domain_labeling"] = {"mode": "chemistry_only"}
    subset_cfg.pop("encoder", None)
    return subset_cfg


def _parse_setting_names(text: str | None) -> list[str]:
    """解析命令行传入的 setting 名称列表。"""

    if text is None or not str(text).strip():
        return list(RETRIEVAL_SETTING_PRESETS.keys())
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    unknown = [name for name in names if name not in RETRIEVAL_SETTING_PRESETS]
    if unknown:
        raise ValueError(f"Unknown retrieval setting(s): {unknown}")
    return names


def _make_retrieval_config(
    base_path: Path,
    output_path: Path,
    *,
    top_k: int,
    setting_name: str,
) -> dict:
    """根据命名检索设置生成 retrieval yaml。"""

    preset = RETRIEVAL_SETTING_PRESETS[setting_name]
    enabled_components = set(str(name) for name in preset["enabled_components"])
    weight_overrides = {str(key): float(value) for key, value in dict(preset.get("weights", {}) or {}).items()}

    rag_cfg = yaml.safe_load(base_path.read_text())
    for component_name in ["d_soh_state", "d_qv_shape", "d_physics", "d_operation", "d_metadata"]:
        rag_cfg[component_name] = component_name in enabled_components
    weights = dict(rag_cfg.get("weights", {}) or {})
    for component_name, value in weight_overrides.items():
        weights[component_name] = float(value)
    rag_cfg["weights"] = weights
    rag_cfg["final_topk"]["top_k"] = int(top_k)
    rag_cfg["final_topk"]["max_neighbors_per_cell"] = 1
    rag_cfg["experiment_setting_name"] = setting_name
    rag_cfg["experiment_setting_description"] = str(preset["description"])
    output_path.write_text(yaml.safe_dump(rag_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return rag_cfg


def _case_rows(case_bank_dir: Path) -> pd.DataFrame:
    parquet_path = case_bank_dir / "case_rows.parquet"
    csv_path = case_bank_dir / "case_rows.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(csv_path)


def _select_query_case_ids(rows: pd.DataFrame, *, num_query_cells: int, random_seed: int) -> list[int]:
    """从 source_train 里随机选若干 cell，并取每个 cell 的最后一个窗口作为 query。"""

    source_rows = rows.loc[rows["split"].astype(str) == "source_train"].copy()
    if source_rows.empty:
        return []
    candidate_cells = sorted(source_rows["cell_uid"].astype(str).unique().tolist())
    rng = np.random.RandomState(int(random_seed))
    if num_query_cells < len(candidate_cells):
        selected_cells = sorted(rng.choice(candidate_cells, size=num_query_cells, replace=False).tolist())
    else:
        selected_cells = candidate_cells
    selected_rows = (
        source_rows.loc[source_rows["cell_uid"].astype(str).isin(selected_cells)]
        .sort_values(["cell_uid", "window_end", "case_id"])
        .groupby("cell_uid", as_index=False)
        .tail(1)
        .sort_values("case_id")
    )
    return selected_rows["case_id"].astype(int).tolist()


def _decode_metadata_json(text: object) -> dict[str, object]:
    if text is None:
        return {}
    if isinstance(text, dict):
        return dict(text)
    try:
        return json.loads(str(text))
    except Exception:
        return {}


@lru_cache(maxsize=256)
def _raw_cycle_map(source_dataset: str, file_path: str) -> dict[int, pd.DataFrame]:
    if source_dataset.startswith("tju"):
        source_dataset = "tju"
    elif source_dataset.startswith("hust"):
        source_dataset = "hust"
    return _load_raw_cycle_tables(source_dataset, file_path)


def _charge_mask(cycle_df: pd.DataFrame) -> pd.Series:
    if "step" in cycle_df.columns:
        step_text = cycle_df["step"].fillna("").astype(str).str.lower()
        has_charge_label = step_text.str.contains("charge", regex=False)
        if has_charge_label.any():
            return has_charge_label & pd.to_numeric(cycle_df["current"], errors="coerce").gt(0)
    return pd.to_numeric(cycle_df["current"], errors="coerce").gt(0)


def _prepare_dqdv_curve(
    cycle_df: pd.DataFrame,
    *,
    voltage_bin_v: float = 0.002,
    min_points: int = 25,
    interpolation_points: int = 300,
    smoothing_window: int = 11,
    derivative_window: int = 11,
) -> tuple[np.ndarray, np.ndarray] | None:
    charge_df = cycle_df.loc[_charge_mask(cycle_df), ["voltage", "capacity"]].copy()
    charge_df["voltage"] = pd.to_numeric(charge_df["voltage"], errors="coerce")
    charge_df["capacity"] = pd.to_numeric(charge_df["capacity"], errors="coerce")
    charge_df = charge_df.replace([np.inf, -np.inf], np.nan).dropna()
    if charge_df.empty:
        return None

    charge_df = charge_df.sort_values("voltage")
    charge_df["capacity"] = charge_df["capacity"] - float(charge_df["capacity"].min())
    if voltage_bin_v > 0:
        charge_df["voltage_bin"] = (charge_df["voltage"] / voltage_bin_v).round() * voltage_bin_v
        charge_df = (
            charge_df.groupby("voltage_bin", as_index=False, sort=True)
            .agg(voltage=("voltage", "mean"), capacity=("capacity", "median"))
            .sort_values("voltage")
        )
    if len(charge_df) < min_points:
        return None

    voltage = charge_df["voltage"].to_numpy(dtype=np.float32)
    capacity = charge_df["capacity"].to_numpy(dtype=np.float32)
    if np.ptp(voltage) <= 1e-3 or np.ptp(capacity) <= 1e-6:
        return None

    grid = np.linspace(float(voltage.min()), float(voltage.max()), interpolation_points, dtype=np.float32)
    capacity_interp = np.interp(grid, voltage, capacity).astype(np.float32)
    capacity_interp = (
        pd.Series(capacity_interp)
        .rolling(window=min(smoothing_window, interpolation_points), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    dqdv = np.gradient(capacity_interp, grid).astype(np.float32)
    dqdv = (
        pd.Series(dqdv)
        .rolling(window=min(derivative_window, interpolation_points), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    finite_mask = np.isfinite(dqdv)
    if int(finite_mask.sum()) < min_points:
        return None
    clip_limit = float(np.quantile(np.abs(dqdv[finite_mask]), 0.995))
    if clip_limit > 0:
        dqdv = np.clip(dqdv, -clip_limit, clip_limit)
    return grid, dqdv.astype(np.float32)


def _dqdv_distance(
    curve_a: tuple[np.ndarray, np.ndarray] | None,
    curve_b: tuple[np.ndarray, np.ndarray] | None,
    *,
    voltage_min: float | None = None,
) -> float:
    if curve_a is None or curve_b is None:
        return float("nan")
    grid_a, dqdv_a = curve_a
    grid_b, dqdv_b = curve_b
    overlap_min = max(float(grid_a.min()), float(grid_b.min()), float(voltage_min) if voltage_min is not None else -np.inf)
    overlap_max = min(float(grid_a.max()), float(grid_b.max()))
    if overlap_max - overlap_min <= 0.05:
        return float("nan")
    common_grid = np.linspace(overlap_min, overlap_max, 240, dtype=np.float32)
    interp_a = np.interp(common_grid, grid_a, dqdv_a).astype(np.float32)
    interp_b = np.interp(common_grid, grid_b, dqdv_b).astype(np.float32)
    scale = np.maximum(np.maximum(np.abs(interp_a), np.abs(interp_b)), 1e-4)
    return float(normalized_l2(interp_a, interp_b, scale=scale))


def _filter_dqdv_curve(
    curve: tuple[np.ndarray, np.ndarray] | None,
    *,
    voltage_min: float,
    min_points: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    """复用标准 dQ/dV 曲线，仅裁剪高电压区间，不重新求导。"""

    if curve is None:
        return None
    voltage, dqdv = curve
    voltage_arr = np.asarray(voltage, dtype=np.float32).reshape(-1)
    dqdv_arr = np.asarray(dqdv, dtype=np.float32).reshape(-1)
    if voltage_arr.size != dqdv_arr.size:
        return None
    mask = np.isfinite(voltage_arr) & np.isfinite(dqdv_arr) & (voltage_arr >= float(voltage_min))
    if int(mask.sum()) < int(min_points):
        return None
    return voltage_arr[mask].astype(np.float32), dqdv_arr[mask].astype(np.float32)


def _load_case_curve(row: pd.Series) -> tuple[pd.DataFrame | None, tuple[np.ndarray, np.ndarray] | None]:
    metadata = _decode_metadata_json(row.get("metadata_json"))
    file_path = str(metadata.get("raw_file_path") or "")
    if not file_path:
        return None, None
    raw_map = _raw_cycle_map(str(row["source_dataset"]), file_path)
    cycle_idx = int(row["cycle_idx_end"])
    cycle_df = raw_map.get(cycle_idx)
    if cycle_df is None or cycle_df.empty:
        return None, None
    return cycle_df, _prepare_dqdv_curve(cycle_df)


def _softmax_distances(distances: np.ndarray, mask: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    logits = -np.asarray(distances, dtype=np.float32) / max(float(temperature), 1e-6)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if logits.size == 0:
        return logits.astype(np.float32)
    logits = np.where(mask_arr > 0, logits, -1e9)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits) * mask_arr
    denom = float(weights.sum())
    if denom <= 0:
        return np.zeros_like(logits, dtype=np.float32)
    return (weights / denom).astype(np.float32)


def _retrieve_with_hust_high_voltage_dqdv(
    retriever: MultiStageBatteryRetriever,
    rows: pd.DataFrame,
    query_case_id: int,
    *,
    high_voltage_min_v: float,
    high_voltage_weight: float,
) -> RetrievalResult:
    """HUST/LFP 专用检索：用 3.2V 之后的 dQ/dV 曲线距离参与最终重排序。"""

    query_idx = retriever.case_id_to_index[int(query_case_id)]
    query_row = rows.iloc[query_idx]
    if str(query_row["source_dataset"]) != "hust_lfp":
        return retriever.retrieve(int(query_case_id))

    candidate_case_ids = retriever._hard_filter_candidate_ids(query_idx, retriever._coarse_candidates(query_idx))
    candidate_case_ids = candidate_case_ids[: retriever.top_m]
    horizon = int(retriever.arrays["future_delta"].shape[1])
    top_k = int(retriever.top_k)
    if candidate_case_ids.size == 0:
        zeros = np.zeros(top_k, dtype=np.float32)
        return RetrievalResult(
            query_case_id=int(query_case_id),
            neighbor_case_ids=np.full(top_k, -1, dtype=np.int64),
            retrieval_mask=zeros.copy(),
            d_soh_state=zeros.copy(),
            d_qv_shape=zeros.copy(),
            d_physics=zeros.copy(),
            d_operation=np.full(top_k, np.nan, dtype=np.float32),
            d_metadata=zeros.copy(),
            composite_distance=np.full(top_k, np.inf, dtype=np.float32),
            retrieval_confidence=0.0,
            retrieval_alpha=zeros.copy(),
            neighbor_future_delta_soh=np.zeros((top_k, horizon), dtype=np.float32),
        )

    query_curve = _filter_dqdv_curve(_load_case_curve(query_row)[1], voltage_min=high_voltage_min_v)
    bundles: list[dict[str, float]] = []
    for case_id in candidate_case_ids.tolist():
        ref_idx = retriever.case_id_to_index[int(case_id)]
        ref_row = rows.iloc[ref_idx]
        bundle = retriever._distance_bundle(query_idx, ref_idx)
        ref_curve = _filter_dqdv_curve(_load_case_curve(ref_row)[1], voltage_min=high_voltage_min_v)
        high_voltage_distance = _dqdv_distance(
            query_curve,
            ref_curve,
            voltage_min=high_voltage_min_v,
        )
        base_distance = float(bundle["composite_distance"])
        if np.isfinite(high_voltage_distance):
            weight = float(np.clip(high_voltage_weight, 0.0, 1.0))
            bundle["composite_distance"] = (1.0 - weight) * base_distance + weight * float(high_voltage_distance)
        bundle["base_composite_distance"] = base_distance
        bundle["dqdv_high_voltage_distance"] = float(high_voltage_distance)
        bundles.append(bundle)

    bundle_frame = pd.DataFrame(bundles)
    order = np.argsort(bundle_frame["composite_distance"].to_numpy(dtype=np.float32))
    candidate_case_ids = candidate_case_ids[order]
    bundle_frame = bundle_frame.iloc[order].reset_index(drop=True)
    selected_case_ids, mmr_scores = retriever._apply_mmr(candidate_case_ids, bundle_frame["composite_distance"].to_numpy(dtype=np.float32))
    if selected_case_ids.size == 0:
        selected_case_ids = candidate_case_ids[:top_k]
        mmr_scores = np.ones(len(selected_case_ids), dtype=np.float32)

    selected_positions = [int(np.flatnonzero(candidate_case_ids == case_id)[0]) for case_id in selected_case_ids.tolist()]
    selected_frame = bundle_frame.iloc[selected_positions].reset_index(drop=True)
    k = min(len(selected_case_ids), top_k)

    neighbor_case_ids = np.full(top_k, -1, dtype=np.int64)
    retrieval_mask = np.zeros(top_k, dtype=np.float32)
    composite_distance = np.full(top_k, np.inf, dtype=np.float32)
    d_soh_state = np.zeros(top_k, dtype=np.float32)
    d_qv_shape = np.zeros(top_k, dtype=np.float32)
    d_physics = np.zeros(top_k, dtype=np.float32)
    d_operation = np.full(top_k, np.nan, dtype=np.float32)
    d_metadata = np.zeros(top_k, dtype=np.float32)
    stage_distance = np.zeros(top_k, dtype=np.float32)
    missing_penalty = np.zeros(top_k, dtype=np.float32)
    reference_compatibility_score = np.zeros(top_k, dtype=np.float32)
    mmr_diversity_score = np.zeros(top_k, dtype=np.float32)
    retrieval_alpha = np.zeros(top_k, dtype=np.float32)
    neighbor_future_delta_soh = np.zeros((top_k, horizon), dtype=np.float32)
    neighbor_metadata = [{} for _ in range(top_k)]

    if k > 0:
        row_indices = [retriever.case_id_to_index[int(case_id)] for case_id in selected_case_ids[:k].tolist()]
        neighbor_case_ids[:k] = selected_case_ids[:k]
        retrieval_mask[:k] = 1.0
        composite_distance[:k] = selected_frame["composite_distance"].to_numpy(dtype=np.float32)[:k]
        d_soh_state[:k] = selected_frame["d_soh_state"].to_numpy(dtype=np.float32)[:k]
        d_qv_shape[:k] = selected_frame["d_qv_shape"].to_numpy(dtype=np.float32)[:k]
        d_physics[:k] = selected_frame["d_physics"].to_numpy(dtype=np.float32)[:k]
        d_metadata[:k] = selected_frame["d_metadata"].to_numpy(dtype=np.float32)[:k]
        if "stage_distance" in selected_frame:
            stage_distance[:k] = selected_frame["stage_distance"].to_numpy(dtype=np.float32)[:k]
        if "missing_penalty" in selected_frame:
            missing_penalty[:k] = selected_frame["missing_penalty"].to_numpy(dtype=np.float32)[:k]
        if "reference_compatibility_score" in selected_frame:
            reference_compatibility_score[:k] = selected_frame["reference_compatibility_score"].to_numpy(dtype=np.float32)[:k]
        mmr_diversity_score[: min(len(mmr_scores), top_k)] = mmr_scores[:top_k]
        retrieval_alpha[:k] = _softmax_distances(composite_distance[:k], retrieval_mask[:k], temperature=0.1)
        neighbor_future_delta_soh[:k] = np.asarray(retriever.arrays["future_delta"][row_indices], dtype=np.float32)
        neighbor_metadata[:k] = [
            retriever.case_rows.iloc[row_idx][
                [
                    "case_id",
                    "cell_uid",
                    "chemistry_family",
                    "domain_label",
                    "source_dataset",
                    "degradation_stage",
                    "anchor_soh",
                    "recent_soh_slope",
                    "voltage_window_bucket",
                ]
            ].to_dict()
            for row_idx in row_indices
        ]

    confidence_payload = {
        "composite_distance": composite_distance[:k],
        "feature_availability_ratio": float(selected_frame["feature_availability_ratio"].mean()) if k > 0 else 0.0,
        "chemistry_match_rate": float(selected_frame["chemistry_match"].mean()) if k > 0 else 0.0,
        "domain_match_rate": float(selected_frame["domain_match"].mean()) if k > 0 else 0.0,
    }
    retrieval_confidence = compute_retrieval_confidence(confidence_payload, retriever.rag_config) if k > 0 else 0.0
    explain_json = {
        "enabled_distance_components": retriever.enabled_component_names,
        "hust_high_voltage_dqdv": True,
        "high_voltage_min_v": float(high_voltage_min_v),
        "high_voltage_weight": float(high_voltage_weight),
        "selected_neighbors": [
            {
                "case_id": int(neighbor_case_ids[pos]),
                "dqdv_high_voltage_distance": float(selected_frame.iloc[pos].get("dqdv_high_voltage_distance", np.nan)),
                "base_composite_distance": float(selected_frame.iloc[pos].get("base_composite_distance", np.nan)),
                "composite_distance": float(composite_distance[pos]),
            }
            for pos in range(k)
        ],
    }
    return RetrievalResult(
        query_case_id=int(query_case_id),
        neighbor_case_ids=neighbor_case_ids,
        retrieval_mask=retrieval_mask,
        d_soh_state=d_soh_state,
        d_qv_shape=d_qv_shape,
        d_physics=d_physics,
        d_operation=d_operation,
        d_metadata=d_metadata,
        composite_distance=composite_distance,
        retrieval_confidence=float(retrieval_confidence),
        retrieval_alpha=retrieval_alpha,
        neighbor_future_delta_soh=neighbor_future_delta_soh,
        neighbor_metadata=neighbor_metadata,
        explain_json=explain_json,
        stage_distance=stage_distance,
        missing_penalty=missing_penalty,
        reference_compatibility_score=reference_compatibility_score,
        mmr_diversity_score=mmr_diversity_score,
    )


def _qv_map_high_voltage_dqdv_curve(
    retriever: MultiStageBatteryRetriever,
    case_idx: int,
    *,
    voltage_min: float = 3.2,
    min_points: int = 20,
) -> tuple[np.ndarray, np.ndarray] | None:
    """从 case bank 的 anchor-cycle Vc(Q) 中提取高电压 dQ/dV 曲线。"""

    qv_map = np.asarray(retriever.arrays["qv_maps"][case_idx, -1], dtype=np.float32)
    qv_mask = np.asarray(retriever.arrays["qv_masks"][case_idx, -1], dtype=np.float32)
    if qv_map.ndim != 2 or qv_map.shape[0] < 1 or qv_mask.size < 1 or qv_mask[0] <= 0:
        return None
    voltage = np.asarray(qv_map[0], dtype=np.float32).reshape(-1)
    q_axis = np.linspace(0.0, 1.0, voltage.size, dtype=np.float32)
    mask = np.isfinite(voltage) & (voltage >= float(voltage_min))
    if int(mask.sum()) < min_points:
        return None
    voltage = voltage[mask]
    q_axis = q_axis[mask]
    order = np.argsort(voltage)
    voltage = voltage[order]
    q_axis = q_axis[order]
    unique_voltage, unique_indices = np.unique(voltage, return_index=True)
    if unique_voltage.size < min_points:
        return None
    q_axis = q_axis[unique_indices]
    q_axis = (
        pd.Series(q_axis)
        .rolling(window=min(9, unique_voltage.size), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    dqdv = np.gradient(q_axis, unique_voltage).astype(np.float32)
    dqdv = (
        pd.Series(dqdv)
        .rolling(window=min(9, unique_voltage.size), center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float32)
    )
    finite = np.isfinite(dqdv)
    if int(finite.sum()) < min_points:
        return None
    clip_limit = float(np.quantile(np.abs(dqdv[finite]), 0.995))
    if clip_limit > 0:
        dqdv = np.clip(dqdv, -clip_limit, clip_limit)
    return unique_voltage.astype(np.float32), dqdv.astype(np.float32)


@lru_cache(maxsize=256)
def _full_life_curve(source_dataset: str, file_path: str) -> tuple[np.ndarray, np.ndarray] | None:
    """加载单个电池的完整归一化 SOH 曲线。"""

    if not file_path:
        return None
    raw_map = _raw_cycle_map(source_dataset, file_path)
    if not raw_map:
        return None
    ordered_cycles = sorted(raw_map.keys())
    soh_seq = []
    cycle_axis = []
    for cycle_idx in ordered_cycles:
        cycle_df = raw_map[cycle_idx]
        capacity = pd.to_numeric(cycle_df.get("capacity"), errors="coerce")
        if capacity is None or capacity.dropna().empty:
            continue
        soh_seq.append(float(capacity.max()))
        cycle_axis.append(int(cycle_idx))
    if not soh_seq:
        return None
    soh_arr = np.asarray(soh_seq, dtype=np.float32)
    base = float(soh_arr[0]) if abs(float(soh_arr[0])) > 1e-6 else 1.0
    return np.asarray(cycle_axis, dtype=np.int32), (soh_arr / base).astype(np.float32)


def _full_soh_rmse(query_row: pd.Series, neighbor_row: pd.Series) -> float:
    """计算两块电池完整归一化 SOH 曲线的 RMSE。"""

    query_meta = _decode_metadata_json(query_row.get("metadata_json"))
    ref_meta = _decode_metadata_json(neighbor_row.get("metadata_json"))
    query_curve = _full_life_curve(str(query_row["source_dataset"]), str(query_meta.get("raw_file_path") or ""))
    ref_curve = _full_life_curve(str(neighbor_row["source_dataset"]), str(ref_meta.get("raw_file_path") or ""))
    if query_curve is None or ref_curve is None:
        return float("nan")
    _, query_soh = query_curve
    _, ref_soh = ref_curve
    if len(query_soh) < 10 or len(ref_soh) < 10:
        return float("nan")
    grid = np.linspace(0.0, 1.0, 400, dtype=np.float32)
    query_interp = np.interp(grid, np.linspace(0.0, 1.0, len(query_soh), dtype=np.float32), query_soh).astype(np.float32)
    ref_interp = np.interp(grid, np.linspace(0.0, 1.0, len(ref_soh), dtype=np.float32), ref_soh).astype(np.float32)
    return float(np.sqrt(np.mean((query_interp - ref_interp) ** 2)))


def _plot_soh_comparison(query_row: pd.Series, neighbor_rows: Sequence[pd.Series], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.2), dpi=180)

    def _plot_cell(row: pd.Series, label: str, color: str, linewidth: float, linestyle: str = "-") -> None:
        metadata = _decode_metadata_json(row.get("metadata_json"))
        file_path = str(metadata.get("raw_file_path") or "")
        curve = _full_life_curve(str(row["source_dataset"]), file_path)
        if curve is None:
            return
        ordered_cycles, normalized_soh = curve
        axis.plot(ordered_cycles, normalized_soh, color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        axis.scatter([int(row["cycle_idx_end"])], [float(row["anchor_soh"])], color=color, s=30, zorder=4)

    _plot_cell(query_row, f"Query | {query_row['cell_uid']}", "#dc2626", 2.4)
    palette = ["#2563eb", "#16a34a", "#9333ea"]
    styles = ["--", "-", "-."]
    for rank, (neighbor_row, color, linestyle) in enumerate(zip(neighbor_rows, palette, styles), start=1):
        label_suffix = f"{neighbor_row['cell_uid']} | {neighbor_row['source_dataset']}"
        _plot_cell(neighbor_row, f"Top-{rank} | {label_suffix}", color, 1.8, linestyle=linestyle)

    axis.set_title("SOH comparison for cross-cell top-k retrieval")
    axis.set_xlabel("Cycle index")
    axis.set_ylabel("Normalized SOH")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_dqdv_comparison(query_row: pd.Series, neighbor_rows: Sequence[pd.Series], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.2), dpi=180)
    query_curve = _load_case_curve(query_row)[1]
    if query_curve is not None:
        axis.plot(query_curve[0], query_curve[1], color="#dc2626", linewidth=2.4, label=f"Query | {query_row['cell_uid']}")
    else:
        axis.text(0.03, 0.95, "Query dQ/dV unavailable", transform=axis.transAxes, va="top", ha="left", color="#dc2626")

    palette = ["#2563eb", "#16a34a", "#9333ea"]
    styles = ["--", "-", "-."]
    for rank, (neighbor_row, color, linestyle) in enumerate(zip(neighbor_rows, palette, styles), start=1):
        curve = _load_case_curve(neighbor_row)[1]
        if curve is not None:
            label_suffix = f"{neighbor_row['cell_uid']} | {neighbor_row['source_dataset']}"
            axis.plot(curve[0], curve[1], color=color, linewidth=1.8, linestyle=linestyle, label=f"Top-{rank} | {label_suffix}")
    axis.set_title("dQ-dV comparison for cross-cell top-k retrieval")
    axis.set_xlabel("Voltage (V)")
    axis.set_ylabel("dQ/dV")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _setting_summary_markdown(
    output_dir: Path,
    setting_name: str,
    retrieval_cfg_path: Path,
    query_df: pd.DataFrame,
    pair_df: pd.DataFrame,
) -> None:
    """写入单个检索设置的 summary.md。"""

    preset = RETRIEVAL_SETTING_PRESETS[setting_name]
    lines = [
        f"# 子集 RAG 检索设置总结: {setting_name}",
        "",
        f"- 检索设置说明: {preset['description']}",
        f"- retrieval 配置: `{retrieval_cfg_path}`",
        f"- 启用距离分量: {', '.join(str(item) for item in preset['enabled_components'])}",
        "",
    ]
    if not pair_df.empty:
        lines.extend(
            [
                "## top-k 平均指标",
                f"- 平均 `d_soh_state`: {_safe_mean(pair_df['d_soh_state']):.4f}",
                f"- 平均 `d_qv_shape`: {_safe_mean(pair_df['d_qv_shape']):.4f}",
                f"- 平均 `d_physics`: {_safe_mean(pair_df['d_physics']):.4f}",
                f"- 平均 `d_metadata`: {_safe_mean(pair_df['d_metadata']):.4f}",
                f"- 平均 `composite_distance`: {_safe_mean(pair_df['composite_distance']):.4f}",
                f"- 平均 dQ-dV 诊断距离: {_safe_mean(pair_df['dqdv_distance']):.4f}",
                f"- 平均 3.2V+ dQ-dV 诊断距离: {_safe_mean(pair_df['dqdv_high_voltage_distance']):.4f}",
                f"- 平均整条 SOH 曲线 RMSE: {_safe_mean(pair_df['full_soh_rmse']):.4f}",
            ]
        )
    if not query_df.empty:
        lines.extend(
            [
                "",
                "## top-1 结果",
                f"- top-1 other-cell 命中率: {_safe_mean(query_df['top1_other_cell_hit']):.4f}",
                f"- top-1 `composite_distance` 均值: {_safe_mean(query_df['top1_composite_distance']):.6f}",
                f"- top-1 dQ-dV 诊断距离均值: {_safe_mean(query_df['top1_dqdv_distance']):.4f}",
                f"- top-1 3.2V+ dQ-dV 诊断距离均值: {_safe_mean(query_df['top1_dqdv_high_voltage_distance']):.4f}",
                f"- top-1 整条 SOH 曲线 RMSE 均值: {_safe_mean(query_df['top1_full_soh_rmse']):.4f}",
                f"- top-1 是否为 top-k 中最佳完整 SOH 匹配的比例: {_safe_mean(query_df['top1_is_best_full_soh']):.4f}",
            ]
        )
        hust_df = query_df.loc[query_df["query_source_dataset"].astype(str) == "hust_lfp"]
        if not hust_df.empty:
            lines.extend(
                [
                    "",
                    "## HUST / LFP 重点指标",
                    f"- HUST top-1 dQ-dV 诊断距离均值: {_safe_mean(hust_df['top1_dqdv_distance']):.4f}",
                    f"- HUST top-1 3.2V+ dQ-dV 诊断距离均值: {_safe_mean(hust_df['top1_dqdv_high_voltage_distance']):.4f}",
                    f"- HUST top-1 整条 SOH 曲线 RMSE 均值: {_safe_mean(hust_df['top1_full_soh_rmse']):.4f}",
                    f"- HUST top-1 是 top-k 中最佳完整 SOH 匹配的比例: {_safe_mean(hust_df['top1_is_best_full_soh']):.4f}",
                ]
            )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _run_single_setting(
    *,
    setting_name: str,
    output_dir: Path,
    case_bank_dir: Path,
    rows: pd.DataFrame,
    query_case_ids: Sequence[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """运行单个命名检索设置。"""

    setting_dir = output_dir / "settings" / setting_name
    figure_dir = setting_dir / "figures"
    setting_dir.mkdir(parents=True, exist_ok=True)

    rag_cfg_path = setting_dir / "retrieval_features.yaml"
    _make_retrieval_config(Path("configs/retrieval_features.yaml"), rag_cfg_path, top_k=args.top_k, setting_name=setting_name)

    db_case_count = int((rows["split"].astype(str) == "source_train").sum())
    retriever = MultiStageBatteryRetriever(
        case_bank_dir=case_bank_dir,
        retrieval_config_path=rag_cfg_path,
        db_splits=["source_train"],
        query_splits=["source_train"],
        top_m=max(db_case_count, args.top_k),
        top_k=args.top_k,
        same_cell_policy=str(args.same_cell_policy),
        exclude_query_case=bool(args.exclude_query_case),
        allow_cross_chemistry=True,
        stage1_embedding_name="handcrafted",
        retrieval_temperature=0.1,
        mmr={"use_mmr": True, "mmr_lambda": 0.75, "max_neighbors_per_cell": 1},
    )

    pair_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    for query_case_id in query_case_ids:
        preset = RETRIEVAL_SETTING_PRESETS[setting_name]
        if bool(preset.get("hust_high_voltage_dqdv", False)):
            result = _retrieve_with_hust_high_voltage_dqdv(
                retriever,
                rows,
                int(query_case_id),
                high_voltage_min_v=float(preset.get("hust_high_voltage_min_v", 3.2)),
                high_voltage_weight=float(preset.get("hust_high_voltage_weight", 0.45)),
            )
        else:
            result = retriever.retrieve(int(query_case_id))
        query_row = rows.iloc[retriever.case_id_to_index[int(query_case_id)]]
        query_idx = retriever.case_id_to_index[int(query_case_id)]
        query_curve = (
            _qv_map_high_voltage_dqdv_curve(retriever, query_idx, voltage_min=3.2)
            if bool(args.skip_raw_diagnostics)
            else _load_case_curve(query_row)[1]
        )
        neighbor_rows: list[pd.Series] = []
        local_pair_rows: list[dict[str, object]] = []
        topk_self_case_hit = False
        topk_same_cell_hit = False

        for rank in range(args.top_k):
            if int(result.retrieval_mask[rank]) <= 0:
                continue
            neighbor_case_id = int(result.neighbor_case_ids[rank])
            neighbor_row = rows.iloc[retriever.case_id_to_index[neighbor_case_id]]
            neighbor_rows.append(neighbor_row)
            same_case = int(neighbor_case_id) == int(query_case_id)
            same_cell = str(neighbor_row["cell_uid"]) == str(query_row["cell_uid"])
            topk_self_case_hit = topk_self_case_hit or same_case
            topk_same_cell_hit = topk_same_cell_hit or same_cell
            neighbor_idx = retriever.case_id_to_index[neighbor_case_id]
            neighbor_curve = (
                _qv_map_high_voltage_dqdv_curve(retriever, neighbor_idx, voltage_min=3.2)
                if bool(args.skip_raw_diagnostics)
                else _load_case_curve(neighbor_row)[1]
            )
            full_soh_rmse = float("nan") if bool(args.skip_raw_diagnostics) else _full_soh_rmse(query_row, neighbor_row)
            dqdv_distance = _dqdv_distance(query_curve, neighbor_curve)
            dqdv_high_voltage_distance = _dqdv_distance(
                query_curve,
                neighbor_curve,
                voltage_min=float(RETRIEVAL_SETTING_PRESETS[setting_name].get("hust_high_voltage_min_v", 3.2)),
            )

            pair_record = {
                "setting_name": setting_name,
                "query_case_id": int(query_case_id),
                "query_cell_uid": str(query_row["cell_uid"]),
                "query_source_dataset": str(query_row["source_dataset"]),
                "query_domain_label": str(query_row["domain_label"]),
                "neighbor_rank": int(rank + 1),
                "neighbor_case_id": neighbor_case_id,
                "neighbor_cell_uid": str(neighbor_row["cell_uid"]),
                "neighbor_source_dataset": str(neighbor_row["source_dataset"]),
                "neighbor_domain_label": str(neighbor_row["domain_label"]),
                "self_case_match": int(same_case),
                "same_cell_match": int(same_cell),
                "abs_anchor_soh_gap": abs(float(query_row["anchor_soh"]) - float(neighbor_row["anchor_soh"])),
                "abs_recent_soh_slope_gap": abs(float(query_row["recent_soh_slope"]) - float(neighbor_row["recent_soh_slope"])),
                "d_soh_state": float(result.d_soh_state[rank]),
                "d_qv_shape": float(result.d_qv_shape[rank]),
                "d_physics": float(result.d_physics[rank]),
                "d_operation": float(result.d_operation[rank]),
                "d_metadata": float(result.d_metadata[rank]),
                "composite_distance": float(result.composite_distance[rank]),
                "retrieval_confidence": float(result.retrieval_confidence),
                "dqdv_distance": dqdv_distance,
                "dqdv_high_voltage_distance": dqdv_high_voltage_distance,
                "full_soh_rmse": full_soh_rmse,
            }
            local_pair_rows.append(pair_record)
            pair_rows.append(pair_record)

        if neighbor_rows:
            top1_row = neighbor_rows[0]
            top1_other_cell_hit = int(str(top1_row["cell_uid"]) != str(query_row["cell_uid"]))
            topk_all_other_cells = int(all(str(row["cell_uid"]) != str(query_row["cell_uid"]) for row in neighbor_rows))
            local_pair_df = pd.DataFrame(local_pair_rows)
            valid_rmse = local_pair_df.loc[np.isfinite(local_pair_df["full_soh_rmse"]), ["neighbor_rank", "full_soh_rmse"]]
            best_rmse_rank = int(valid_rmse.sort_values("full_soh_rmse").iloc[0]["neighbor_rank"]) if not valid_rmse.empty else -1
            best_rmse_value = float(valid_rmse["full_soh_rmse"].min()) if not valid_rmse.empty else float("nan")
            top1_pair = local_pair_rows[0]
            query_rows.append(
                {
                    "setting_name": setting_name,
                    "query_case_id": int(query_case_id),
                    "query_cell_uid": str(query_row["cell_uid"]),
                    "query_source_dataset": str(query_row["source_dataset"]),
                    "query_domain_label": str(query_row["domain_label"]),
                    "topk": int(args.top_k),
                    "top1_case_id": int(top1_row["case_id"]),
                    "top1_cell_uid": str(top1_row["cell_uid"]),
                    "top1_source_dataset": str(top1_row["source_dataset"]),
                    "top1_other_cell_hit": top1_other_cell_hit,
                    "topk_all_other_cells": topk_all_other_cells,
                    "top1_self_case_hit": int(int(top1_row["case_id"]) == int(query_case_id)),
                    "topk_self_case_hit": int(topk_self_case_hit),
                    "top1_same_cell_hit": int(str(top1_row["cell_uid"]) == str(query_row["cell_uid"])),
                    "topk_same_cell_hit": int(topk_same_cell_hit),
                    "top1_d_soh_state": float(result.d_soh_state[0]),
                    "top1_d_qv_shape": float(result.d_qv_shape[0]),
                    "top1_d_physics": float(result.d_physics[0]),
                    "top1_d_metadata": float(result.d_metadata[0]),
                    "top1_composite_distance": float(result.composite_distance[0]),
                    "top1_dqdv_distance": float(top1_pair["dqdv_distance"]) if pd.notna(top1_pair["dqdv_distance"]) else float("nan"),
                    "top1_dqdv_high_voltage_distance": float(top1_pair["dqdv_high_voltage_distance"]) if pd.notna(top1_pair["dqdv_high_voltage_distance"]) else float("nan"),
                    "top1_full_soh_rmse": float(top1_pair["full_soh_rmse"]) if pd.notna(top1_pair["full_soh_rmse"]) else float("nan"),
                    "best_topk_full_soh_rank": int(best_rmse_rank),
                    "best_topk_full_soh_rmse": float(best_rmse_value) if not math.isnan(best_rmse_value) else float("nan"),
                    "top1_is_best_full_soh": int(best_rmse_rank == 1),
                    "topk_mean_d_soh_state": float(np.mean(result.d_soh_state[: len(neighbor_rows)])),
                    "topk_mean_d_qv_shape": float(np.mean(result.d_qv_shape[: len(neighbor_rows)])),
                    "topk_mean_d_physics": float(np.mean(result.d_physics[: len(neighbor_rows)])),
                    "topk_mean_d_metadata": float(np.mean(result.d_metadata[: len(neighbor_rows)])),
                    "topk_mean_composite_distance": float(np.mean(result.composite_distance[: len(neighbor_rows)])),
                    "topk_mean_dqdv_distance": _safe_mean(local_pair_df["dqdv_distance"]),
                    "topk_mean_dqdv_high_voltage_distance": _safe_mean(local_pair_df["dqdv_high_voltage_distance"]),
                    "topk_mean_full_soh_rmse": _safe_mean(local_pair_df["full_soh_rmse"]),
                    "retrieval_confidence": float(result.retrieval_confidence),
                }
            )
            query_figure_stem = f"query_case_{int(query_case_id):05d}_{str(query_row['cell_uid'])}"
            if not bool(args.skip_plots):
                _plot_soh_comparison(query_row, neighbor_rows, figure_dir / f"{query_figure_stem}_soh.png")
                _plot_dqdv_comparison(query_row, neighbor_rows, figure_dir / f"{query_figure_stem}_dqdv.png")

    pair_df = pd.DataFrame(pair_rows)
    query_df = pd.DataFrame(query_rows)
    pair_df.to_csv(setting_dir / "query_topk_similarity.csv", index=False)
    query_df.to_csv(setting_dir / "query_summary.csv", index=False)

    setting_info = {
        "setting_name": setting_name,
        "setting_description": RETRIEVAL_SETTING_PRESETS[setting_name]["description"],
        "retrieval_config_path": str(rag_cfg_path),
        "num_queries": int(len(query_df)),
        "num_query_pairs": int(len(pair_df)),
    }
    (setting_dir / "run_info.json").write_text(json.dumps(setting_info, indent=2, ensure_ascii=False), encoding="utf-8")
    _setting_summary_markdown(setting_dir, setting_name, rag_cfg_path, query_df, pair_df)
    return pair_df, query_df, setting_info


def _aggregate_setting_summary(query_df: pd.DataFrame, pair_df: pd.DataFrame) -> pd.DataFrame:
    """聚合每个设置的关键检索指标。"""

    rows = []
    for setting_name in sorted(query_df["setting_name"].astype(str).unique().tolist()):
        query_part = query_df.loc[query_df["setting_name"].astype(str) == setting_name].copy()
        pair_part = pair_df.loc[pair_df["setting_name"].astype(str) == setting_name].copy()
        hust_query = query_part.loc[query_part["query_source_dataset"].astype(str) == "hust_lfp"].copy()
        rows.append(
            {
                "setting_name": setting_name,
                "setting_description": RETRIEVAL_SETTING_PRESETS[setting_name]["description"],
                "top1_other_cell_hit_rate": _safe_mean(query_part["top1_other_cell_hit"]),
                "topk_all_other_cells_rate": _safe_mean(query_part["topk_all_other_cells"]),
                "top1_mean_composite_distance": _safe_mean(query_part["top1_composite_distance"]),
                "top1_mean_dqdv_distance": _safe_mean(query_part["top1_dqdv_distance"]),
                "top1_mean_dqdv_high_voltage_distance": _safe_mean(query_part["top1_dqdv_high_voltage_distance"]),
                "top1_mean_full_soh_rmse": _safe_mean(query_part["top1_full_soh_rmse"]),
                "top1_best_full_soh_rate": _safe_mean(query_part["top1_is_best_full_soh"]),
                "topk_mean_d_soh_state": _safe_mean(pair_part["d_soh_state"]),
                "topk_mean_d_qv_shape": _safe_mean(pair_part["d_qv_shape"]),
                "topk_mean_d_physics": _safe_mean(pair_part["d_physics"]),
                "topk_mean_d_metadata": _safe_mean(pair_part["d_metadata"]),
                "hust_top1_mean_dqdv_distance": _safe_mean(hust_query["top1_dqdv_distance"]),
                "hust_top1_mean_dqdv_high_voltage_distance": _safe_mean(hust_query["top1_dqdv_high_voltage_distance"]),
                "hust_top1_mean_full_soh_rmse": _safe_mean(hust_query["top1_full_soh_rmse"]),
                "hust_top1_best_full_soh_rate": _safe_mean(hust_query["top1_is_best_full_soh"]),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_setting_dataset_summary(query_df: pd.DataFrame) -> pd.DataFrame:
    """聚合每个设置在不同数据集上的 top-1 指标。"""

    rows = []
    for (setting_name, source_dataset), group in query_df.groupby(["setting_name", "query_source_dataset"], sort=True):
        rows.append(
            {
                "setting_name": str(setting_name),
                "query_source_dataset": str(source_dataset),
                "num_queries": int(len(group)),
                "top1_mean_composite_distance": _safe_mean(group["top1_composite_distance"]),
                "top1_mean_dqdv_distance": _safe_mean(group["top1_dqdv_distance"]),
                "top1_mean_full_soh_rmse": _safe_mean(group["top1_full_soh_rmse"]),
                "top1_best_full_soh_rate": _safe_mean(group["top1_is_best_full_soh"]),
            }
        )
    return pd.DataFrame(rows)


def _plot_setting_metric_bar(
    summary_df: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    """绘制每个检索设置的聚合柱状图。"""

    figure, axis = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    names = summary_df["setting_name"].astype(str).tolist()
    values = pd.to_numeric(summary_df[value_column], errors="coerce").fillna(np.nan).to_numpy(dtype=float)
    positions = np.arange(len(names), dtype=float)
    palette = ["#2563eb", "#16a34a", "#9333ea", "#f97316", "#0f766e", "#dc2626"]
    bars = axis.bar(positions, values, color=[palette[i % len(palette)] for i in range(len(names))])
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=15, ha="right")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            axis.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _plot_hust_query_comparison(query_df: pd.DataFrame, output_path: Path) -> None:
    """绘制 HUST query 在不同设置下的 top-1 完整 SOH RMSE 对照图。"""

    hust_df = query_df.loc[query_df["query_source_dataset"].astype(str) == "hust_lfp"].copy()
    if hust_df.empty:
        return
    pivot = hust_df.pivot_table(index="query_cell_uid", columns="setting_name", values="top1_full_soh_rmse", aggfunc="mean")
    pivot = pivot.sort_index()
    figure, axis = plt.subplots(figsize=(9.0, 4.8), dpi=180)
    x = np.arange(len(pivot.index), dtype=float)
    palette = ["#2563eb", "#16a34a", "#9333ea", "#f97316", "#0f766e", "#dc2626"]
    for idx, setting_name in enumerate(pivot.columns.tolist()):
        color = palette[idx % len(palette)]
        axis.plot(x, pivot[setting_name].to_numpy(dtype=float), marker="o", linewidth=2.0, color=color, label=setting_name)
    axis.set_xticks(x)
    axis.set_xticklabels(pivot.index.tolist(), rotation=25, ha="right")
    axis.set_ylabel("Top-1 full-life SOH RMSE")
    axis.set_title("HUST LFP top-1 full-life SOH similarity by retrieval setting")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def _write_comparison_summary(
    output_dir: Path,
    *,
    run_info: dict[str, object],
    setting_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> None:
    """写入所有设置的总体对照总结。"""

    lines = [
        "# 子集 RAG 检索并排对照总结",
        "",
        "## 实验设置",
        f"- case bank 路径: `{run_info['case_bank_dir']}`",
        f"- reference cells: HUST(LFP)={run_info['num_hust_reference_cells']}，TJU Dataset_1(NCA)={run_info['num_nca_reference_cells']}，TJU Dataset_2(NCM)={run_info['num_ncm_reference_cells']}",
        f"- query cells: 从 source_train 随机抽取 {run_info['num_query_cells']} 个 cell 的最后一个窗口",
        f"- same-cell policy: `{run_info['same_cell_policy']}`",
        f"- exclude query case: `{run_info['exclude_query_case']}`",
        f"- 检索设置: {', '.join(run_info['setting_names'])}",
        "",
        "## 各设置说明",
    ]
    for setting_name in run_info["setting_names"]:
        lines.append(f"- `{setting_name}`: {RETRIEVAL_SETTING_PRESETS[str(setting_name)]['description']}")

    if not setting_summary.empty:
        best_hust = setting_summary.sort_values("hust_top1_mean_full_soh_rmse").iloc[0]
        best_global = setting_summary.sort_values("top1_mean_full_soh_rmse").iloc[0]
        lines.extend(
            [
                "",
                "## 关键对照结果",
                f"- 全部 query 上 top-1 完整 SOH RMSE 最低的设置: `{best_global['setting_name']}` ({float(best_global['top1_mean_full_soh_rmse']):.4f})",
                f"- HUST / LFP 上 top-1 完整 SOH RMSE 最低的设置: `{best_hust['setting_name']}` ({float(best_hust['hust_top1_mean_full_soh_rmse']):.4f})",
                f"- HUST / LFP 上 top-1 属于 top-k 中最佳完整 SOH 匹配的最高比例: {float(setting_summary['hust_top1_best_full_soh_rate'].max()):.4f}",
                "",
                "## HUST / LFP 解读",
                "- 如果 `state_metadata` 明显劣于含 `d_qv_shape` 或 `d_physics` 的设置，说明原始异常主要来自排序特征不足，而不是绘图问题。",
                "- 如果加入 `d_qv_shape` 后 HUST 的 top-1 dQ-dV 距离和完整 SOH RMSE 同时下降，说明 LFP 场景需要曲线形态信息辅助排序。",
                "- 如果再加入 `d_physics` 后继续改善，说明 DeltaV/R 与 partial charge 的低维退化 proxy 进一步帮助区分局部状态相近的电池。",
                "- 当前 `d_operation` 默认关闭；充放电电流、温度和归一化容量变化已经合并到 `d_metadata` 的原始数值匹配中。",
            ]
        )

    if not dataset_summary.empty:
        lines.extend(
            [
                "",
                "## 分数据集建议",
            ]
        )
        for _, row in dataset_summary.sort_values(["query_source_dataset", "setting_name"]).iterrows():
            lines.append(
                f"- `{row['query_source_dataset']}` | `{row['setting_name']}`: "
                f"top-1 full-life SOH RMSE={float(row['top1_mean_full_soh_rmse']):.4f}, "
                f"top-1 dQ-dV distance={float(row['top1_mean_dqdv_distance']):.4f}, "
                f"top-1 3.2V+ dQ-dV distance={float(row.get('top1_mean_dqdv_high_voltage_distance', float('nan'))):.4f}, "
                f"top-1 best-full-SOH rate={float(row['top1_best_full_soh_rate']):.4f}"
            )
    (output_dir / "comparison_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice.", category=RuntimeWarning)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    setting_names = _parse_setting_names(args.retrieval_settings)

    cfg = load_config(args.config)
    case_bank_dir = output_dir / "case_bank"
    manifest_path = output_dir / "split_manifest.csv"
    if bool(args.reuse_case_bank) and (case_bank_dir / "case_rows.parquet").exists() and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        subset_meta = {
            "num_hust_reference_cells": int((manifest["source_dataset"].astype(str) == "hust_lfp").sum()),
            "num_nca_reference_cells": int((manifest["source_dataset"].astype(str) == "tju_dataset_1_nca").sum()),
            "num_ncm_reference_cells": int((manifest["source_dataset"].astype(str) == "tju_dataset_2_ncm").sum()),
            "db_cell_uids": sorted(manifest["cell_uid"].astype(str).tolist()),
            "query_selection_policy": "random source_train cells, latest window per selected cell",
            "case_bank_reused": True,
        }
    else:
        subset_cfg = _make_subset_cfg(cfg, output_dir)
        all_cells, subset_meta = _prepare_reference_cells(
            cfg,
            max_hust_cells=args.max_hust_cells,
            max_nca_cells=args.max_nca_cells,
            max_ncm_cells=args.max_ncm_cells,
        )

        manifest = _build_manifest(all_cells)
        manifest.to_csv(manifest_path, index=False)

        subset_cfg_path = output_dir / "subset_case_bank_config.yaml"
        subset_cfg_path.write_text(yaml.safe_dump(subset_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

        case_bank_result = build_case_bank_from_cells(subset_cfg, all_cells, split_manifest=manifest)
        case_bank_dir = Path(case_bank_result["output_dir"])
    rows = _case_rows(case_bank_dir)
    query_case_ids = _select_query_case_ids(rows, num_query_cells=args.num_query_cells, random_seed=int(cfg.get("random_seed", 7)))

    pair_frames = []
    query_frames = []
    setting_infos = []
    for setting_name in setting_names:
        pair_df, query_df, setting_info = _run_single_setting(
            setting_name=setting_name,
            output_dir=output_dir,
            case_bank_dir=case_bank_dir,
            rows=rows,
            query_case_ids=query_case_ids,
            args=args,
        )
        pair_frames.append(pair_df)
        query_frames.append(query_df)
        setting_infos.append(setting_info)

    comparison_pair_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    comparison_query_df = pd.concat(query_frames, ignore_index=True) if query_frames else pd.DataFrame()
    comparison_pair_df.to_csv(output_dir / "comparison_pair_summary.csv", index=False)
    comparison_query_df.to_csv(output_dir / "comparison_query_summary.csv", index=False)

    setting_summary = _aggregate_setting_summary(comparison_query_df, comparison_pair_df) if not comparison_query_df.empty else pd.DataFrame()
    dataset_summary = _aggregate_setting_dataset_summary(comparison_query_df) if not comparison_query_df.empty else pd.DataFrame()
    setting_summary.to_csv(output_dir / "comparison_setting_summary.csv", index=False)
    dataset_summary.to_csv(output_dir / "comparison_setting_summary_by_dataset.csv", index=False)

    comparison_figure_dir = output_dir / "figures" / "comparison"
    if not setting_summary.empty:
        _plot_setting_metric_bar(
            setting_summary,
            value_column="top1_mean_full_soh_rmse",
            title="Top-1 full-life SOH RMSE by retrieval setting",
            ylabel="Mean top-1 full-life SOH RMSE",
            output_path=comparison_figure_dir / "top1_full_soh_rmse_by_setting.png",
        )
        _plot_setting_metric_bar(
            setting_summary,
            value_column="top1_mean_dqdv_distance",
            title="Top-1 dQ-dV distance by retrieval setting",
            ylabel="Mean top-1 dQ-dV distance",
            output_path=comparison_figure_dir / "top1_dqdv_distance_by_setting.png",
        )
        _plot_setting_metric_bar(
            setting_summary,
            value_column="top1_mean_dqdv_high_voltage_distance",
            title="Top-1 3.2V+ dQ-dV distance by retrieval setting",
            ylabel="Mean top-1 3.2V+ dQ-dV distance",
            output_path=comparison_figure_dir / "top1_high_voltage_dqdv_distance_by_setting.png",
        )
        _plot_setting_metric_bar(
            setting_summary,
            value_column="hust_top1_best_full_soh_rate",
            title="HUST LFP: top-1 best-full-SOH rate by retrieval setting",
            ylabel="Rate",
            output_path=comparison_figure_dir / "hust_top1_best_full_soh_rate.png",
        )
    if not comparison_query_df.empty:
        _plot_hust_query_comparison(comparison_query_df, comparison_figure_dir / "hust_top1_full_soh_rmse_by_query.png")

    run_info = {
        **subset_meta,
        "case_bank_dir": str(case_bank_dir),
        "num_query_cells": int(len(query_case_ids)),
        "same_cell_policy": str(args.same_cell_policy),
        "exclude_query_case": bool(args.exclude_query_case),
        "setting_names": setting_names,
        "setting_infos": setting_infos,
    }
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_comparison_summary(output_dir, run_info=run_info, setting_summary=setting_summary, dataset_summary=dataset_summary)
    return run_info


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run subset battery cross-cell retrieval comparison on source-train cells.")
    parser.add_argument("--config", type=str, default="configs/battery_soh.yaml", help="Base project config.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory for outputs.")
    parser.add_argument("--max-hust-cells", type=int, default=12, help="Number of HUST reference cells to include.")
    parser.add_argument("--max-nca-cells", type=int, default=12, help="Number of TJU Dataset_1 NCA reference cells to include.")
    parser.add_argument("--max-ncm-cells", type=int, default=12, help="Number of TJU Dataset_2 NCM reference cells to include.")
    parser.add_argument("--num-query-cells", type=int, default=6, help="Number of random source-train cells to use as query.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved reference cells to keep.")
    parser.add_argument("--same-cell-policy", type=str, choices=["exclude", "past_only", "allow"], default="exclude", help="Whether retrieval can return windows from the same cell.")
    parser.add_argument("--exclude-query-case", action=argparse.BooleanOptionalAction, default=True, help="Whether retrieval should explicitly remove the exact query case from candidate references.")
    parser.add_argument("--reuse-case-bank", action=argparse.BooleanOptionalAction, default=False, help="Reuse an existing case_bank/ under output-dir instead of rebuilding raw preprocessing.")
    parser.add_argument("--skip-raw-diagnostics", action=argparse.BooleanOptionalAction, default=False, help="Avoid raw file dQ/dV and full-life SOH diagnostics; use case-bank QV maps for fast high-voltage dQ/dV checks.")
    parser.add_argument("--skip-plots", action=argparse.BooleanOptionalAction, default=False, help="Skip per-query matplotlib plots for fast diagnostic reruns.")
    parser.add_argument(
        "--retrieval-settings",
        type=str,
        default="state_metadata,state_metadata_qv,state_metadata_qv_physics,state_metadata_qv_physics_hust_hv_dqdv",
        help="Comma-separated retrieval setting names to compare.",
    )
    args = parser.parse_args(argv)
    result = run_experiment(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
