"""retrieval.retrieval_diagnostics

提供检索诊断阶段的通用函数：
- RAG-only / nearest1 / persistence 预测；
- 命名距离分量的摘要；
- 每个 query 的可视化与 metadata 表导出。
所有输出均使用明确的距离名称，例如 `d_soh_state`、`d_qv_shape`。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retrieval.multistage_retriever import COMPONENT_NAMES, CORE_COMPONENTS, MultiStageBatteryRetriever, RetrievalResult


def rag_only_prediction(result: RetrievalResult) -> np.ndarray:
    """根据 top-k 检索案例的 `future_delta_soh` 加权平均得到 RAG-only 预测。"""

    valid = np.flatnonzero(result.retrieval_mask > 0)
    if valid.size == 0:
        return np.zeros(result.neighbor_future_delta_soh.shape[-1], dtype=np.float32)
    weights = np.asarray(result.retrieval_alpha[valid], dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-6)
    return np.sum(result.neighbor_future_delta_soh[valid] * weights[:, None], axis=0).astype(np.float32)


def nearest1_prediction(result: RetrievalResult) -> np.ndarray:
    """返回最近邻 reference 的未来 `delta_soh` 作为预测。"""

    valid = np.flatnonzero(result.retrieval_mask > 0)
    if valid.size == 0:
        return np.zeros(result.neighbor_future_delta_soh.shape[-1], dtype=np.float32)
    return np.asarray(result.neighbor_future_delta_soh[int(valid[0])], dtype=np.float32)


def persistence_delta_prediction(horizon: int) -> np.ndarray:
    """持久性基线：对 delta SOH 任务默认预测全零。"""

    return np.zeros(int(horizon), dtype=np.float32)


def summarize_retrieval_result(query_row: pd.Series, result: RetrievalResult, retriever: MultiStageBatteryRetriever) -> Dict[str, object]:
    """把单个 query 的 top-k 检索结果摘要成一行 CSV 记录。"""

    valid = np.flatnonzero(result.retrieval_mask > 0)
    component_mean = result.component_distances[valid].mean(axis=0) if valid.size else np.zeros(len(COMPONENT_NAMES), dtype=np.float32)
    component_std = result.component_distances[valid].std(axis=0) if valid.size else np.zeros(len(COMPONENT_NAMES), dtype=np.float32)
    chemistry_match = []
    domain_match = []
    for pos in valid.tolist():
        ref_case_id = int(result.neighbor_case_ids[pos])
        ref_row = retriever.case_rows.iloc[retriever.case_id_to_index[ref_case_id]]
        chemistry_match.append(float(str(ref_row["chemistry_family"]) == str(query_row["chemistry_family"])))
        domain_match.append(float(str(ref_row["domain_label"]) == str(query_row["domain_label"])))
    summary = {
        "query_case_id": int(query_row["case_id"]),
        "split": str(query_row["split"]),
        "chemistry_family": str(query_row["chemistry_family"]),
        "domain_label": str(query_row["domain_label"]),
        "topk_mean_d_soh_state": float(component_mean[0]),
        "topk_mean_d_qv_shape": float(component_mean[1]),
        "topk_mean_d_physics": float(component_mean[2]),
        "topk_mean_d_operation": float(component_mean[3]),
        "topk_mean_d_metadata": float(component_mean[4]),
        "topk_std_d_soh_state": float(component_std[0]),
        "topk_std_d_qv_shape": float(component_std[1]),
        "topk_std_d_physics": float(component_std[2]),
        "topk_std_d_operation": float(component_std[3]),
        "topk_std_d_metadata": float(component_std[4]),
        "topk_mean_composite_distance": float(result.composite_distance[valid].mean()) if valid.size else np.nan,
        "topk_std_composite_distance": float(result.composite_distance[valid].std()) if valid.size else np.nan,
        "retrieval_confidence": float(result.retrieval_confidence),
        "chemistry_match_rate": float(np.mean(chemistry_match)) if chemistry_match else 0.0,
        "domain_match_rate": float(np.mean(domain_match)) if domain_match else 0.0,
    }
    return summary


def save_component_distance_bar(result: RetrievalResult, save_path: str | Path) -> None:
    """保存每个 top-k neighbor 的命名距离柱状图。"""

    valid = np.flatnonzero(result.retrieval_mask > 0)
    figure, axis = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    if valid.size == 0:
        axis.text(0.5, 0.5, "No valid neighbors", ha="center", va="center", transform=axis.transAxes)
    else:
        active_component_names = list(CORE_COMPONENTS)
        width = 0.11
        x = np.arange(len(valid))
        for comp_plot_idx, comp_name in enumerate(active_component_names):
            comp_idx = COMPONENT_NAMES.index(comp_name)
            axis.bar(x + comp_plot_idx * width, result.component_distances[valid, comp_idx], width=width, label=comp_name)
        axis.set_xticks(x + width * (len(active_component_names) - 1) / 2)
        axis.set_xticklabels([f"neighbor_{rank}" for rank in range(1, len(valid) + 1)])
    axis.set_title("Top-k retrieval component distances")
    axis.set_ylabel("Distance (lower is better)")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(ncols=3, fontsize=8)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_metadata_table(query_row: pd.Series, result: RetrievalResult, retriever: MultiStageBatteryRetriever, save_path: str | Path) -> pd.DataFrame:
    """导出 query 与 top-k reference 的元信息和命名距离表。"""

    records = [
        {
            "role": "query",
            "case_id": int(query_row["case_id"]),
            "cell_uid": query_row["cell_uid"],
            "chemistry_family": query_row["chemistry_family"],
            "domain_label": query_row["domain_label"],
            "degradation_stage": query_row["degradation_stage"],
            "anchor_soh": float(query_row["anchor_soh"]),
            "recent_soh_slope": float(query_row["recent_soh_slope"]),
            "d_soh_state": 0.0,
            "d_qv_shape": 0.0,
            "d_physics": 0.0,
            "d_operation": 0.0,
            "d_metadata": 0.0,
            "composite_distance": 0.0,
        }
    ]
    valid = np.flatnonzero(result.retrieval_mask > 0)
    for pos in valid.tolist():
        ref_case_id = int(result.neighbor_case_ids[pos])
        ref_row = retriever.case_rows.iloc[retriever.case_id_to_index[ref_case_id]]
        records.append(
            {
                "role": f"neighbor_{pos + 1}",
                "case_id": int(ref_case_id),
                "cell_uid": ref_row["cell_uid"],
                "chemistry_family": ref_row["chemistry_family"],
                "domain_label": ref_row["domain_label"],
                "degradation_stage": ref_row["degradation_stage"],
                "anchor_soh": float(ref_row["anchor_soh"]),
                "recent_soh_slope": float(ref_row["recent_soh_slope"]),
                "d_soh_state": float(result.d_soh_state[pos]),
                "d_qv_shape": float(result.d_qv_shape[pos]),
                "d_physics": float(result.d_physics[pos]),
                "d_operation": float(result.d_operation[pos]),
                "d_metadata": float(result.d_metadata[pos]),
                "composite_distance": float(result.composite_distance[pos]),
            }
        )
    frame = pd.DataFrame(records)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(save_path, index=False)
    return frame


def save_soh_history_figure(
    query_idx: int,
    query_row: pd.Series,
    result: RetrievalResult,
    retriever: MultiStageBatteryRetriever,
    save_path: str | Path,
) -> None:
    """保存 query 历史 SOH、true future 和 top-k future 的对齐图。"""

    figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
    anchor_soh = float(query_row["anchor_soh"])
    history = retriever.arrays["soh_seq"][query_idx]
    axis.plot(np.arange(len(history)), history, label="query_history", linewidth=2.0, color="black")
    true_future = retriever.arrays["future_soh"][query_idx]
    true_series = np.concatenate([[anchor_soh], true_future])
    true_x = np.arange(len(history) - 1, len(history) - 1 + len(true_series))
    axis.plot(true_x, true_series, label="query_future_true", linewidth=2.0, color="tab:red")

    valid = np.flatnonzero(result.retrieval_mask > 0)
    for rank, pos in enumerate(valid.tolist(), start=1):
        ref_case_id = int(result.neighbor_case_ids[pos])
        ref_idx = retriever.case_id_to_index[ref_case_id]
        ref_hist = retriever.arrays["soh_seq"][ref_idx]
        ref_anchor = float(retriever.case_rows.iloc[ref_idx]["anchor_soh"])
        aligned_hist = ref_hist - ref_anchor + anchor_soh
        axis.plot(np.arange(len(aligned_hist)), aligned_hist, alpha=0.4, linewidth=1.0, label=f"ref_{rank}_history")
        ref_future = retriever.arrays["future_delta"][ref_idx] + anchor_soh
        axis.plot(np.arange(len(history), len(history) + len(ref_future)), ref_future, alpha=0.35, linewidth=1.0)

    rag_pred = rag_only_prediction(result) + anchor_soh
    axis.plot(np.arange(len(history), len(history) + len(rag_pred)), rag_pred, label="rag_weighted_pred", linewidth=2.0, color="tab:blue")
    axis.set_title(f"SOH history and retrieved futures | case {query_row['case_id']}")
    axis.set_xlabel("Relative step")
    axis.set_ylabel("SOH")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_qv_overlay(
    query_idx: int,
    result: RetrievalResult,
    retriever: MultiStageBatteryRetriever,
    save_path: str | Path,
) -> None:
    """保存 query 与 top-k reference 的 DeltaV(Q) / R(Q) 叠加图。"""

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    query_qv = retriever.arrays["qv_maps"][query_idx, -1]
    query_mask = retriever.arrays["qv_masks"][query_idx, -1]
    q_grid = np.linspace(0.0, 1.0, query_qv.shape[-1], dtype=np.float32)
    if query_mask[4] > 0:
        axes[0].plot(q_grid, query_qv[4], label="query DeltaV(Q)", linewidth=2.0, color="black")
    else:
        axes[0].text(0.5, 0.5, "query DeltaV unavailable", ha="center", va="center", transform=axes[0].transAxes)
    if query_mask[5] > 0:
        axes[1].plot(q_grid, query_qv[5], label="query R(Q)", linewidth=2.0, color="black")
    else:
        axes[1].text(0.5, 0.5, "query R(Q) unavailable", ha="center", va="center", transform=axes[1].transAxes)
    valid = np.flatnonzero(result.retrieval_mask > 0)
    for rank, pos in enumerate(valid.tolist(), start=1):
        ref_case_id = int(result.neighbor_case_ids[pos])
        ref_idx = retriever.case_id_to_index[ref_case_id]
        ref_qv = retriever.arrays["qv_maps"][ref_idx, -1]
        ref_mask = retriever.arrays["qv_masks"][ref_idx, -1]
        if ref_mask[4] > 0:
            axes[0].plot(q_grid, ref_qv[4], alpha=0.4, linewidth=1.0, label=f"ref_{rank}" if rank <= 3 else None)
        if ref_mask[5] > 0:
            axes[1].plot(q_grid, ref_qv[5], alpha=0.4, linewidth=1.0, label=f"ref_{rank}" if rank <= 3 else None)
    axes[0].set_title("DeltaV(Q) overlay")
    axes[1].set_title("R(Q) overlay")
    for axis in axes:
        axis.set_xlabel("Normalized capacity Q")
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if labels:
            axis.legend(fontsize=8)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_partial_charge_overlay(
    query_idx: int,
    result: RetrievalResult,
    retriever: MultiStageBatteryRetriever,
    save_path: str | Path,
) -> None:
    """保存 query 与 top-k reference 的 partial charge 叠加图。"""

    figure, axis = plt.subplots(figsize=(6.5, 4.5), dpi=180)
    query_pc = retriever.arrays["partial_charge"][query_idx, -1]
    query_pc_mask = retriever.arrays["partial_charge_mask"][query_idx, -1]
    if query_pc_mask > 0:
        axis.plot(query_pc, label="query", linewidth=2.0, color="black")
    else:
        axis.text(0.5, 0.5, "query partial charge unavailable", ha="center", va="center", transform=axis.transAxes)
    valid = np.flatnonzero(result.retrieval_mask > 0)
    for _, pos in enumerate(valid.tolist(), start=1):
        ref_case_id = int(result.neighbor_case_ids[pos])
        ref_idx = retriever.case_id_to_index[ref_case_id]
        ref_pc_mask = retriever.arrays["partial_charge_mask"][ref_idx, -1]
        if ref_pc_mask > 0:
            axis.plot(retriever.arrays["partial_charge"][ref_idx, -1], alpha=0.4, linewidth=1.0)
    axis.set_title("Partial charge overlay")
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(fontsize=8)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
