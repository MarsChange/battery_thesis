from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retrieval.multistage_retriever import COMPONENT_NAMES, MultiStageBatteryRetriever, RetrievalResult


def rag_only_prediction(result: RetrievalResult) -> np.ndarray:
    if result.neighbor_future_delta_soh.size == 0:
        return np.zeros(0, dtype=np.float32)
    weights = np.asarray(result.retrieval_alpha, dtype=np.float32)[:, None]
    return np.sum(result.neighbor_future_delta_soh * weights, axis=0).astype(np.float32)


def nearest1_prediction(result: RetrievalResult) -> np.ndarray:
    valid = np.flatnonzero(result.retrieval_mask > 0)
    if valid.size == 0:
        return np.zeros(result.neighbor_future_delta_soh.shape[-1], dtype=np.float32)
    return np.asarray(result.neighbor_future_delta_soh[int(valid[0])], dtype=np.float32)


def summarize_retrieval_result(query_row: pd.Series, result: RetrievalResult, retriever: MultiStageBatteryRetriever) -> Dict[str, object]:
    valid = np.flatnonzero(result.retrieval_mask > 0)
    chemistry_match = []
    domain_match = []
    stage_match = []
    same_cell = []
    for idx in valid.tolist():
        ref_case_id = int(result.neighbor_case_ids[idx])
        ref_row = retriever.case_rows.iloc[retriever.case_id_to_index[ref_case_id]]
        chemistry_match.append(float(str(ref_row["chemistry_family"]) == str(query_row["chemistry_family"])))
        domain_match.append(float(str(ref_row["domain_label"]) == str(query_row["domain_label"])))
        stage_match.append(float(str(ref_row["degradation_stage"]) == str(query_row["degradation_stage"])))
        same_cell.append(float(str(ref_row["cell_uid"]) == str(query_row["cell_uid"])))
    component_mean = result.component_distances[valid].mean(axis=0) if len(valid) else np.zeros(len(COMPONENT_NAMES), dtype=np.float32)
    return {
        "query_case_id": int(query_row["case_id"]),
        "split": str(query_row["split"]),
        "chemistry_family": str(query_row["chemistry_family"]),
        "domain_label": str(query_row["domain_label"]),
        "topk_mean_composite_distance": float(result.composite_distances[valid].mean()) if len(valid) else np.nan,
        "topk_mean_tsfm_distance": float(component_mean[0]),
        "topk_mean_soh_distance": float(component_mean[1]),
        "topk_mean_qv_distance": float(component_mean[2]),
        "topk_mean_physics_distance": float(component_mean[3]),
        "topk_mean_operation_distance": float(component_mean[4]),
        "topk_mean_metadata_distance": float(component_mean[5]),
        "topk_mean_stage_distance": float(component_mean[6]),
        "topk_mean_missing_distance": float(component_mean[7]),
        "topk_chemistry_match_rate": float(np.mean(chemistry_match)) if chemistry_match else 0.0,
        "topk_domain_match_rate": float(np.mean(domain_match)) if domain_match else 0.0,
        "topk_degradation_stage_match_rate": float(np.mean(stage_match)) if stage_match else 0.0,
        "topk_same_cell_rate": float(np.mean(same_cell)) if same_cell else 0.0,
        "retrieval_confidence": float(result.retrieval_confidence),
    }


def save_component_distance_bar(result: RetrievalResult, save_path: str | Path) -> None:
    valid = np.flatnonzero(result.retrieval_mask > 0)
    figure, axis = plt.subplots(figsize=(10, 4.5), dpi=180)
    if len(valid) == 0:
        axis.text(0.5, 0.5, "No valid neighbors", ha="center", va="center", transform=axis.transAxes)
    else:
        width = 0.08
        x = np.arange(len(valid))
        for comp_idx, comp_name in enumerate(COMPONENT_NAMES):
            axis.bar(x + comp_idx * width, result.component_distances[valid, comp_idx], width=width, label=comp_name)
        axis.set_xticks(x + width * len(COMPONENT_NAMES) / 2)
        axis.set_xticklabels([f"n{i+1}" for i in range(len(valid))])
    axis.set_title("Retrieval component distances")
    axis.set_ylabel("Distance")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(ncols=4, fontsize=8)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_metadata_table(query_row: pd.Series, result: RetrievalResult, retriever: MultiStageBatteryRetriever, save_path: str | Path) -> pd.DataFrame:
    records = []
    valid = np.flatnonzero(result.retrieval_mask > 0)
    records.append(
        {
            "role": "query",
            "case_id": int(query_row["case_id"]),
            "cell_uid": query_row["cell_uid"],
            "chemistry_family": query_row["chemistry_family"],
            "domain_label": query_row["domain_label"],
            "degradation_stage": query_row["degradation_stage"],
            "anchor_soh": float(query_row["anchor_soh"]),
            "recent_soh_slope": float(query_row["recent_soh_slope"]),
            "composite_distance": 0.0,
        }
    )
    for idx in valid.tolist():
        ref_case_id = int(result.neighbor_case_ids[idx])
        ref_row = retriever.case_rows.iloc[retriever.case_id_to_index[ref_case_id]]
        records.append(
            {
                "role": f"neighbor_{idx+1}",
                "case_id": int(ref_case_id),
                "cell_uid": ref_row["cell_uid"],
                "chemistry_family": ref_row["chemistry_family"],
                "domain_label": ref_row["domain_label"],
                "degradation_stage": ref_row["degradation_stage"],
                "anchor_soh": float(ref_row["anchor_soh"]),
                "recent_soh_slope": float(ref_row["recent_soh_slope"]),
                "composite_distance": float(result.composite_distances[idx]),
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
    figure, axis = plt.subplots(figsize=(9, 5), dpi=180)
    anchor_soh = float(query_row["anchor_soh"])
    history = retriever.arrays["soh_seq"][query_idx]
    axis.plot(np.arange(len(history)), history, label="query_history", linewidth=2.0, color="black")
    true_future = retriever.arrays["future_soh"][query_idx]
    true_series = np.concatenate([[anchor_soh], true_future])
    true_x = np.arange(len(history) - 1, len(history) - 1 + len(true_series))
    axis.plot(true_x, true_series, label="query_future_true", linewidth=2.0, color="tab:red")

    valid = np.flatnonzero(result.retrieval_mask > 0)
    for rank, idx in enumerate(valid.tolist(), start=1):
        ref_case_id = int(result.neighbor_case_ids[idx])
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
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    query_qv = retriever.arrays["qv_maps"][query_idx, -1]
    q_grid = np.linspace(0.0, 1.0, query_qv.shape[-1], dtype=np.float32)
    axes[0].plot(q_grid, query_qv[4], label="query DeltaV", linewidth=2.0, color="black")
    axes[1].plot(q_grid, query_qv[5], label="query R", linewidth=2.0, color="black")
    valid = np.flatnonzero(result.retrieval_mask > 0)
    for rank, idx in enumerate(valid.tolist(), start=1):
        ref_case_id = int(result.neighbor_case_ids[idx])
        ref_idx = retriever.case_id_to_index[ref_case_id]
        ref_qv = retriever.arrays["qv_maps"][ref_idx, -1]
        axes[0].plot(q_grid, ref_qv[4], alpha=0.4, linewidth=1.0, label=f"ref_{rank}" if rank <= 3 else None)
        axes[1].plot(q_grid, ref_qv[5], alpha=0.4, linewidth=1.0, label=f"ref_{rank}" if rank <= 3 else None)
    axes[0].set_title("DeltaV(Q) overlay")
    axes[1].set_title("R(Q) overlay")
    for axis in axes:
        axis.set_xlabel("Normalized capacity Q")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_partial_relax_overlay(
    query_idx: int,
    result: RetrievalResult,
    retriever: MultiStageBatteryRetriever,
    save_path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=180)
    query_pc = retriever.arrays["partial_charge"][query_idx, -1]
    query_relax = retriever.arrays["relaxation"][query_idx, -1]
    axes[0].plot(query_pc, label="query", linewidth=2.0, color="black")
    axes[1].plot(query_relax, label="query", linewidth=2.0, color="black")
    valid = np.flatnonzero(result.retrieval_mask > 0)
    for rank, idx in enumerate(valid.tolist(), start=1):
        ref_case_id = int(result.neighbor_case_ids[idx])
        ref_idx = retriever.case_id_to_index[ref_case_id]
        axes[0].plot(retriever.arrays["partial_charge"][ref_idx, -1], alpha=0.4, linewidth=1.0)
        axes[1].plot(retriever.arrays["relaxation"][ref_idx, -1], alpha=0.4, linewidth=1.0)
    axes[0].set_title("Partial charge overlay")
    axes[1].set_title("Relaxation overlay")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
