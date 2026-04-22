from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json


def _ensure_path(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_training_curves(history: pd.DataFrame, figure_dir: str | Path) -> None:
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    x = history["epoch"].to_numpy()

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    axis.plot(x, history["train_loss"], label="train_loss")
    axis.plot(x, history["val_loss"], label="val_loss")
    axis.set_title("Training loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "loss_curve.png", bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    axis.plot(x, history["val_mae"], label="val_mae")
    axis.plot(x, history["val_rmse"], label="val_rmse")
    axis.plot(x, history["val_mape"], label="val_mape")
    axis.set_title("Validation metrics")
    axis.set_xlabel("Epoch")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "val_metrics_curve.png", bbox_inches="tight")
    plt.close(figure)

    if "expert_weight_mean_json" in history.columns:
        expert_matrix = np.stack(history["expert_weight_mean_json"].apply(lambda s: np.asarray(json.loads(s), dtype=np.float32)).tolist())
        figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
        for idx in range(expert_matrix.shape[1]):
            axis.plot(x, expert_matrix[:, idx], label=f"expert_{idx+1}")
        axis.set_title("Expert weight evolution")
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8, ncols=2)
        figure.tight_layout()
        figure.savefig(figure_dir / "expert_weight_evolution.png", bbox_inches="tight")
        plt.close(figure)

    if "fusion_weight_mean_json" in history.columns:
        fusion_matrix = np.stack(history["fusion_weight_mean_json"].apply(lambda s: np.asarray(json.loads(s), dtype=np.float32)).tolist())
        figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
        labels = ["fm", "rag", "pair", "moe"]
        for idx in range(fusion_matrix.shape[1]):
            axis.plot(x, fusion_matrix[:, idx], label=labels[idx] if idx < len(labels) else f"branch_{idx+1}")
        axis.set_title("Fusion weight evolution")
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figure_dir / "fusion_weight_evolution.png", bbox_inches="tight")
        plt.close(figure)

    if "horizon_mae_json" in history.columns:
        horizon_matrix = np.stack(history["horizon_mae_json"].apply(lambda s: np.asarray(json.loads(s), dtype=np.float32)).tolist())
        figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
        for idx in range(horizon_matrix.shape[1]):
            axis.plot(x, horizon_matrix[:, idx], alpha=0.7)
        axis.set_title("Horizon-wise validation MAE")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MAE")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(figure_dir / "horizon_wise_val_mae.png", bbox_inches="tight")
        plt.close(figure)


def plot_horizon_error(values: np.ndarray, title: str, save_path: str | Path) -> None:
    values = np.asarray(values, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    axis.plot(np.arange(1, len(values) + 1), values, marker="o")
    axis.set_title(title)
    axis.set_xlabel("Horizon")
    axis.set_ylabel("Error")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(_ensure_path(save_path), bbox_inches="tight")
    plt.close(figure)


def plot_group_bar(frame: pd.DataFrame, value_col: str, category_col: str, title: str, save_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    axis.bar(np.arange(len(frame)), frame[value_col].to_numpy(dtype=np.float32), color="#2563eb")
    axis.set_xticks(np.arange(len(frame)))
    axis.set_xticklabels(frame[category_col].astype(str).tolist(), rotation=30, ha="right")
    axis.set_title(title)
    axis.set_ylabel(value_col)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(_ensure_path(save_path), bbox_inches="tight")
    plt.close(figure)


def plot_weight_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, save_path: str | Path) -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(max(6, 0.4 * len(col_labels)), max(3, 0.3 * len(row_labels))), dpi=180)
    image = axis.imshow(matrix, aspect="auto", cmap="magma")
    axis.set_xticks(np.arange(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(_ensure_path(save_path), bbox_inches="tight")
    plt.close(figure)
