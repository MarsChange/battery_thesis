from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_bar(data: Dict[str, float], title: str, ylabel: str, save_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    labels = list(data.keys())
    values = [float(data[label]) for label in labels]
    axis.bar(np.arange(len(labels)), values, color="#2563eb")
    axis.set_xticks(np.arange(len(labels)))
    axis.set_xticklabels(labels, rotation=30, ha="right")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, save_path: str | Path, cmap: str = "viridis") -> None:
    matrix = np.asarray(matrix, dtype=np.float32)
    figure, axis = plt.subplots(figsize=(max(6, 0.5 * len(col_labels)), max(3, 0.35 * len(row_labels))), dpi=180)
    image = axis.imshow(matrix, aspect="auto", cmap=cmap)
    axis.set_xticks(np.arange(len(col_labels)))
    axis.set_xticklabels(col_labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_boxplot(grouped_values: Dict[str, List[float]], title: str, ylabel: str, save_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    labels = list(grouped_values.keys())
    values = [grouped_values[label] if grouped_values[label] else [0.0] for label in labels]
    axis.boxplot(values, tick_labels=labels, showfliers=False)
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)


def save_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str, save_path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4.5), dpi=180)
    axis.scatter(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), s=14, alpha=0.7)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, bbox_inches="tight")
    plt.close(figure)
