from __future__ import annotations

from typing import Dict

import numpy as np


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mae = np.abs(pred - target).mean()
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    denom = np.maximum(np.abs(target), 1e-6)
    mape = np.mean(np.abs((pred - target) / denom))
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape)}


def horizon_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, np.ndarray]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    mae = np.abs(pred - target).mean(axis=0)
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=0))
    denom = np.maximum(np.abs(target), 1e-6)
    mape = np.mean(np.abs((pred - target) / denom), axis=0)
    return {"mae": mae.astype(np.float32), "rmse": rmse.astype(np.float32), "mape": mape.astype(np.float32)}
