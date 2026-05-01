"""forecasting.generate_baseline_oof

Stage B: 按 cell_uid 划分 OOF folds，为 source_train 生成 baseline_delta_oof 和 residual_target_oof。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from forecasting.data import BatterySOHForecastDataset
from forecasting.losses import compute_base_model_loss
from forecasting.model import BatterySOHForecaster
from forecasting.train import infer_model_init_from_dataset, load_config, move_batch_to_device, resolve_device


def make_cell_uid_folds(case_rows, indices: Iterable[int], num_folds: int) -> List[List[int]]:
    rows = case_rows.iloc[list(indices)].copy()
    rows["cell_uid"] = rows["cell_uid"].astype(str)
    unique_cells = sorted(rows["cell_uid"].unique().tolist())
    folds = [[] for _ in range(max(int(num_folds), 1))]
    for position, cell_uid in enumerate(unique_cells):
        fold_id = position % len(folds)
        cell_indices = rows.index[rows["cell_uid"] == cell_uid].tolist()
        folds[fold_id].extend(cell_indices)
    return folds


def _train_fold_model(train_subset: Subset, model_init: Dict[str, object], cfg: Dict[str, object], device: str) -> BatterySOHForecaster:
    model = BatterySOHForecaster(**model_init).to(device)
    optimizer = AdamW(model.parameters(), lr=float(cfg.get("generate_oof_baseline", {}).get("lr", 5e-4)))
    epochs = int(cfg.get("generate_oof_baseline", {}).get("epochs", 5))
    batch_size = min(int(cfg.get("train", {}).get("batch_size", 64)), max(len(train_subset), 1))
    loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train(True)
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss_dict = compute_base_model_loss(outputs, batch, cfg.get("loss", {}))
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
    return model


def generate_oof_baseline(cfg: Dict[str, object]) -> Dict[str, object]:
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    dataset = BatterySOHForecastDataset(case_bank_dir=case_bank_dir, splits=["source_train"], retrieval_cfg={})
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    model_init = infer_model_init_from_dataset(dataset, cfg)

    full_rows = dataset.case_rows.reset_index(drop=True)
    total_cases = len(full_rows)
    baseline_delta = np.zeros((total_cases, dataset.arrays["future_delta_soh"].shape[1]), dtype=np.float32)
    residual_target = np.zeros_like(baseline_delta)
    fold_assignments = np.full(total_cases, -1, dtype=np.int64)

    folds = make_cell_uid_folds(full_rows, dataset.indices.tolist(), int(cfg.get("generate_oof_baseline", {}).get("num_folds", 5)))
    source_train_index_set = set(dataset.indices.tolist())

    for fold_id, heldout_indices in enumerate(folds):
        heldout_indices = [idx for idx in heldout_indices if idx in source_train_index_set]
        if not heldout_indices:
            continue
        train_indices = [idx for idx in dataset.indices.tolist() if idx not in heldout_indices]
        if not train_indices:
            continue
        fold_assignments[heldout_indices] = fold_id
        model = _train_fold_model(
            train_subset=Subset(dataset, [dataset.query_case_id_to_local[int(dataset.case_rows.iloc[idx]["case_id"])] for idx in train_indices]),
            model_init=model_init,
            cfg=cfg,
            device=device,
        )
        model.eval()
        heldout_subset = Subset(dataset, [dataset.query_case_id_to_local[int(dataset.case_rows.iloc[idx]["case_id"])] for idx in heldout_indices])
        loader = DataLoader(heldout_subset, batch_size=min(len(heldout_subset), int(cfg.get("train", {}).get("batch_size", 64))), shuffle=False)
        pointer = 0
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                outputs = model(batch)
                base_delta = outputs["base_delta"].detach().cpu().numpy().astype(np.float32)
                batch_size = base_delta.shape[0]
                original_rows = heldout_indices[pointer : pointer + batch_size]
                baseline_delta[original_rows] = base_delta
                target = batch["query"]["target_delta_soh"].detach().cpu().numpy().astype(np.float32)
                residual_target[original_rows] = target - base_delta
                pointer += batch_size

    source_train_rows = dataset.case_rows.index[dataset.case_rows["split"].astype(str) == "source_train"].to_numpy(dtype=np.int64)
    residual_values = residual_target[source_train_rows]
    residual_mean = residual_values.mean(axis=0).astype(float).tolist() if len(source_train_rows) else []
    residual_std = np.maximum(residual_values.std(axis=0), 1e-6).astype(float).tolist() if len(source_train_rows) else []

    np.save(case_bank_dir / "case_baseline_delta_oof.npy", baseline_delta.astype(np.float32))
    np.save(case_bank_dir / "case_residual_target_oof.npy", residual_target.astype(np.float32))
    stats = {
        "mean": residual_mean,
        "std": residual_std,
        "fold_assignments": fold_assignments.astype(int).tolist(),
        "num_folds": len(folds),
        "folding_key": "cell_uid",
    }
    (case_bank_dir / "case_residual_norm_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True))
    return {
        "case_bank_dir": str(case_bank_dir),
        "num_source_train_cases": int(len(source_train_rows)),
        "num_folds": int(len(folds)),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate cell-wise OOF baseline predictions")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = generate_oof_baseline(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
