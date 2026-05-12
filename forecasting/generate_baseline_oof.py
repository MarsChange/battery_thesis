"""forecasting.generate_baseline_oof

Stage B: 按 cell_uid 划分 OOF folds，为 source_train 生成 baseline_delta_oof 和 residual_target_oof。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

tqdm.monitor_interval = 0

from forecasting.data import BatterySOHForecastDataset, RESIDUAL_FACTOR_NAMES, compute_residual_factor_scores
from forecasting.losses import compute_base_model_loss
from forecasting.model import BatterySOHForecaster
from forecasting.train import freeze_for_base_training, infer_model_init_from_dataset, load_config, move_batch_to_device, resolve_device


OOF_ARTIFACT_FILENAMES = [
    "case_baseline_delta_oof.npy",
    "case_residual_target_oof.npy",
    "case_residual_factor_scores.npy",
    "case_residual_norm_stats.json",
]


def _case_bank_signature(case_bank_dir: Path) -> str:
    """Return a cheap fingerprint for files that define OOF residual targets."""

    relevant_files = [
        "case_rows.parquet",
        "case_rows.csv",
        "case_soh_seq.npy",
        "case_future_delta_soh.npy",
        "case_expert_seq.npy",
    ]
    payload = []
    for filename in relevant_files:
        path = case_bank_dir / filename
        if not path.exists():
            continue
        stat = path.stat()
        payload.append({"name": filename, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _oof_config_signature(cfg: Dict[str, object], model_init: Dict[str, object]) -> str:
    """Fingerprint the settings that affect OOF fold predictions."""

    relevant = {
        "generate_oof_baseline": {
            key: value
            for key, value in dict(cfg.get("generate_oof_baseline", {})).items()
            if key not in {"enabled", "reuse_existing"}
        },
        "train": {
            key: dict(cfg.get("train", {})).get(key)
            for key in ["batch_size", "lr", "weight_decay", "grad_clip_norm", "device"]
        },
        "loss": cfg.get("loss", {}),
        "model_init": model_init,
        "retrieval": cfg.get("retrieval", {}),
    }
    text = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _can_reuse_existing_oof(
    case_bank_dir: Path,
    dataset: BatterySOHForecastDataset,
    cfg: Dict[str, object],
    model_init: Dict[str, object],
) -> bool:
    if not all((case_bank_dir / filename).exists() for filename in OOF_ARTIFACT_FILENAMES):
        return False
    stats_path = case_bank_dir / "case_residual_norm_stats.json"
    try:
        stats = json.loads(stats_path.read_text())
        baseline = np.load(case_bank_dir / "case_baseline_delta_oof.npy", mmap_mode="r")
        residual = np.load(case_bank_dir / "case_residual_target_oof.npy", mmap_mode="r")
        factor_scores = np.load(case_bank_dir / "case_residual_factor_scores.npy", mmap_mode="r")
    except Exception:
        return False
    expected_delta_shape = tuple(dataset.arrays["future_delta_soh"].shape)
    expected_factor_shape = (int(len(dataset.case_rows)), len(RESIDUAL_FACTOR_NAMES))
    if tuple(baseline.shape) != expected_delta_shape or tuple(residual.shape) != expected_delta_shape:
        return False
    if tuple(factor_scores.shape) != expected_factor_shape:
        return False
    return (
        stats.get("case_bank_signature") == _case_bank_signature(case_bank_dir)
        and stats.get("oof_config_signature") == _oof_config_signature(cfg, model_init)
        and stats.get("folding_key") == "cell_uid"
    )


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


def _train_fold_model(
    train_subset: Subset,
    model_init: Dict[str, object],
    cfg: Dict[str, object],
    device: str,
    *,
    fold_id: int,
    num_folds: int,
) -> BatterySOHForecaster:
    model = BatterySOHForecaster(**model_init).to(device)
    freeze_for_base_training(model)
    oof_cfg = cfg.get("generate_oof_baseline", {})
    train_cfg = cfg.get("train", {})
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(oof_cfg.get("lr", train_cfg.get("lr", 1e-3))),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(cfg.get("generate_oof_baseline", {}).get("epochs", 5))
    batch_size = min(int(cfg.get("train", {}).get("batch_size", 64)), max(len(train_subset), 1))
    num_workers = int(train_cfg.get("num_workers", 0))
    loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=bool(num_workers > 0),
    )
    show_progress = bool(cfg.get("train", {}).get("show_progress", True))
    for epoch in range(1, epochs + 1):
        model.train(True)
        progress = tqdm(
            loader,
            desc=f"OOF fold {fold_id + 1:02d}/{num_folds:02d} epoch {epoch:03d}/{epochs:03d}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )
        running_loss = 0.0
        for step, batch in enumerate(progress, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss_dict = compute_base_model_loss(outputs, batch, cfg.get("loss", {}))
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            grad_clip_norm = float(train_cfg.get("grad_clip_norm", 0.0) or 0.0)
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            running_loss += float(loss_dict["loss"].detach().cpu().item())
            progress.set_postfix(loss=f"{running_loss / step:.6f}")
        if show_progress:
            tqdm.write(
                f"OOF fold {fold_id + 1:02d}/{num_folds:02d} "
                f"epoch {epoch:03d}/{epochs:03d} loss={running_loss / max(len(loader), 1):.6f}"
            )
    return model


def generate_oof_baseline(cfg: Dict[str, object]) -> Dict[str, object]:
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    dataset = BatterySOHForecastDataset(
        case_bank_dir=case_bank_dir,
        splits=["source_train"],
        retrieval_cfg=dict(cfg.get("retrieval", {})),
    )
    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    model_init = infer_model_init_from_dataset(dataset, cfg)
    oof_cfg = cfg.get("generate_oof_baseline", {})
    if bool(oof_cfg.get("reuse_existing", False)) and _can_reuse_existing_oof(case_bank_dir, dataset, cfg, model_init):
        return {
            "case_bank_dir": str(case_bank_dir),
            "num_source_train_cases": int((dataset.case_rows["split"].astype(str) == "source_train").sum()),
            "num_folds": int(oof_cfg.get("num_folds", 5)),
            "reused_existing": True,
        }

    full_rows = dataset.case_rows.reset_index(drop=True)
    total_cases = len(full_rows)
    baseline_delta = np.zeros((total_cases, dataset.arrays["future_delta_soh"].shape[1]), dtype=np.float32)
    residual_target = np.zeros_like(baseline_delta)
    fold_assignments = np.full(total_cases, -1, dtype=np.int64)

    folds = make_cell_uid_folds(full_rows, dataset.indices.tolist(), int(cfg.get("generate_oof_baseline", {}).get("num_folds", 5)))
    source_train_index_set = set(dataset.indices.tolist())

    show_progress = bool(cfg.get("train", {}).get("show_progress", True))
    for fold_id, heldout_indices in enumerate(tqdm(folds, desc="OOF folds", unit="fold", disable=not show_progress)):
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
            fold_id=fold_id,
            num_folds=len(folds),
        )
        model.eval()
        heldout_subset = Subset(dataset, [dataset.query_case_id_to_local[int(dataset.case_rows.iloc[idx]["case_id"])] for idx in heldout_indices])
        num_workers = int(cfg.get("train", {}).get("num_workers", 0))
        loader = DataLoader(
            heldout_subset,
            batch_size=min(len(heldout_subset), int(cfg.get("train", {}).get("batch_size", 64))),
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=bool(num_workers > 0),
        )
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
    factor_scores = compute_residual_factor_scores(dataset.case_rows, dataset.arrays.get("expert_seq"))
    np.save(case_bank_dir / "case_residual_factor_scores.npy", factor_scores.astype(np.float32))
    factor_context_columns = ["case_id", "cell_uid", "split", "source_dataset", "chemistry_family", "domain_label", "anchor_soh"]
    factor_frame = dataset.case_rows[[col for col in factor_context_columns if col in dataset.case_rows.columns]].copy()
    for column_idx, column_name in enumerate(RESIDUAL_FACTOR_NAMES):
        factor_frame[column_name] = factor_scores[:, column_idx].astype(np.float32)
        factor_frame[f"{column_name}_active"] = factor_frame[column_name] >= 0.5
    factor_frame.to_csv(case_bank_dir / "case_residual_factor_scores.csv", index=False)
    stats = {
        "mean": residual_mean,
        "std": residual_std,
        "fold_assignments": fold_assignments.astype(int).tolist(),
        "num_folds": len(folds),
        "folding_key": "cell_uid",
        "case_bank_signature": _case_bank_signature(case_bank_dir),
        "oof_config_signature": _oof_config_signature(cfg, model_init),
        "source_train_case_ids": dataset.case_rows.loc[source_train_rows, "case_id"].astype(int).tolist(),
        "factor_score_columns": RESIDUAL_FACTOR_NAMES,
        "factor_active_threshold": 0.5,
    }
    (case_bank_dir / "case_residual_norm_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=True))
    return {
        "case_bank_dir": str(case_bank_dir),
        "num_source_train_cases": int(len(source_train_rows)),
        "num_folds": int(len(folds)),
        "reused_existing": False,
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
