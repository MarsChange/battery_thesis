from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from forecasting.data import BatterySOHForecastDataset
from forecasting.losses import compute_base_model_loss
from forecasting.metrics import horizon_metrics, regression_metrics
from forecasting.model import BatterySOHForecaster
from forecasting.visualization import plot_training_curves


def load_config(path: str) -> dict:
    import yaml

    with open(path) as handle:
        return yaml.safe_load(handle)


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def move_batch_to_device(batch, device: str):
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    return batch


def infer_model_init_from_dataset(dataset: BatterySOHForecastDataset, cfg: Dict[str, object]) -> Dict[str, object]:
    arrays = dataset.arrays
    model_cfg = cfg.get("model", {})
    return {
        "horizon": int(arrays["future_delta_soh"].shape[1]),
        "cycle_feature_dim": int(arrays["cycle_stats"].shape[-1]),
        "physics_dim": int(arrays["physics_features"].shape[-1]),
        "operation_dim": int(arrays["operation_seq"].shape[-1]),
        "future_operation_dim": int(arrays["future_ops"].shape[-1]),
        "meta_dim": len(dataset.meta_fields),
        "qv_width": int(arrays["qv_maps"].shape[-1]),
        "partial_charge_points": int(arrays["partial_charge"].shape[-1]),
        "expert_seq_dim": int(arrays["expert_seq"].shape[-1]) if arrays.get("expert_seq") is not None else 14,
        "hidden_dim": int(model_cfg.get("hidden_dim", 256)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "meta_embedding_dim": int(model_cfg.get("meta_embedding_dim", 16)),
        "expert_names": list(model_cfg.get("expert_names", [])),
        "top_k_experts": int(model_cfg.get("top_k_experts", 2)),
    }


def run_epoch(
    model: BatterySOHForecaster,
    loader: DataLoader,
    device: str,
    optimizer: AdamW | None,
    loss_cfg: Dict[str, object],
    grad_clip_norm: float = 0.0,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], List[Dict[str, object]]]:
    is_train = optimizer is not None
    model.train(is_train)

    all_pred = []
    all_target = []
    loss_accumulator = []
    expert_weights = []
    fusion_weights = []
    retrieval_confidence = []

    with torch.set_grad_enabled(is_train):
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss_dict = compute_base_model_loss(outputs, batch, loss_cfg)
            if is_train:
                optimizer.zero_grad()
                loss_dict["loss"].backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            loss_accumulator.append({key: float(value.detach().cpu().item()) for key, value in loss_dict.items()})
            all_pred.append(outputs["pred_delta"].detach().cpu().numpy())
            all_target.append(batch["query"]["target_delta_soh"].detach().cpu().numpy())
            expert_weights.append(outputs["expert_weights"].detach().cpu().numpy())
            fusion_weights.append(outputs["fusion_weights"].detach().cpu().numpy())
            retrieval_confidence.append(outputs["retrieval_confidence"].detach().cpu().numpy())

    pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0, 0), dtype=np.float32)
    target = np.concatenate(all_target, axis=0) if all_target else np.zeros((0, 0), dtype=np.float32)
    metrics = regression_metrics(pred, target) if pred.size else {"mae": np.nan, "rmse": np.nan, "mape": np.nan}
    h_metrics = horizon_metrics(pred, target) if pred.size else {"mae": np.zeros(0), "rmse": np.zeros(0), "mape": np.zeros(0)}

    loss_summary = {}
    if loss_accumulator:
        for key in loss_accumulator[0]:
            loss_summary[key] = float(np.mean([row[key] for row in loss_accumulator]))
    extra = {
        "expert_weights_mean": np.mean(np.concatenate(expert_weights, axis=0), axis=0) if expert_weights else np.zeros(0),
        "expert_weights_std": np.std(np.concatenate(expert_weights, axis=0), axis=0) if expert_weights else np.zeros(0),
        "fusion_weights_mean": np.mean(np.concatenate(fusion_weights, axis=0), axis=0) if fusion_weights else np.zeros(0),
        "fusion_weights_std": np.std(np.concatenate(fusion_weights, axis=0), axis=0) if fusion_weights else np.zeros(0),
        "retrieval_confidence_mean": float(np.mean(np.concatenate(retrieval_confidence, axis=0))) if retrieval_confidence else 0.0,
        "retrieval_confidence_std": float(np.std(np.concatenate(retrieval_confidence, axis=0))) if retrieval_confidence else 0.0,
    }
    return {**loss_summary, **metrics}, h_metrics, [extra]


def build_datasets(cfg: Dict[str, object]) -> Tuple[BatterySOHForecastDataset, BatterySOHForecastDataset]:
    case_bank_dir = cfg.get("output_dir", "output/case_bank")
    retrieval_cfg = dict(cfg.get("retrieval", {}))
    train_dataset = BatterySOHForecastDataset(
        case_bank_dir=case_bank_dir,
        splits=list(cfg.get("train", {}).get("train_splits", ["source_train"])),
        retrieval_cfg=retrieval_cfg,
    )
    val_dataset = BatterySOHForecastDataset(
        case_bank_dir=case_bank_dir,
        splits=list(cfg.get("train", {}).get("val_splits", ["source_val"])),
        retrieval_cfg=retrieval_cfg,
    )
    return train_dataset, val_dataset


def train(cfg: Dict[str, object]) -> Dict[str, object]:
    train_dataset, val_dataset = build_datasets(cfg)
    model_init = infer_model_init_from_dataset(train_dataset, cfg)
    model = BatterySOHForecaster(**model_init)

    train_cfg = cfg.get("train", {})
    device = resolve_device(str(train_cfg.get("device", "auto")))
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    batch_size = int(train_cfg.get("batch_size", 64))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    output_dir = Path(cfg.get("model_output_dir", "output/forecasting"))
    checkpoint_dir = output_dir / "checkpoints"
    figure_dir = output_dir / "figures" / "training"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "meta_vocab.json").write_text(json.dumps(train_dataset.meta_vocab, indent=2, ensure_ascii=True))
    normalization_path = Path(cfg.get("output_dir", "output/case_bank")) / "normalization_stats.json"
    if normalization_path.exists():
        (output_dir / "normalization_stats.json").write_text(normalization_path.read_text())
    import yaml

    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    history_rows = []
    best_val = float("inf")
    best_epoch = -1
    patience = int(train_cfg.get("patience", 15))
    grad_clip_norm = float(train_cfg.get("grad_clip_norm", 1.0))

    epochs = int(train_cfg.get("epochs", 100))
    for epoch in range(1, epochs + 1):
        model.train(True)
        train_metrics, _, _ = run_epoch(model, train_loader, device, optimizer, cfg.get("loss", {}), grad_clip_norm=grad_clip_norm)
        model.train(False)
        val_metrics, h_metrics, extras = run_epoch(model, val_loader, device, None, cfg.get("loss", {}))
        if not np.isfinite(val_metrics.get("loss", np.nan)):
            val_metrics = dict(train_metrics)
        extra = extras[0]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_mape": val_metrics["mape"],
            "horizon_mae_json": json.dumps(h_metrics["mae"].astype(float).tolist()),
            "expert_weight_mean_json": json.dumps(extra["expert_weights_mean"].astype(float).tolist()),
            "expert_weight_std_json": json.dumps(extra["expert_weights_std"].astype(float).tolist()),
            "fusion_weight_mean_json": json.dumps(extra["fusion_weights_mean"].astype(float).tolist()),
            "fusion_weight_std_json": json.dumps(extra["fusion_weights_std"].astype(float).tolist()),
            "retrieval_confidence_mean": extra["retrieval_confidence_mean"],
            "retrieval_confidence_std": extra["retrieval_confidence_std"],
        }
        history_rows.append(row)

        checkpoint = {
            "model_state": model.state_dict(),
            "model_init": model_init,
            "config": cfg,
            "meta_vocab": train_dataset.meta_vocab,
            "meta_fields": train_dataset.meta_fields,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, checkpoint_dir / "base_model_last.pt")
        if val_metrics["loss"] < best_val:
            best_val = float(val_metrics["loss"])
            best_epoch = epoch
            torch.save(checkpoint, checkpoint_dir / "base_model_best.pt")
        if epoch - best_epoch >= patience:
            break

    history = pd.DataFrame(history_rows)
    history.to_csv(output_dir / "train_log.csv", index=False)
    (output_dir / "train_log.json").write_text(json.dumps(history.to_dict(orient="records"), indent=2, ensure_ascii=True))
    plot_training_curves(history, figure_dir)
    return {
        "output_dir": str(output_dir),
        "best_checkpoint": str(checkpoint_dir / "base_model_best.pt"),
        "last_checkpoint": str(checkpoint_dir / "base_model_last.pt"),
        "best_val_loss": float(best_val),
        "epochs_completed": int(len(history_rows)),
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train battery SOH forecasting model")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = train(cfg)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
