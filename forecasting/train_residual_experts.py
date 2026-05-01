"""forecasting.train_residual_experts

Stage C: 冻结 base model，只训练 physical_router 和 residual experts。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from forecasting.data import BatterySOHForecastDataset
from forecasting.losses import compute_residual_expert_loss
from forecasting.model import BatterySOHForecaster
from forecasting.train import infer_model_init_from_dataset, load_config, move_batch_to_device, resolve_device


def _freeze_for_residual_training(model: BatterySOHForecaster) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module in [model.physical_router, model.expert_bank]:
        for param in module.parameters():
            param.requires_grad = True


def train_residual_experts(cfg: Dict[str, object], checkpoint_path: str | Path | None = None) -> Dict[str, object]:
    case_bank_dir = Path(cfg.get("output_dir", "output/case_bank"))
    dataset = BatterySOHForecastDataset(
        case_bank_dir=case_bank_dir,
        splits=list(cfg.get("train", {}).get("train_splits", ["source_train"])),
        retrieval_cfg=dict(cfg.get("retrieval", {})),
    )
    if dataset.arrays.get("residual_target_oof") is None:
        raise FileNotFoundError("Missing case_residual_target_oof.npy. Run forecasting.generate_baseline_oof first.")

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = BatterySOHForecaster(**checkpoint["model_init"])
        model.load_state_dict(checkpoint["model_state"])
        model_init = checkpoint["model_init"]
    else:
        model_init = infer_model_init_from_dataset(dataset, cfg)
        model = BatterySOHForecaster(**model_init)

    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    model.to(device)
    _freeze_for_residual_training(model)
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(cfg.get("train_residual_experts", {}).get("lr", 5e-4)),
    )
    loader = DataLoader(dataset, batch_size=int(cfg.get("train", {}).get("batch_size", 64)), shuffle=True, num_workers=0)

    output_dir = Path(cfg.get("model_output_dir", "output/forecasting"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_epoch = -1
    logs = []
    patience = int(cfg.get("train_residual_experts", {}).get("patience", 12))
    epochs = int(cfg.get("train_residual_experts", {}).get("epochs", 60))

    for epoch in range(1, epochs + 1):
        model.train(True)
        epoch_losses = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss_dict = compute_residual_expert_loss(outputs, batch, cfg.get("loss", {}))
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(loss_dict["loss"].detach().cpu().item()))
        mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        logs.append({"epoch": epoch, "loss": mean_loss})
        payload = {
            "model_state": model.state_dict(),
            "model_init": model_init,
            "config": cfg,
            "epoch": epoch,
            "loss": mean_loss,
        }
        torch.save(payload, checkpoint_dir / "residual_experts_last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            torch.save(payload, checkpoint_dir / "residual_experts_best.pt")
        if epoch - best_epoch >= patience:
            break

    log_df = pd.DataFrame(logs)
    log_df.to_csv(output_dir / "residual_expert_train_log.csv", index=False)
    (output_dir / "residual_expert_train_log.json").write_text(json.dumps(logs, indent=2, ensure_ascii=True))
    return {
        "best_checkpoint": str(checkpoint_dir / "residual_experts_best.pt"),
        "last_checkpoint": str(checkpoint_dir / "residual_experts_last.pt"),
        "best_loss": best_loss,
    }


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train residual experts on OOF residual targets")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = train_residual_experts(cfg, checkpoint_path=args.checkpoint)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
