from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from forecasting.data import BatterySOHForecastDataset
from forecasting.losses import compute_total_loss
from forecasting.model import BatterySOHForecaster
from forecasting.train import load_config, move_batch_to_device, resolve_device


def _freeze_modules(model: BatterySOHForecaster, cfg: Dict[str, object]) -> None:
    fewshot_cfg = cfg.get("fewshot", {})
    if bool(fewshot_cfg.get("freeze_curve_encoders", True)):
        for module in [model.qv_encoder, model.partial_charge_encoder, model.relaxation_encoder]:
            for param in module.parameters():
                param.requires_grad = False
    if bool(fewshot_cfg.get("freeze_generalist_head", True)):
        for param in model.generalist_head.parameters():
            param.requires_grad = False
    if not bool(fewshot_cfg.get("train_router", True)):
        for param in model.physical_router.parameters():
            param.requires_grad = False
    if not bool(fewshot_cfg.get("train_fusion_router", True)):
        for param in model.fusion_router.parameters():
            param.requires_grad = False
    if not bool(fewshot_cfg.get("train_experts", True)):
        for param in model.expert_bank.parameters():
            param.requires_grad = False
    if not bool(fewshot_cfg.get("train_pairwise_branch", True)):
        for param in model.pairwise_branch.parameters():
            param.requires_grad = False


def fewshot_adapt(cfg: Dict[str, object], checkpoint_path: str | Path) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = BatterySOHForecaster(**checkpoint["model_init"])
    model.load_state_dict(checkpoint["model_state"])

    support_splits = list(cfg.get("fewshot", {}).get("support_splits", ["target_support"]))
    dataset = BatterySOHForecastDataset(
        case_bank_dir=cfg.get("output_dir", "output/case_bank"),
        splits=support_splits,
        retrieval_cfg=dict(cfg.get("retrieval", {})),
    )
    if len(dataset) == 0:
        raise ValueError("Few-shot adaptation requires non-empty target_support.")

    device = resolve_device(str(cfg.get("train", {}).get("device", "auto")))
    model.to(device)
    _freeze_modules(model, cfg)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable, lr=float(cfg.get("fewshot", {}).get("lr", 3e-4)))
    loader = DataLoader(dataset, batch_size=min(int(cfg.get("train", {}).get("batch_size", 64)), len(dataset)), shuffle=True)

    output_dir = Path(cfg.get("model_output_dir", "output/forecasting"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_path = checkpoint_dir / "best_adapted.pt"
    last_path = checkpoint_dir / "last_adapted.pt"
    patience = int(cfg.get("fewshot", {}).get("patience", 10))
    best_epoch = -1
    logs = []

    for epoch in range(1, int(cfg.get("fewshot", {}).get("epochs", 50)) + 1):
        model.train(True)
        epoch_losses = []
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss_dict = compute_total_loss(outputs, batch, cfg.get("loss", {}))
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
            epoch_losses.append(float(loss_dict["loss"].detach().cpu().item()))
        mean_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        logs.append({"epoch": epoch, "support_loss": mean_loss})
        checkpoint_payload = {
            "model_state": model.state_dict(),
            "model_init": checkpoint["model_init"],
            "config": cfg,
            "meta_vocab": checkpoint.get("meta_vocab", dataset.meta_vocab),
            "meta_fields": checkpoint.get("meta_fields", dataset.meta_fields),
            "epoch": epoch,
            "support_loss": mean_loss,
        }
        torch.save(checkpoint_payload, last_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_epoch = epoch
            torch.save(checkpoint_payload, best_path)
        if epoch - best_epoch >= patience:
            break

    log_path = output_dir / "fewshot_log.json"
    log_path.write_text(json.dumps(logs, indent=2, ensure_ascii=True))
    import pandas as pd

    log_df = pd.DataFrame(logs)
    log_df.to_csv(output_dir / "fewshot_log.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=180)
    axis.plot(log_df["epoch"], log_df["support_loss"], marker="o")
    axis.set_title("Few-shot support loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / "fewshot_loss_curve.png", bbox_inches="tight")
    plt.close(figure)
    return {"best_adapted_checkpoint": str(best_path), "last_adapted_checkpoint": str(last_path), "best_support_loss": best_loss}


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Few-shot adapt battery SOH forecasting model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = fewshot_adapt(cfg, args.checkpoint)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
