from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def forecast_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor, criterion: str = "huber") -> torch.Tensor:
    if criterion == "mse":
        return F.mse_loss(pred_delta, target_delta)
    return F.huber_loss(pred_delta, target_delta)


def pairwise_aux_loss(
    pair_residual: torch.Tensor,
    target_delta: torch.Tensor,
    ref_future_delta_soh: torch.Tensor,
    retrieval_alpha: torch.Tensor,
    retrieval_mask: torch.Tensor,
) -> torch.Tensor:
    target_pair_residual = target_delta[:, None, :] - ref_future_delta_soh
    loss = F.huber_loss(pair_residual, target_pair_residual, reduction="none").mean(dim=-1)
    weight = retrieval_alpha * retrieval_mask
    denom = weight.sum().clamp_min(1e-6)
    return (loss * weight).sum() / denom


def monotonic_loss(
    pred_soh: torch.Tensor,
    epsilon: float = 5e-4,
) -> torch.Tensor:
    if pred_soh.size(1) < 2:
        return pred_soh.new_tensor(0.0)
    violation = F.relu(pred_soh[:, 1:] - pred_soh[:, :-1] - float(epsilon))
    return violation.mean()


def smoothness_loss(pred_soh: torch.Tensor) -> torch.Tensor:
    if pred_soh.size(1) < 3:
        return pred_soh.new_tensor(0.0)
    second_diff = pred_soh[:, 2:] - 2 * pred_soh[:, 1:-1] + pred_soh[:, :-2]
    return second_diff.abs().mean()


def expert_load_balance_loss(expert_weights: torch.Tensor) -> torch.Tensor:
    if expert_weights.numel() == 0:
        return expert_weights.new_tensor(0.0)
    mean_weight = expert_weights.mean(dim=0)
    num_experts = expert_weights.size(-1)
    uniform = torch.full_like(mean_weight, 1.0 / max(num_experts, 1))
    return F.kl_div((mean_weight + 1e-8).log(), uniform, reduction="batchmean")


def retrieval_consistency_loss(
    pred_delta: torch.Tensor,
    rag_delta: torch.Tensor,
    retrieval_confidence: torch.Tensor,
) -> torch.Tensor:
    diff = (pred_delta - rag_delta).abs().mean(dim=-1)
    return (diff * retrieval_confidence).mean()


def fusion_regularization(fusion_weights: torch.Tensor) -> torch.Tensor:
    if fusion_weights.numel() == 0:
        return fusion_weights.new_tensor(0.0)
    entropy = -(fusion_weights * (fusion_weights + 1e-8).log()).sum(dim=-1)
    return (-entropy).mean()


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Dict[str, torch.Tensor]],
    loss_cfg: Dict[str, float | str],
) -> Dict[str, torch.Tensor]:
    target_delta = batch["query"]["target_delta_soh"]
    pred_delta = outputs["pred_delta"]
    pred_soh = outputs["pred_soh"]
    forecast = forecast_loss(pred_delta, target_delta, criterion=str(loss_cfg.get("criterion", "huber")))
    pairwise = pairwise_aux_loss(
        outputs["pair_residual"],
        target_delta,
        batch["retrieval"]["ref_future_delta_soh"],
        batch["retrieval"]["retrieval_alpha"],
        batch["retrieval"]["retrieval_mask"],
    )
    mono = monotonic_loss(pred_soh, epsilon=float(loss_cfg.get("monotonic_epsilon", 5e-4)))
    smooth = smoothness_loss(pred_soh)
    expert_balance = expert_load_balance_loss(outputs["expert_weights"])
    retrieval_consistency = retrieval_consistency_loss(
        pred_delta,
        outputs["rag_delta"],
        outputs["retrieval_confidence"],
    )
    fusion_reg = fusion_regularization(outputs["fusion_weights"])
    total = (
        float(loss_cfg.get("forecast", 1.0)) * forecast
        + float(loss_cfg.get("pairwise", 0.5)) * pairwise
        + float(loss_cfg.get("monotonic", 0.05)) * mono
        + float(loss_cfg.get("smoothness", 0.01)) * smooth
        + float(loss_cfg.get("expert_balance", 0.01)) * expert_balance
        + float(loss_cfg.get("retrieval_consistency", 0.02)) * retrieval_consistency
        + float(loss_cfg.get("fusion_regularization", 0.0)) * fusion_reg
    )
    return {
        "loss": total,
        "forecast_loss": forecast,
        "pairwise_aux_loss": pairwise,
        "monotonic_loss": mono,
        "smoothness_loss": smooth,
        "expert_load_balance_loss": expert_balance,
        "retrieval_consistency_loss": retrieval_consistency,
        "fusion_regularization": fusion_reg,
    }
