"""forecasting.losses

定义基础模型阶段和 residual experts 阶段的损失函数。
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def horizon_weights(horizon: int, device: torch.device, dtype: torch.dtype, end_weight: float = 1.2) -> torch.Tensor:
    if horizon <= 1:
        return torch.ones(horizon, device=device, dtype=dtype)
    return torch.linspace(1.0, float(end_weight), horizon, device=device, dtype=dtype)


def weighted_mse_forecast_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor, end_weight: float = 1.2) -> torch.Tensor:
    weights = horizon_weights(pred_delta.shape[-1], pred_delta.device, pred_delta.dtype, end_weight=end_weight)
    return ((pred_delta - target_delta).pow(2) * weights.unsqueeze(0)).mean()


def forecast_loss(
    pred_delta: torch.Tensor,
    target_delta: torch.Tensor,
    criterion: str = "weighted_mse",
    horizon_end_weight: float = 1.2,
) -> torch.Tensor:
    if criterion in {"weighted_mse", "mse_weighted"}:
        return weighted_mse_forecast_loss(pred_delta, target_delta, end_weight=horizon_end_weight)
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


def monotonic_loss(pred_soh: torch.Tensor, epsilon: float = 5e-4) -> torch.Tensor:
    if pred_soh.size(1) < 2:
        return pred_soh.new_tensor(0.0)
    return F.relu(pred_soh[:, 1:] - pred_soh[:, :-1] - float(epsilon)).mean()


def smoothness_loss(pred_soh: torch.Tensor) -> torch.Tensor:
    if pred_soh.size(1) < 3:
        return pred_soh.new_tensor(0.0)
    second_diff = pred_soh[:, 2:] - 2 * pred_soh[:, 1:-1] + pred_soh[:, :-2]
    return second_diff.abs().mean()


def total_variation_loss(sequence: torch.Tensor) -> torch.Tensor:
    """Mean absolute first difference used to suppress residual sawtooth noise."""

    if sequence.size(1) < 2:
        return sequence.new_tensor(0.0)
    return (sequence[:, 1:] - sequence[:, :-1]).abs().mean()


def expert_load_balance_loss(expert_weights: torch.Tensor) -> torch.Tensor:
    if expert_weights.numel() == 0:
        return expert_weights.new_tensor(0.0)
    mean_weight = expert_weights.mean(dim=0)
    uniform = torch.full_like(mean_weight, 1.0 / max(expert_weights.size(-1), 1))
    return F.kl_div((mean_weight + 1e-8).log(), uniform, reduction="batchmean")


def route_kl_loss(final_prior: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    """KL(stopgrad semantic prior || final expert weights)."""

    prior = final_prior.detach().clamp_min(1e-8)
    weights = expert_weights.clamp_min(1e-8)
    return F.kl_div(weights.log(), prior, reduction="batchmean")


def retrieval_consistency_loss(
    base_delta: torch.Tensor,
    rag_delta: torch.Tensor,
    retrieval_confidence: torch.Tensor,
) -> torch.Tensor:
    diff = (base_delta - rag_delta).abs().mean(dim=-1)
    return (diff * retrieval_confidence).mean()


def residual_supervision_loss(
    moe_residual: torch.Tensor,
    residual_target: torch.Tensor,
    criterion: str = "weighted_mse",
    horizon_end_weight: float = 1.2,
) -> torch.Tensor:
    return forecast_loss(moe_residual, residual_target, criterion=criterion, horizon_end_weight=horizon_end_weight)


def residual_magnitude_loss(moe_residual: torch.Tensor) -> torch.Tensor:
    """Keep residual experts as small corrections around the base prediction."""

    return moe_residual.abs().mean()


def late_horizon_mse_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor, fraction: float = 0.35) -> torch.Tensor:
    """Extra supervision on the forecast tail where long-horizon drift appears."""

    horizon = int(pred_delta.shape[-1])
    if horizon <= 1:
        return F.mse_loss(pred_delta, target_delta)
    start = max(0, min(horizon - 1, int(round(horizon * (1.0 - float(fraction))))))
    return F.mse_loss(pred_delta[:, start:], target_delta[:, start:])


def terminal_mse_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    """MSE on the last predicted horizon step."""

    return F.mse_loss(pred_delta[:, -1], target_delta[:, -1])


def under_degradation_loss(pred_delta: torch.Tensor, target_delta: torch.Tensor, fraction: float = 0.35) -> torch.Tensor:
    """Penalize tail predictions that stay above the true SOH trajectory.

    In delta-SOH space, `pred_delta > target_delta` means the predicted SOH is
    too high, i.e. the model underestimates degradation. This term is optional
    and should stay small because it is intentionally asymmetric.
    """

    horizon = int(pred_delta.shape[-1])
    start = max(0, min(horizon - 1, int(round(horizon * (1.0 - float(fraction))))))
    return F.relu(pred_delta[:, start:] - target_delta[:, start:]).pow(2).mean()


def compute_base_model_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Dict[str, torch.Tensor]],
    loss_cfg: Dict[str, float | str],
) -> Dict[str, torch.Tensor]:
    target_delta = batch["query"]["target_delta_soh"]
    base_delta = outputs["base_delta"]
    anchor_soh = batch["query"].get("anchor_soh")
    if anchor_soh is None:
        anchor_soh = torch.zeros(base_delta.shape[0], device=base_delta.device, dtype=base_delta.dtype)
    base_soh = anchor_soh.unsqueeze(-1) + base_delta
    forecast = forecast_loss(
        base_delta,
        target_delta,
        criterion=str(loss_cfg.get("criterion", "weighted_mse")),
        horizon_end_weight=float(loss_cfg.get("horizon_end_weight", 1.2)),
    )
    pairwise = pairwise_aux_loss(
        outputs["pair_residual"],
        target_delta,
        batch["retrieval"]["ref_future_delta_soh"],
        batch["retrieval"]["retrieval_alpha"],
        batch["retrieval"]["retrieval_mask"],
    )
    mono = monotonic_loss(base_soh, epsilon=float(loss_cfg.get("monotonic_epsilon", 5e-4)))
    smooth = smoothness_loss(base_soh)
    retrieval_consistency = retrieval_consistency_loss(
        base_delta,
        outputs["rag_delta"],
        outputs["retrieval_confidence"],
    )
    total = (
        float(loss_cfg.get("forecast", 1.0)) * forecast
        + float(loss_cfg.get("pairwise", 0.5)) * pairwise
        + float(loss_cfg.get("monotonic", 0.05)) * mono
        + float(loss_cfg.get("smoothness", 0.01)) * smooth
        + float(loss_cfg.get("retrieval_consistency", 0.02)) * retrieval_consistency
    )
    return {
        "loss": total,
        "forecast_loss": forecast,
        "pairwise_aux_loss": pairwise,
        "monotonic_loss": mono,
        "smoothness_loss": smooth,
        "retrieval_consistency_loss": retrieval_consistency,
    }


def compute_residual_expert_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Dict[str, torch.Tensor]],
    loss_cfg: Dict[str, float | str],
) -> Dict[str, torch.Tensor]:
    target_delta = batch["query"]["target_delta_soh"]
    residual_target = batch["query"]["residual_target_oof"]
    pred_delta = outputs["pred_delta"]
    pred_soh = outputs["pred_soh"]
    residual_loss = residual_supervision_loss(
        outputs["moe_residual"],
        residual_target,
        criterion=str(loss_cfg.get("criterion", "weighted_mse")),
        horizon_end_weight=float(loss_cfg.get("horizon_end_weight", 1.2)),
    )
    final_forecast = forecast_loss(
        pred_delta,
        target_delta,
        criterion=str(loss_cfg.get("criterion", "weighted_mse")),
        horizon_end_weight=float(loss_cfg.get("horizon_end_weight", 1.2)),
    )
    mono = monotonic_loss(pred_soh, epsilon=float(loss_cfg.get("monotonic_epsilon", 5e-4)))
    smooth = smoothness_loss(pred_soh)
    residual_smooth = smoothness_loss(outputs["moe_residual"])
    residual_tv = total_variation_loss(outputs["moe_residual"])
    residual_mag = residual_magnitude_loss(outputs["moe_residual"])
    late_fraction = float(loss_cfg.get("late_horizon_fraction", 0.35))
    late_forecast = late_horizon_mse_loss(pred_delta, target_delta, fraction=late_fraction)
    terminal_forecast = terminal_mse_loss(pred_delta, target_delta)
    late_residual = late_horizon_mse_loss(outputs["moe_residual"], residual_target, fraction=late_fraction)
    under_degradation = under_degradation_loss(pred_delta, target_delta, fraction=late_fraction)
    expert_balance = expert_load_balance_loss(outputs["expert_weights"])
    route_kl = route_kl_loss(outputs["final_router_prior"], outputs["expert_weights"])
    total = (
        float(loss_cfg.get("forecast", 1.0)) * final_forecast
        + float(loss_cfg.get("residual", 1.0)) * residual_loss
        + float(loss_cfg.get("late_forecast", 0.0)) * late_forecast
        + float(loss_cfg.get("terminal_forecast", 0.0)) * terminal_forecast
        + float(loss_cfg.get("late_residual", 0.0)) * late_residual
        + float(loss_cfg.get("under_degradation", 0.0)) * under_degradation
        + float(loss_cfg.get("route_kl", loss_cfg.get("route", 0.01))) * route_kl
        + float(loss_cfg.get("monotonic", 0.05)) * mono
        + float(loss_cfg.get("smoothness", 0.01)) * smooth
        + float(loss_cfg.get("residual_smoothness", 0.0)) * residual_smooth
        + float(loss_cfg.get("residual_total_variation", 0.0)) * residual_tv
        + float(loss_cfg.get("residual_magnitude", 0.0)) * residual_mag
        + float(loss_cfg.get("expert_balance", 0.01)) * expert_balance
    )
    return {
        "loss": total,
        "residual_loss": residual_loss,
        "final_forecast_loss": final_forecast,
        "monotonic_loss": mono,
        "smoothness_loss": smooth,
        "residual_smoothness_loss": residual_smooth,
        "residual_total_variation_loss": residual_tv,
        "residual_magnitude_loss": residual_mag,
        "late_forecast_loss": late_forecast,
        "terminal_forecast_loss": terminal_forecast,
        "late_residual_loss": late_residual,
        "under_degradation_loss": under_degradation,
        "expert_load_balance_loss": expert_balance,
        "route_kl_loss": route_kl,
    }


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, Dict[str, torch.Tensor]],
    loss_cfg: Dict[str, float | str],
) -> Dict[str, torch.Tensor]:
    if "residual_target_oof" in batch["query"]:
        return compute_residual_expert_loss(outputs, batch, loss_cfg)
    return compute_base_model_loss(outputs, batch, loss_cfg)
