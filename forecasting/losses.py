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


def expert_load_balance_loss(expert_weights: torch.Tensor) -> torch.Tensor:
    if expert_weights.numel() == 0:
        return expert_weights.new_tensor(0.0)
    weights = expert_weights.reshape(-1, expert_weights.shape[-1])
    mean_weight = weights.mean(dim=0)
    uniform = torch.full_like(mean_weight, 1.0 / max(expert_weights.size(-1), 1))
    return F.kl_div((mean_weight + 1e-8).log(), uniform, reduction="batchmean")


def route_kl_loss(final_prior: torch.Tensor, expert_weights: torch.Tensor) -> torch.Tensor:
    """KL(stopgrad semantic prior || final expert weights)."""

    prior = final_prior.detach().clamp_min(1e-8)
    weights = expert_weights.clamp_min(1e-8)
    if weights.ndim == 3 and prior.ndim == 2:
        prior = prior.unsqueeze(1).expand_as(weights)
    if weights.ndim > 2:
        weights = weights.reshape(-1, weights.shape[-1])
        prior = prior.reshape(-1, prior.shape[-1])
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


def residual_direction_loss(residual_direction: torch.Tensor, residual_target: torch.Tensor) -> torch.Tensor:
    """Supervise whether residual should lift or lower the base forecast."""

    target_sign = torch.sign(residual_target)
    importance = residual_target.abs()
    importance = importance / importance.detach().mean().clamp_min(1e-6)
    return (F.smooth_l1_loss(residual_direction, target_sign, reduction="none") * importance).mean()


def residual_sign_error_loss(moe_residual: torch.Tensor, residual_target: torch.Tensor) -> torch.Tensor:
    """Penalize residual corrections that point opposite to the OOF target."""

    importance = residual_target.abs()
    importance = importance / importance.detach().mean().clamp_min(1e-6)
    return (F.relu(-moe_residual * residual_target) * importance).mean()


def factor_weighted_magnitude_loss(
    selected_mode_magnitudes: torch.Tensor,
    residual_target: torch.Tensor,
    factor_scores: torch.Tensor,
) -> torch.Tensor:
    """Train each factor expert on residual magnitude where its factor is active.

    `factor_scores` columns correspond to high-temperature, high-current,
    high-cycle-aging, and high-power experts. This is a soft split: every
    sample can contribute to multiple experts, but high-score samples dominate
    the matching expert's magnitude supervision.
    """

    if selected_mode_magnitudes.ndim != 3 or factor_scores.ndim != 2:
        return residual_target.new_tensor(0.0)
    num_modes = selected_mode_magnitudes.shape[1]
    factor_scores = factor_scores[:, :num_modes].to(selected_mode_magnitudes.dtype).clamp(0.0, 1.0)
    if factor_scores.numel() == 0:
        return residual_target.new_tensor(0.0)
    weights = factor_scores.unsqueeze(-1)
    target_magnitude = residual_target.abs().unsqueeze(1)
    loss = F.smooth_l1_loss(selected_mode_magnitudes, target_magnitude.expand_as(selected_mode_magnitudes), reduction="none")
    denom = weights.sum().clamp_min(1.0) * residual_target.shape[-1]
    return (loss * weights).sum() / denom


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
    if "residual_direction" in outputs:
        direction_loss = residual_direction_loss(outputs["residual_direction"], residual_target)
    else:
        direction_loss = residual_target.new_tensor(0.0)
    sign_loss = residual_sign_error_loss(outputs["moe_residual"], residual_target)
    if "selected_mode_magnitudes" in outputs and "residual_factor_scores" in batch["query"]:
        factor_magnitude = factor_weighted_magnitude_loss(
            outputs["selected_mode_magnitudes"],
            residual_target,
            batch["query"]["residual_factor_scores"],
        )
    else:
        factor_magnitude = residual_target.new_tensor(0.0)
    if "expert_weights_by_horizon" in outputs:
        horizon_expert_weights = outputs["expert_weights_by_horizon"]
    else:
        horizon_expert_weights = outputs["expert_weights"]
    if "final_router_prior_by_horizon" in outputs:
        horizon_router_prior = outputs["final_router_prior_by_horizon"]
    else:
        horizon_router_prior = outputs["final_router_prior"]
    expert_balance = expert_load_balance_loss(horizon_expert_weights)
    route_kl = route_kl_loss(horizon_router_prior, horizon_expert_weights)
    # Residual expert training is intentionally kept compact:
    # 1) final forecast accuracy, 2) OOF residual target matching,
    # 3) residual direction/sign supervision, 4) factor-specific magnitude
    # supervision, 5) lightweight trajectory/router regularization.
    total = (
        float(loss_cfg.get("forecast", 1.0)) * final_forecast
        + float(loss_cfg.get("residual", 1.0)) * residual_loss
        + float(loss_cfg.get("route_kl", loss_cfg.get("route", 0.01))) * route_kl
        + float(loss_cfg.get("monotonic", 0.05)) * mono
        + float(loss_cfg.get("smoothness", 0.01)) * smooth
        + float(loss_cfg.get("residual_direction", 0.0)) * direction_loss
        + float(loss_cfg.get("residual_sign", 0.0)) * sign_loss
        + float(loss_cfg.get("factor_magnitude", 0.0)) * factor_magnitude
        + float(loss_cfg.get("expert_balance", 0.01)) * expert_balance
    )
    return {
        "loss": total,
        "residual_loss": residual_loss,
        "final_forecast_loss": final_forecast,
        "monotonic_loss": mono,
        "smoothness_loss": smooth,
        "residual_direction_loss": direction_loss,
        "residual_sign_loss": sign_loss,
        "factor_magnitude_loss": factor_magnitude,
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
