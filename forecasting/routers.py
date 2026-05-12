"""Interpretable routers for numerical battery SOH forecasting.

The main router is `SemanticHierarchicalRouter`: chemistry family is handled by
hard routing outside this module, while this module selects one of four
semantic residual-expert modes inside the selected chemistry branch.

No TSFM embedding, language model output, text prompt, or relaxation feature is
used by these routers.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import nn


CHEMISTRY_FAMILIES = ["LFP", "NCM", "NCA"]
SEMANTIC_EXPERT_MODES = [
    "high_temperature_expert",
    "high_current_expert",
    "high_cycle_expert",
    "high_power_expert",
]
SEMANTIC_CONCEPT_NAMES = [
    "concept_high_temperature",
    "concept_high_current",
    "concept_high_cycle_aging",
    "concept_high_power",
    "concept_low_retrieval_reliability",
]


class GroupedAdditiveRouter(nn.Module):
    """Legacy grouped additive router retained for tests and diagnostics."""

    def __init__(
        self,
        group_dims: Dict[str, int],
        num_outputs: int,
        top_k: int | None = None,
    ):
        super().__init__()
        self.group_dims = {name: int(dim) for name, dim in group_dims.items()}
        self.num_outputs = int(num_outputs)
        self.top_k = top_k
        self.group_layers = nn.ModuleDict(
            {
                name: nn.Linear(dim, self.num_outputs) if dim > 0 else nn.Identity()
                for name, dim in self.group_dims.items()
            }
        )
        self.bias = nn.Parameter(torch.zeros(self.num_outputs))

    def forward(self, group_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        contributions = {}
        for name, dim in self.group_dims.items():
            x = group_inputs[name]
            if dim <= 0:
                contributions[name] = torch.zeros(x.shape[0], self.num_outputs, device=x.device, dtype=x.dtype)
            else:
                contributions[name] = self.group_layers[name](x)
        logits = self.bias.unsqueeze(0)
        for value in contributions.values():
            logits = logits + value

        if self.top_k is not None and self.top_k < self.num_outputs:
            topk_values, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
            del topk_values
            topk_mask = torch.zeros_like(logits)
            topk_mask.scatter_(1, topk_indices, 1.0)
            masked_logits = logits.masked_fill(topk_mask == 0, float("-inf"))
            weights = torch.softmax(masked_logits, dim=-1)
        else:
            topk_mask = torch.ones_like(logits)
            weights = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "weights": weights,
            "topk_mask": topk_mask,
            "contributions": contributions,
        }


class PhysicalDegradationRouter(GroupedAdditiveRouter):
    """Compatibility router with the old grouped-additive interface."""

    REQUIRED_GROUPS = [
        "soh_state",
        "qv_polarization",
        "operation",
        "chemistry",
        "retrieval",
        "neighbor_vote",
    ]

    GROUP_DESCRIPTIONS = {
        "soh_state": "anchor_soh、recent_soh_slope、recent_soh_curvature，用于判断相对退化状态和趋势。",
        "qv_polarization": "放电 Q-V 窗口 dQ/dV 峰值、峰值电压、面积和容量跨度。",
        "operation": "温度、电流、功率/能量 proxy 和容量变化统计，用于表示运行压力。",
        "chemistry": "电池 chemistry metadata embedding，用于材料体系上下文。",
        "retrieval": "RAG confidence、top-k 距离统计和参考适配度，用于判断历史案例是否可靠。",
        "neighbor_vote": "top-k 邻居物理模式投票，用于把检索邻居的模式信息传给专家路由。",
    }

    def __init__(self, group_dims: Dict[str, int], num_experts: int, top_k_experts: int = 2):
        super().__init__(group_dims=group_dims, num_outputs=num_experts, top_k=top_k_experts)


class BranchFusionRouter(GroupedAdditiveRouter):
    """Router that fuses `fm_delta`, `rag_delta`, and `pair_delta`."""

    def __init__(self, group_dims: Dict[str, int], num_branches: int = 3):
        super().__init__(group_dims=group_dims, num_outputs=num_branches, top_k=None)


def _pad_or_trim(values: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Return a fixed-width feature tensor without changing semantics."""

    if values.shape[-1] == target_dim:
        return values
    if values.shape[-1] > target_dim:
        return values[..., :target_dim]
    pad = torch.zeros(*values.shape[:-1], target_dim - values.shape[-1], device=values.device, dtype=values.dtype)
    return torch.cat([values, pad], dim=-1)


def _prepare_expert_active_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    num_modes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Normalize an optional expert active mask to `[B, M]` boolean form."""

    if mask is None:
        return None
    active = _pad_or_trim(torch.nan_to_num(mask.to(device=device, dtype=torch.float32)), num_modes) > 0.5
    if active.ndim != 2 or active.shape[0] != batch_size:
        return None
    # Never allow an all-masked row; this keeps softmax numerically valid if a
    # caller provides only conditional experts.
    all_disabled = ~active.any(dim=-1, keepdim=True)
    return torch.where(all_disabled, torch.ones_like(active), active)


def _mask_logits(logits: torch.Tensor, active_mask: torch.Tensor | None) -> torch.Tensor:
    if active_mask is None:
        return logits
    if logits.ndim == 2:
        return logits.masked_fill(~active_mask, -1.0e9)
    return logits.masked_fill(~active_mask.unsqueeze(1), -1.0e9)


def _mask_distribution(distribution: torch.Tensor, active_mask: torch.Tensor | None) -> torch.Tensor:
    if active_mask is None:
        return distribution
    weight = active_mask.to(distribution.dtype)
    if distribution.ndim == 3:
        weight = weight.unsqueeze(1)
    masked = distribution * weight
    return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class SemanticConceptExtractor(nn.Module):
    """Build semantic concept scores from named physical/state features.

    Concepts are bounded in [0, 1] and are intentionally constructed from named
    interpretable inputs rather than opaque hidden states.
    """

    def __init__(self):
        super().__init__()
        self.concept_temperature = nn.Parameter(torch.ones(len(SEMANTIC_CONCEPT_NAMES)))

    def forward(
        self,
        *,
        soh_state: torch.Tensor,
        qv_polarization: torch.Tensor,
        partial_charge_shape: torch.Tensor,
        operation_stress: torch.Tensor,
        retrieval: torch.Tensor,
    ) -> torch.Tensor:
        soh_state = torch.nan_to_num(_pad_or_trim(soh_state.to(torch.float32), 3))
        qv = torch.nan_to_num(_pad_or_trim(qv_polarization.to(torch.float32), 12))
        operation = torch.nan_to_num(operation_stress.to(torch.float32))
        retrieval = torch.nan_to_num(_pad_or_trim(retrieval.to(torch.float32), 6))

        anchor_soh = soh_state[:, 0].clamp(0.0, 1.2)
        if operation.shape[-1] >= 24:
            hist_mean = operation[:, :8]
            hist_std = operation[:, 8:16]
            future_mean = operation[:, 16:24]
        else:
            hist_mean = _pad_or_trim(operation, 8)
            hist_std = torch.zeros_like(hist_mean)
            future_mean = torch.zeros_like(hist_mean)

        # Feature indices follow DEFAULT_OPERATION_FEATURES:
        # current_abs_mean/std/max, temp_mean/std/max, energy_charge_delta_1,
        # energy_discharge_delta_1.
        current_level = torch.maximum(hist_mean[:, 0].abs(), hist_mean[:, 2].abs())
        current_level = torch.maximum(current_level, qv[:, 9].abs() if qv.shape[-1] > 9 else qv[:, 7].abs())
        temp_level = torch.maximum(hist_mean[:, 3], hist_mean[:, 5])
        temp_level = torch.maximum(temp_level, qv[:, 6] if qv.shape[-1] > 6 else qv[:, 4])
        energy_level = hist_mean[:, 6].abs() + hist_mean[:, 7].abs()
        energy_level = torch.maximum(energy_level, qv[:, 10].abs() if qv.shape[-1] > 10 else torch.zeros_like(energy_level))

        has_temperature = temp_level.abs() > 1e-3
        has_current = current_level.abs() > 1e-6
        has_power = energy_level.abs() > 1e-8

        high_temperature = torch.where(has_temperature, torch.sigmoid((temp_level - 35.0) / 5.0), torch.zeros_like(temp_level))
        high_current = torch.where(has_current, torch.sigmoid((current_level - 2.0) / 0.75), torch.zeros_like(current_level))
        cycle_aging = torch.sigmoid((0.90 - anchor_soh) * 35.0 + qv[:, 11].clamp(0.0, 1.5) - 0.5)
        high_power = torch.where(has_power, torch.sigmoid((torch.log1p(energy_level.abs()) - 1.0) / 0.75), torch.zeros_like(energy_level))

        retrieval_confidence = retrieval[:, 0].clamp(0.0, 1.0)
        low_retrieval_reliability = (1.0 - retrieval_confidence).clamp(0.0, 1.0)

        concepts = torch.stack(
            [
                high_temperature,
                high_current,
                cycle_aging,
                high_power,
                low_retrieval_reliability,
            ],
            dim=-1,
        )
        return torch.sigmoid((concepts - 0.5) * self.concept_temperature.abs().unsqueeze(0) * 2.0)


class SemanticHierarchicalRouter(nn.Module):
    """Semantic prior plus small calibrator for 4 residual expert modes.

    Chemistry selection is hard-routed outside this module using known metadata.
    This router only chooses the semantic mode inside the selected chemistry
    branch: high-temperature, high-current, high-cycle, or high-power factor.
    """

    REQUIRED_GROUPS = [
        "soh_state",
        "qv_polarization",
        "partial_charge_shape",
        "operation_stress",
        "retrieval",
        "neighbor_vote",
    ]

    def __init__(
        self,
        group_dims: Dict[str, int],
        mode_names: Iterable[str] | None = None,
        gamma: float = 0.2,
        rag_eta: float = 0.5,
        horizon_context_dim: int = 6,
        horizon_prior_strength: float = 1.4,
        group_calibration_logit_clip: float = 0.75,
        total_calibration_logit_clip: float = 1.25,
        horizon_calibration_logit_clip: float = 0.35,
    ):
        super().__init__()
        self.mode_names = list(mode_names or SEMANTIC_EXPERT_MODES)
        self.num_modes = len(self.mode_names)
        self.group_dims = {name: int(dim) for name, dim in group_dims.items()}
        self.gamma = float(gamma)
        self.rag_eta = float(rag_eta)
        self.horizon_context_dim = int(horizon_context_dim)
        self.horizon_prior_strength = float(horizon_prior_strength)
        self.group_calibration_logit_clip = max(float(group_calibration_logit_clip), 0.0)
        self.total_calibration_logit_clip = max(float(total_calibration_logit_clip), 0.0)
        self.horizon_calibration_logit_clip = max(float(horizon_calibration_logit_clip), 0.0)
        self.concept_extractor = SemanticConceptExtractor()
        self.semantic_to_mode = nn.Linear(len(SEMANTIC_CONCEPT_NAMES), self.num_modes)
        self.group_calibrators = nn.ModuleDict(
            {
                name: nn.Linear(max(int(dim), 1), self.num_modes)
                for name, dim in self.group_dims.items()
            }
        )
        self.horizon_calibrator = nn.Linear(max(self.horizon_context_dim, 1), self.num_modes)
        self._init_semantic_prior()

    @staticmethod
    def _clip_logit_correction(values: torch.Tensor, clip_value: float) -> torch.Tensor:
        """Bound learned router corrections so they cannot override semantics.

        The router is designed as semantic prior plus a small learnable
        calibration term. Without a bound, one high-scale group can create
        extreme logits and collapse the expert weights to a single expert.
        """

        if clip_value <= 0:
            return values
        clip = torch.as_tensor(float(clip_value), device=values.device, dtype=values.dtype)
        return clip * torch.tanh(values / clip.clamp_min(1e-6))

    def _init_semantic_prior(self) -> None:
        with torch.no_grad():
            self.semantic_to_mode.weight.zero_()
            self.semantic_to_mode.bias.zero_()
            # Columns: high_temperature, high_current, high_cycle,
            # high_power, low_retrieval_reliability.
            template = torch.tensor(
                [
                    [2.2, -0.4, 0.1, 0.4, 0.0],
                    [-0.4, 2.2, 0.1, 0.5, 0.0],
                    [0.0, 0.0, 2.3, 0.2, 0.1],
                    [0.5, 0.5, 0.2, 2.2, 0.0],
                ],
                dtype=self.semantic_to_mode.weight.dtype,
            )
            rows = min(template.shape[0], self.semantic_to_mode.weight.shape[0])
            self.semantic_to_mode.weight[:rows, :] = template[:rows]
            for calibrator in self.group_calibrators.values():
                nn.init.zeros_(calibrator.weight)
                nn.init.zeros_(calibrator.bias)
            # Horizon-wise routing starts equivalent to the global semantic
            # router. Residual-expert training can then learn small step-wise
            # corrections without immediately breaking the semantic prior.
            nn.init.zeros_(self.horizon_calibrator.weight)
            nn.init.zeros_(self.horizon_calibrator.bias)

    def _horizon_semantic_logits(
        self,
        concepts: torch.Tensor,
        horizon_context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return horizon-wise semantic logit offsets for expert switching.

        The window-level semantic prior says which expert mode is plausible for
        the current battery state. This horizon term says how that plausibility
        should evolve as the forecast moves from near-term to far-term:

        - high-cycle and high-power factors can increase toward later horizon
          steps because accumulated stress matters more as the forecast extends;
        - high-temperature and high-current factors remain active throughout the
          horizon when their measured values are elevated;
        - low retrieval reliability keeps the router more conservative.

        It uses only named semantic concept scores and prediction-time horizon
        context. It does not access target future SOH.
        """

        if horizon_context is None:
            return None
        horizon_context = torch.nan_to_num(horizon_context.to(torch.float32))
        horizon_context = _pad_or_trim(horizon_context, self.horizon_context_dim)
        step_fraction = horizon_context[..., 0].clamp(0.0, 1.0)
        near = (1.0 - step_fraction).clamp(0.0, 1.0)
        far = step_fraction

        temperature_concept = concepts[:, 0].unsqueeze(1)
        current_concept = concepts[:, 1].unsqueeze(1)
        cycle_concept = concepts[:, 2].unsqueeze(1)
        power_concept = concepts[:, 3].unsqueeze(1)
        low_retrieval_concept = concepts[:, 4].unsqueeze(1)

        offsets = torch.zeros(
            horizon_context.shape[0],
            horizon_context.shape[1],
            self.num_modes,
            device=horizon_context.device,
            dtype=horizon_context.dtype,
        )
        mode_to_idx = {name: idx for idx, name in enumerate(self.mode_names)}
        if "high_temperature_expert" in mode_to_idx:
            offsets[..., mode_to_idx["high_temperature_expert"]] = 1.2 * temperature_concept + 0.2 * far * power_concept
        if "high_current_expert" in mode_to_idx:
            offsets[..., mode_to_idx["high_current_expert"]] = 1.2 * current_concept + 0.3 * far * power_concept
        if "high_cycle_expert" in mode_to_idx:
            offsets[..., mode_to_idx["high_cycle_expert"]] = 1.6 * far * cycle_concept + 0.4 * far * low_retrieval_concept
        if "high_power_expert" in mode_to_idx:
            offsets[..., mode_to_idx["high_power_expert"]] = 1.5 * far * power_concept + 0.4 * (temperature_concept + current_concept)

        # Remove per-step mean so this term changes relative expert preference
        # without globally sharpening or flattening the distribution.
        offsets = offsets - offsets.mean(dim=-1, keepdim=True)
        return self.horizon_prior_strength * offsets

    def semantic_prior_from_groups(self, group_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        concepts = self.concept_extractor(
            soh_state=group_inputs["soh_state"],
            qv_polarization=group_inputs["qv_polarization"],
            partial_charge_shape=group_inputs["partial_charge_shape"],
            operation_stress=group_inputs["operation_stress"],
            retrieval=group_inputs["retrieval"],
        )
        logits = self.semantic_to_mode(concepts)
        prior = torch.softmax(logits, dim=-1)
        return {"concepts": concepts, "semantic_logits": logits, "semantic_prior": prior}

    def _calibration_logits(self, group_inputs: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        contributions: Dict[str, torch.Tensor] = {}
        total = None
        for name, layer in self.group_calibrators.items():
            value = group_inputs.get(name)
            if value is None:
                batch = next(iter(group_inputs.values())).shape[0]
                value = torch.zeros(batch, 1, device=next(iter(group_inputs.values())).device)
            value = torch.nan_to_num(value.to(torch.float32))
            value = _pad_or_trim(value, layer.in_features)
            contribution = self.gamma * layer(value)
            contribution = self._clip_logit_correction(contribution, self.group_calibration_logit_clip)
            contributions[name] = contribution
            total = contribution if total is None else total + contribution
        if total is None:
            batch = next(iter(group_inputs.values())).shape[0]
            total = torch.zeros(batch, self.num_modes, device=next(iter(group_inputs.values())).device)
        return self._clip_logit_correction(total, self.total_calibration_logit_clip), contributions

    def forward(
        self,
        group_inputs: Dict[str, torch.Tensor],
        rag_semantic_prior: torch.Tensor | None = None,
        horizon_context: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        prior_payload = self.semantic_prior_from_groups(group_inputs)
        query_prior = prior_payload["semantic_prior"]
        active_mask = _prepare_expert_active_mask(
            group_inputs.get("expert_active_mask"),
            batch_size=query_prior.shape[0],
            num_modes=self.num_modes,
            device=query_prior.device,
        )
        retrieval_confidence = _pad_or_trim(group_inputs["retrieval"], 6)[:, 0].to(torch.float32).clamp(0.0, 1.0)
        if rag_semantic_prior is None:
            rag_semantic_prior = query_prior
        rag_semantic_prior = torch.nan_to_num(rag_semantic_prior.to(torch.float32), nan=0.25)
        rag_semantic_prior = rag_semantic_prior / rag_semantic_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        blend = (1.0 - self.rag_eta * retrieval_confidence).unsqueeze(-1) * query_prior
        blend = blend + (self.rag_eta * retrieval_confidence).unsqueeze(-1) * rag_semantic_prior
        final_prior = blend / blend.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        final_prior = _mask_distribution(final_prior, active_mask)
        horizon_semantic_offsets = self._horizon_semantic_logits(
            prior_payload["concepts"],
            horizon_context,
        )
        if horizon_semantic_offsets is None:
            final_prior_by_horizon = final_prior.unsqueeze(1)
        else:
            horizon_prior_logits = torch.log(final_prior.clamp_min(1e-6)).unsqueeze(1) + horizon_semantic_offsets
            final_prior_by_horizon = torch.softmax(horizon_prior_logits, dim=-1)
        final_prior_by_horizon = _mask_distribution(final_prior_by_horizon, active_mask)

        calibration, calibration_contributions = self._calibration_logits(group_inputs)
        logits = _mask_logits(torch.log(final_prior.clamp_min(1e-6)) + calibration, active_mask)
        if horizon_context is None:
            weights_by_horizon = torch.softmax(logits, dim=-1).unsqueeze(1)
            logits_by_horizon = logits.unsqueeze(1)
            horizon_contribution = torch.zeros_like(logits_by_horizon)
        else:
            horizon_context = torch.nan_to_num(horizon_context.to(torch.float32))
            horizon_context = _pad_or_trim(horizon_context, self.horizon_calibrator.in_features)
            horizon_contribution = self.gamma * self.horizon_calibrator(horizon_context)
            horizon_contribution = self._clip_logit_correction(
                horizon_contribution,
                self.horizon_calibration_logit_clip,
            )
            logits_by_horizon = _mask_logits(
                torch.log(final_prior_by_horizon.clamp_min(1e-6)) + calibration.unsqueeze(1) + horizon_contribution,
                active_mask,
            )
            weights_by_horizon = torch.softmax(logits_by_horizon, dim=-1)
        weights = weights_by_horizon.mean(dim=1)
        topk_mask = torch.ones_like(weights)
        contributions = {
            **calibration_contributions,
            "semantic_prior": torch.log(final_prior.clamp_min(1e-6)),
            "calibrator": calibration,
            "horizon_dynamic": horizon_contribution,
            "horizon_semantic_prior": horizon_semantic_offsets
            if horizon_semantic_offsets is not None
            else torch.zeros_like(logits_by_horizon),
        }
        return {
            "logits": logits,
            "logits_by_horizon": logits_by_horizon,
            "weights": weights,
            "weights_by_horizon": weights_by_horizon,
            "topk_mask": topk_mask,
            "contributions": contributions,
            "concept_scores": prior_payload["concepts"],
            "semantic_prior": query_prior,
            "rag_semantic_prior": rag_semantic_prior,
            "final_prior": final_prior,
            "final_prior_by_horizon": final_prior_by_horizon,
            "expert_active_mask": active_mask
            if active_mask is not None
            else torch.ones_like(weights, dtype=torch.bool),
        }


def semantic_mode_names() -> List[str]:
    return list(SEMANTIC_EXPERT_MODES)


def chemistry_family_names() -> List[str]:
    return list(CHEMISTRY_FAMILIES)
