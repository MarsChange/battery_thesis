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
    "slow_linear",
    "accelerating",
    "high_polarization",
    "curve_polarization_expert",
]
SEMANTIC_CONCEPT_NAMES = [
    "concept_slow_linear",
    "concept_accelerating",
    "concept_high_polarization",
    "concept_curve_polarization",
    "concept_high_operation_stress",
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
        "qv_polarization": "DeltaV(Q) 与 R(Q) 统计 proxy，用于表示极化和内阻相关状态。",
        "operation": "充放电倍率、温度统计和 protocol_change_rate，用于表示运行压力。",
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
        qv = torch.nan_to_num(_pad_or_trim(qv_polarization.to(torch.float32), 8))
        partial = torch.nan_to_num(_pad_or_trim(partial_charge_shape.to(torch.float32), 4))
        operation = torch.nan_to_num(operation_stress.to(torch.float32))
        retrieval = torch.nan_to_num(_pad_or_trim(retrieval.to(torch.float32), 6))

        slope = soh_state[:, 1]
        curvature = soh_state[:, 2]
        neg_slope = torch.sigmoid(-250.0 * slope)
        neg_curvature = torch.sigmoid(-1000.0 * curvature)
        stable_slope = torch.exp(-250.0 * slope.abs().clamp_max(0.05))
        stable_curvature = torch.exp(-1000.0 * curvature.abs().clamp_max(0.02))

        delta_v_std = qv[:, 1].abs()
        delta_v_q95 = qv[:, 2].abs()
        r_mean = qv[:, 3].abs()
        r_std = qv[:, 4].abs()
        r_q95 = qv[:, 5].abs()
        vc_slope = qv[:, 6].abs()
        vd_slope = qv[:, 7].abs()
        dq_dv_peak = partial[:, 3].abs()

        operation_level = operation.abs().mean(dim=-1)
        operation_volatility = operation.std(dim=-1, unbiased=False) if operation.shape[-1] > 1 else torch.zeros_like(operation_level)
        operation_stress_score = torch.sigmoid(0.6 * operation_level + 0.4 * operation_volatility)

        retrieval_confidence = retrieval[:, 0].clamp(0.0, 1.0)
        low_retrieval_reliability = (1.0 - retrieval_confidence).clamp(0.0, 1.0)

        slow_linear = (stable_slope * stable_curvature * torch.exp(-operation_stress_score)).clamp(0.0, 1.0)
        accelerating = torch.sigmoid(2.0 * neg_slope + 2.0 * neg_curvature + operation_volatility - 2.5)
        high_polarization = torch.sigmoid(1.2 * r_mean + 1.0 * r_std + 1.4 * r_q95 + 0.3 * operation_stress_score - 1.0)
        curve_polarization = torch.sigmoid(1.0 * delta_v_std + 1.2 * delta_v_q95 + 0.5 * vc_slope + 0.7 * vd_slope + 0.8 * dq_dv_peak - 1.0)

        concepts = torch.stack(
            [
                slow_linear,
                accelerating,
                high_polarization,
                curve_polarization,
                operation_stress_score,
                low_retrieval_reliability,
            ],
            dim=-1,
        )
        return torch.sigmoid((concepts - 0.5) * self.concept_temperature.abs().unsqueeze(0) * 2.0)


class SemanticHierarchicalRouter(nn.Module):
    """Semantic prior plus small calibrator for 4 residual expert modes.

    Chemistry selection is hard-routed outside this module using known metadata.
    This router only chooses the semantic mode inside the selected chemistry
    branch: slow-linear, accelerating, high-polarization, or curve-polarization.
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
    ):
        super().__init__()
        self.mode_names = list(mode_names or SEMANTIC_EXPERT_MODES)
        self.num_modes = len(self.mode_names)
        self.group_dims = {name: int(dim) for name, dim in group_dims.items()}
        self.gamma = float(gamma)
        self.rag_eta = float(rag_eta)
        self.concept_extractor = SemanticConceptExtractor()
        self.semantic_to_mode = nn.Linear(len(SEMANTIC_CONCEPT_NAMES), self.num_modes)
        self.group_calibrators = nn.ModuleDict(
            {
                name: nn.Linear(max(int(dim), 1), self.num_modes)
                for name, dim in self.group_dims.items()
            }
        )
        self._init_semantic_prior()

    def _init_semantic_prior(self) -> None:
        with torch.no_grad():
            self.semantic_to_mode.weight.zero_()
            self.semantic_to_mode.bias.zero_()
            # Columns: slow_linear, accelerating, high_polarization,
            # curve_polarization, operation_stress, low_retrieval_reliability.
            template = torch.tensor(
                [
                    [2.0, -0.8, -0.6, -0.6, -0.4, 0.0],
                    [-0.8, 2.2, 0.2, 0.2, 0.4, 0.0],
                    [-0.6, 0.2, 2.2, 0.6, 0.5, 0.0],
                    [-0.6, 0.2, 0.6, 2.2, 0.2, 0.0],
                ],
                dtype=self.semantic_to_mode.weight.dtype,
            )
            rows = min(template.shape[0], self.semantic_to_mode.weight.shape[0])
            self.semantic_to_mode.weight[:rows, :] = template[:rows]
            for calibrator in self.group_calibrators.values():
                nn.init.zeros_(calibrator.weight)
                nn.init.zeros_(calibrator.bias)

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
            contribution = layer(value)
            contributions[name] = self.gamma * contribution
            total = contribution if total is None else total + contribution
        if total is None:
            batch = next(iter(group_inputs.values())).shape[0]
            total = torch.zeros(batch, self.num_modes, device=next(iter(group_inputs.values())).device)
        return self.gamma * total, contributions

    def forward(
        self,
        group_inputs: Dict[str, torch.Tensor],
        rag_semantic_prior: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        prior_payload = self.semantic_prior_from_groups(group_inputs)
        query_prior = prior_payload["semantic_prior"]
        retrieval_confidence = _pad_or_trim(group_inputs["retrieval"], 6)[:, 0].to(torch.float32).clamp(0.0, 1.0)
        if rag_semantic_prior is None:
            rag_semantic_prior = query_prior
        rag_semantic_prior = torch.nan_to_num(rag_semantic_prior.to(torch.float32), nan=0.25)
        rag_semantic_prior = rag_semantic_prior / rag_semantic_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        blend = (1.0 - self.rag_eta * retrieval_confidence).unsqueeze(-1) * query_prior
        blend = blend + (self.rag_eta * retrieval_confidence).unsqueeze(-1) * rag_semantic_prior
        final_prior = blend / blend.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        calibration, calibration_contributions = self._calibration_logits(group_inputs)
        logits = torch.log(final_prior.clamp_min(1e-6)) + calibration
        weights = torch.softmax(logits, dim=-1)
        topk_mask = torch.ones_like(weights)
        contributions = {
            **calibration_contributions,
            "semantic_prior": torch.log(final_prior.clamp_min(1e-6)),
            "calibrator": calibration,
        }
        return {
            "logits": logits,
            "weights": weights,
            "topk_mask": topk_mask,
            "contributions": contributions,
            "concept_scores": prior_payload["concepts"],
            "semantic_prior": query_prior,
            "rag_semantic_prior": rag_semantic_prior,
            "final_prior": final_prior,
        }


def semantic_mode_names() -> List[str]:
    return list(SEMANTIC_EXPERT_MODES)


def chemistry_family_names() -> List[str]:
    return list(CHEMISTRY_FAMILIES)

