"""Numerical multi-step SOH forecaster.

The model is intentionally free of external time-series foundation models. It
uses named battery signals, RAG references, a reference-conditioned pairwise
branch, and LSTM specialists selected by an interpretable physical router.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from forecasting.routers import (
    CHEMISTRY_FAMILIES,
    SEMANTIC_EXPERT_MODES,
    BranchFusionRouter,
    SemanticHierarchicalRouter,
)
from retrieval.multistage_retriever import COMPONENT_NAMES


DEFAULT_CHEMISTRY_FAMILIES = list(CHEMISTRY_FAMILIES)
DEFAULT_EXPERT_NAMES = list(SEMANTIC_EXPERT_MODES)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    summed = (values * weight).sum(dim=dim)
    denom = weight.sum(dim=dim).clamp_min(1e-6)
    return summed / denom


def _fixed_bin_mean_1d(values: torch.Tensor, bins: int) -> torch.Tensor:
    """MPS-safe 1-D mean pooling into a fixed number of bins."""

    length = int(values.shape[-1])
    pooled = []
    for bin_idx in range(int(bins)):
        start = int(round(bin_idx * length / bins))
        end = int(round((bin_idx + 1) * length / bins))
        if end <= start:
            end = min(length, start + 1)
        pooled.append(values[..., start:end].mean(dim=-1))
    return torch.stack(pooled, dim=-1)


def _fixed_bin_mean_2d(values: torch.Tensor, width_bins: int) -> torch.Tensor:
    """MPS-safe 2-D pooling over curve channels and fixed width bins."""

    values = values.mean(dim=-2)
    return _fixed_bin_mean_1d(values, width_bins)


class CycleStatsEncoder(nn.Module):
    """GRU encoder over historical cycle statistics and SOH sequence."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cycle_stats: torch.Tensor, soh_seq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cycle_stats, soh_seq], dim=-1)
        _, hidden = self.gru(x)
        return self.dropout(hidden[-1])


class QVMapEncoder(nn.Module):
    """Small CNN encoder for Q-indexed voltage/current maps."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
        )
        self.proj = nn.Linear(16 * 4, hidden_dim)

    def forward(self, qv_maps: torch.Tensor, qv_masks: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, width = qv_maps.shape
        conv = self.net(qv_maps.reshape(batch * steps, 1, channels, width))
        feat = _fixed_bin_mean_2d(conv, 4).reshape(batch, steps, -1)
        feat = self.proj(feat)
        cycle_mask = (qv_masks.sum(dim=-1) > 0).to(qv_maps.dtype)
        return _masked_mean(feat, cycle_mask, dim=1)


class SequenceCurveEncoder(nn.Module):
    """1-D CNN encoder for optional partial-charge curves."""

    def __init__(self, length: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.proj = nn.Linear(16 * 4, hidden_dim)

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, steps, length = values.shape
        conv = self.net(values.reshape(batch * steps, 1, length))
        feat = _fixed_bin_mean_1d(conv, 4).reshape(batch, steps, -1)
        feat = self.proj(feat)
        return _masked_mean(feat, masks, dim=1)


class FutureOperationEncoder(nn.Module):
    """GRU encoder for known future operation features, if provided."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, future_ops: torch.Tensor, future_ops_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden = self.gru(future_ops * future_ops_mask)
        return hidden[-1], outputs


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualLSTMExpert(nn.Module):
    """LSTM specialist that predicts only a residual correction.

    输入：
    - expert_seq: SOH、Q-V proxy、DeltaV/R 和工况组成的窗口级数值序列。
    - expert_context: 可解释全局上下文，包括 SOH 状态、物理 proxy、metadata、检索置信度和 baseline summary。
    - baseline_delta_detached: 当前基础预测，使用 detach 防止专家反向影响基础分支。

    输出：
    - residual correction，最终 `pred_delta = base_delta + moe_residual`。
    """

    def __init__(
        self,
        expert_seq_dim: int,
        expert_context_dim: int,
        horizon: int,
        future_operation_dim: int,
        hidden_dim: int,
        dropout: float,
        residual_decoder_type: str = "direct",
        residual_basis_points: int = 8,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.future_operation_dim = int(future_operation_dim)
        self.residual_decoder_type = str(residual_decoder_type)
        self.residual_basis_points = max(2, min(int(residual_basis_points), self.horizon))
        self.input_proj = nn.Linear(expert_seq_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.context_proj = nn.Sequential(nn.Linear(expert_context_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.future_gru = nn.GRU(future_operation_dim, hidden_dim, batch_first=True) if future_operation_dim > 0 else None
        decoder_output_dim = self.residual_basis_points if self.residual_decoder_type == "smooth_basis" else self.horizon
        self.decoder = MLPHead(hidden_dim * 3 + self.horizon, decoder_output_dim, hidden_dim, dropout)

    def _decode_residual(self, decoder_input: torch.Tensor) -> torch.Tensor:
        raw = self.decoder(decoder_input)
        if self.residual_decoder_type != "smooth_basis":
            return raw
        if raw.shape[-1] == self.horizon:
            return raw
        # Long-horizon residuals should be smooth corrections, not independent
        # high-frequency point estimates. The expert predicts a small number of
        # semantic trend control points and linearly interpolates them to the
        # forecast horizon.
        return F.interpolate(raw.unsqueeze(1), size=self.horizon, mode="linear", align_corners=True).squeeze(1)

    def forward(
        self,
        expert_seq: torch.Tensor,
        expert_context: torch.Tensor,
        baseline_delta_detached: torch.Tensor,
        future_ops: torch.Tensor | None = None,
        future_ops_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_hidden = self.input_proj(expert_seq)
        _, (last_hidden, _) = self.lstm(seq_hidden)
        seq_summary = last_hidden[-1]
        context_summary = self.context_proj(expert_context)
        if self.future_gru is not None and future_ops is not None and future_ops_mask is not None:
            _, future_hidden = self.future_gru(future_ops * future_ops_mask)
            future_summary = future_hidden[-1]
        else:
            future_summary = torch.zeros_like(seq_summary)
        decoder_input = torch.cat([seq_summary, context_summary, future_summary, baseline_delta_detached], dim=-1)
        return self._decode_residual(decoder_input)


class BatterySOHForecaster(nn.Module):
    """Purely numerical battery SOH forecaster."""

    def __init__(
        self,
        horizon: int,
        cycle_feature_dim: int,
        physics_dim: int,
        operation_dim: int,
        future_operation_dim: int,
        meta_dim: int,
        qv_width: int = 100,
        partial_charge_points: int = 50,
        expert_seq_dim: int = 14,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        meta_embedding_dim: int = 16,
        expert_names: List[str] | None = None,
        chemistry_families: List[str] | None = None,
        top_k_experts: int = 2,
        residual_decoder_type: str = "direct",
        residual_basis_points: int = 8,
        residual_max_abs: float = 0.006,
        residual_confidence_floor: float = 0.35,
        base_trend_blend: float = 0.60,
        residual_stage_floor: float = 0.25,
        residual_stage_mid_soh: float = 0.98,
        residual_stage_sharpness: float = 60.0,
    ):
        super().__init__()
        del qv_width
        self.horizon = int(horizon)
        self.cycle_feature_dim = int(cycle_feature_dim)
        self.physics_dim = int(physics_dim)
        self.operation_dim = int(operation_dim)
        self.future_operation_dim = int(future_operation_dim)
        self.meta_dim = int(meta_dim)
        self.expert_seq_dim = int(expert_seq_dim)
        self.residual_decoder_type = str(residual_decoder_type)
        self.residual_basis_points = int(residual_basis_points)
        self.residual_max_abs = float(residual_max_abs)
        self.residual_confidence_floor = float(residual_confidence_floor)
        self.base_trend_blend = float(base_trend_blend)
        self.residual_stage_floor = float(residual_stage_floor)
        self.residual_stage_mid_soh = float(residual_stage_mid_soh)
        self.residual_stage_sharpness = float(residual_stage_sharpness)
        requested_experts = list(expert_names or DEFAULT_EXPERT_NAMES)
        self.expert_names = [name for name in requested_experts if name in SEMANTIC_EXPERT_MODES]
        if not self.expert_names:
            self.expert_names = list(DEFAULT_EXPERT_NAMES)
        self.chemistry_families = list(chemistry_families or DEFAULT_CHEMISTRY_FAMILIES)
        self.chemistry_families = [name for name in self.chemistry_families if name in CHEMISTRY_FAMILIES] or list(DEFAULT_CHEMISTRY_FAMILIES)
        self.pairwise_uses_raw_external_embedding = False
        self.experts_use_raw_external_embedding = False

        self.cycle_encoder = CycleStatsEncoder(self.cycle_feature_dim + 1, hidden_dim, dropout)
        self.qv_encoder = QVMapEncoder(hidden_dim)
        self.partial_charge_encoder = SequenceCurveEncoder(partial_charge_points, hidden_dim)
        self.future_op_encoder = FutureOperationEncoder(self.future_operation_dim, hidden_dim)
        self.meta_embedding = nn.Embedding(512, meta_embedding_dim)

        state_dim = 3
        pair_repr_dim = hidden_dim * 4 + meta_embedding_dim + state_dim + physics_dim
        pair_input_dim = pair_repr_dim * 4 + self.horizon + len(COMPONENT_NAMES) + 1
        expert_context_dim = 3 + physics_dim + meta_embedding_dim + 5 + 3 + self.future_operation_dim

        self.generalist_head = MLPHead(hidden_dim * 4 + meta_embedding_dim + state_dim + physics_dim, self.horizon, hidden_dim, dropout)
        self.pairwise_branch = MLPHead(pair_input_dim, self.horizon, hidden_dim, dropout)
        self.expert_bank = nn.ModuleDict(
            {
                chemistry: nn.ModuleDict(
                    {
                        mode_name: ResidualLSTMExpert(
                            expert_seq_dim=self.expert_seq_dim,
                            expert_context_dim=expert_context_dim,
                            horizon=self.horizon,
                            future_operation_dim=self.future_operation_dim,
                            hidden_dim=hidden_dim,
                            dropout=dropout,
                            residual_decoder_type=self.residual_decoder_type,
                            residual_basis_points=self.residual_basis_points,
                        )
                        for mode_name in self.expert_names
                    }
                )
                for chemistry in self.chemistry_families
            }
        )

        self.semantic_router = SemanticHierarchicalRouter(
            group_dims={
                "soh_state": 3,
                "qv_polarization": min(8, physics_dim),
                "partial_charge_shape": max(physics_dim - 8, 0),
                "operation_stress": operation_dim * 2 + future_operation_dim,
                "retrieval": 6,
                "neighbor_vote": len(self.expert_names),
            },
            mode_names=self.expert_names,
            gamma=0.2,
            rag_eta=0.5,
        )
        # Compatibility alias for older training utilities/tests. This is the
        # semantic router, not the old opaque hidden-state router.
        self.physical_router = self.semantic_router
        self.base_fusion_router = BranchFusionRouter(
            group_dims={
                "retrieval": 4,
                "state": 3,
                "feature_availability": 2,
                "chemistry": meta_embedding_dim,
            },
            num_branches=3,
        )

    def _meta_encode(self, meta_ids: torch.Tensor) -> torch.Tensor:
        return self.meta_embedding(meta_ids.long().clamp(0, 511)).mean(dim=1)

    def _state_features(self, anchor_soh: torch.Tensor, soh_seq: torch.Tensor) -> torch.Tensor:
        seq = soh_seq.squeeze(-1)
        slope = seq[:, -1] - seq[:, -2] if seq.size(1) >= 2 else torch.zeros_like(anchor_soh)
        curvature = seq[:, -1] - 2 * seq[:, -2] + seq[:, -3] if seq.size(1) >= 3 else torch.zeros_like(anchor_soh)
        return torch.stack([anchor_soh, slope, curvature], dim=-1)

    def _linear_trend_delta(self, soh_seq: torch.Tensor, horizon: int) -> torch.Tensor:
        """Deterministic local SOH trend prior from the observed history.

        This is not a separate expert. It is a conservative anchor for the base
        prediction when retrieved references are weak. The slope is estimated
        by least squares over the most recent history window to avoid using only
        one noisy last-cycle difference.
        """

        seq = soh_seq.squeeze(-1).to(torch.float32)
        window = min(int(seq.shape[1]), 16)
        y = seq[:, -window:]
        x = torch.arange(window, device=seq.device, dtype=seq.dtype)
        x = x - x.mean()
        y_centered = y - y.mean(dim=1, keepdim=True)
        denom = (x.pow(2).sum()).clamp_min(1e-6)
        slope = (y_centered * x.unsqueeze(0)).sum(dim=1) / denom
        steps = torch.arange(1, int(horizon) + 1, device=seq.device, dtype=seq.dtype)
        return slope.unsqueeze(-1) * steps.unsqueeze(0)

    def _feature_availability(self, qv_masks: torch.Tensor, physics_mask: torch.Tensor, future_ops_mask: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                qv_masks.to(torch.float32).mean(dim=(1, 2)),
                physics_mask.to(torch.float32).mean(dim=(1, 2)),
                future_ops_mask.to(torch.float32).mean(dim=(1, 2)),
            ],
            dim=-1,
        )

    def _operation_group_features(self, operation_seq: torch.Tensor, future_ops: torch.Tensor) -> torch.Tensor:
        return torch.cat([operation_seq.mean(dim=1), operation_seq.std(dim=1), future_ops.mean(dim=1)], dim=-1)

    def _neighbor_vote(self, retrieval: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if "neighbor_physics_mode_vote" in retrieval:
            vote = retrieval["neighbor_physics_mode_vote"].to(torch.float32)
            if vote.shape[-1] == len(self.expert_names):
                return vote
        return torch.zeros(batch_size, len(self.expert_names), device=device)

    def _chemistry_ids(self, query: Dict[str, torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if "chemistry_id" in query:
            ids = query["chemistry_id"].to(device).long().reshape(-1)
            return ids.clamp(0, len(self.chemistry_families) - 1)
        return torch.zeros(batch_size, device=device, dtype=torch.long)

    def _fallback_expert_seq(self, query: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq = torch.cat(
            [
                query["soh_seq"].to(torch.float32),
                query["cycle_stats"].to(torch.float32)[..., : min(5, query["cycle_stats"].shape[-1])],
                query["physics_features"].to(torch.float32)[..., : min(4, query["physics_features"].shape[-1])],
                query["operation_seq"].to(torch.float32)[..., : min(4, query["operation_seq"].shape[-1])],
            ],
            dim=-1,
        )
        if seq.shape[-1] < self.expert_seq_dim:
            pad = torch.zeros(seq.shape[0], seq.shape[1], self.expert_seq_dim - seq.shape[-1], device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=-1)
        return seq[..., : self.expert_seq_dim]

    def _encode_modalities(
        self,
        cycle_stats: torch.Tensor,
        soh_seq: torch.Tensor,
        qv_maps: torch.Tensor,
        qv_masks: torch.Tensor,
        partial_charge: torch.Tensor,
        partial_charge_mask: torch.Tensor,
        future_ops: torch.Tensor,
        future_ops_mask: torch.Tensor,
        anchor_physics_features: torch.Tensor,
        meta_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_cycle = self.cycle_encoder(cycle_stats, soh_seq)
        h_qv = self.qv_encoder(qv_maps, qv_masks)
        h_pc = self.partial_charge_encoder(partial_charge, partial_charge_mask)
        h_future_ops, future_horizon_context = self.future_op_encoder(future_ops, future_ops_mask)
        meta_embedding = self._meta_encode(meta_ids)
        return {
            "h_cycle": h_cycle,
            "h_qv": h_qv,
            "h_pc": h_pc,
            "h_future_ops": h_future_ops,
            "future_horizon_context": future_horizon_context,
            "meta_embedding": meta_embedding,
            "anchor_physics_features": anchor_physics_features,
        }

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        query = batch["query"]
        retrieval = batch["retrieval"]
        device = query["cycle_stats"].device
        anchor_soh = query["anchor_soh"].to(torch.float32)

        query_encoded = self._encode_modalities(
            cycle_stats=query["cycle_stats"].to(torch.float32),
            soh_seq=query["soh_seq"].to(torch.float32),
            qv_maps=query["qv_maps"].to(torch.float32),
            qv_masks=query["qv_masks"].to(torch.float32),
            partial_charge=query["partial_charge"].to(torch.float32),
            partial_charge_mask=query["partial_charge_mask"].to(torch.float32),
            future_ops=query["future_ops"].to(torch.float32),
            future_ops_mask=query["future_ops_mask"].to(torch.float32),
            anchor_physics_features=query["anchor_physics_features"].to(torch.float32),
            meta_ids=query["meta_ids"].to(device),
        )

        state_features = self._state_features(anchor_soh, query["soh_seq"].to(torch.float32))
        availability = self._feature_availability(
            query["qv_masks"].to(torch.float32),
            query["physics_feature_masks"].to(torch.float32),
            query["future_ops_mask"].to(torch.float32),
        )
        operation_group = self._operation_group_features(query["operation_seq"].to(torch.float32), query["future_ops"].to(torch.float32))

        retrieval_alpha = retrieval["retrieval_alpha"].to(torch.float32)
        retrieval_mask = retrieval["retrieval_mask"].to(torch.float32)
        ref_future_delta = retrieval["ref_future_delta_soh"].to(torch.float32)
        rag_delta = torch.sum(ref_future_delta * retrieval_alpha.unsqueeze(-1), dim=1)
        composite_distance = retrieval["composite_distance"].to(torch.float32)
        compatibility = retrieval["reference_compatibility_score"].to(torch.float32)
        masked_composite = composite_distance.masked_fill(retrieval_mask <= 0, 0.0)
        valid_counts = retrieval_mask.sum(dim=1).clamp_min(1.0)
        composite_mean = masked_composite.sum(dim=1) / valid_counts
        composite_var = ((masked_composite - composite_mean.unsqueeze(-1)) ** 2 * retrieval_mask).sum(dim=1) / valid_counts
        composite_std = torch.sqrt(composite_var.clamp_min(0.0))
        compatibility_mean = (compatibility * retrieval_mask).sum(dim=1) / valid_counts
        top1_distance = composite_distance[:, 0].masked_fill(retrieval_mask[:, 0] <= 0, 0.0)
        chemistry_ids = self._chemistry_ids(query, anchor_soh.shape[0], device)
        if "ref_chemistry_id" in retrieval:
            ref_chemistry_id = retrieval["ref_chemistry_id"].to(device).long()
            chemistry_match_ratio = ((ref_chemistry_id == chemistry_ids[:, None]).to(torch.float32) * retrieval_alpha * retrieval_mask).sum(dim=1) / valid_counts
        else:
            chemistry_match_ratio = torch.zeros_like(composite_mean)

        fm_input = torch.cat(
            [
                query_encoded["h_cycle"],
                query_encoded["h_qv"],
                query_encoded["h_pc"],
                query_encoded["h_future_ops"],
                query_encoded["meta_embedding"],
                state_features,
                query_encoded["anchor_physics_features"],
            ],
            dim=-1,
        )
        fm_delta = self.generalist_head(fm_input)

        batch_size, top_k = ref_future_delta.shape[:2]
        ref_encoded = self._encode_modalities(
            cycle_stats=retrieval["ref_cycle_stats"].reshape(batch_size * top_k, *retrieval["ref_cycle_stats"].shape[2:]).to(torch.float32),
            soh_seq=retrieval["ref_soh_seq"].reshape(batch_size * top_k, *retrieval["ref_soh_seq"].shape[2:]).to(torch.float32),
            qv_maps=retrieval["ref_qv_maps"].reshape(batch_size * top_k, *retrieval["ref_qv_maps"].shape[2:]).to(torch.float32),
            qv_masks=(retrieval["ref_qv_maps"].abs().sum(dim=-1) > 0).reshape(batch_size * top_k, *retrieval["ref_qv_maps"].shape[2:4]).to(torch.float32),
            partial_charge=retrieval["ref_partial_charge"].reshape(batch_size * top_k, *retrieval["ref_partial_charge"].shape[2:]).to(torch.float32),
            partial_charge_mask=(retrieval["ref_partial_charge"].abs().sum(dim=-1) > 0).reshape(batch_size * top_k, retrieval["ref_partial_charge"].shape[2]).to(torch.float32),
            future_ops=query["future_ops"].repeat_interleave(top_k, dim=0).to(torch.float32),
            future_ops_mask=query["future_ops_mask"].repeat_interleave(top_k, dim=0).to(torch.float32),
            anchor_physics_features=retrieval["ref_anchor_physics_features"].reshape(batch_size * top_k, -1).to(torch.float32),
            meta_ids=retrieval["ref_meta_ids"].reshape(batch_size * top_k, -1).to(device),
        )
        query_repr = torch.cat(
            [
                query_encoded["h_cycle"],
                query_encoded["h_qv"],
                query_encoded["h_pc"],
                query_encoded["h_future_ops"],
                query_encoded["meta_embedding"],
                state_features,
                query_encoded["anchor_physics_features"],
            ],
            dim=-1,
        )
        ref_repr = torch.cat(
            [
                ref_encoded["h_cycle"],
                ref_encoded["h_qv"],
                ref_encoded["h_pc"],
                ref_encoded["h_future_ops"],
                ref_encoded["meta_embedding"],
                self._state_features(
                    retrieval["ref_anchor_soh"].reshape(batch_size * top_k).to(torch.float32),
                    retrieval["ref_soh_seq"].reshape(batch_size * top_k, *retrieval["ref_soh_seq"].shape[2:]).to(torch.float32),
                ),
                ref_encoded["anchor_physics_features"],
            ],
            dim=-1,
        ).view(batch_size, top_k, -1)
        query_pair = query_repr[:, None, :].expand(-1, top_k, -1)
        pair_features = torch.cat(
            [
                query_pair,
                ref_repr,
                query_pair - ref_repr,
                query_pair * ref_repr,
                ref_future_delta,
                torch.nan_to_num(retrieval["component_distances"].to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0),
                compatibility.unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_residual = self.pairwise_branch(pair_features)
        pair_delta = torch.sum((ref_future_delta + pair_residual) * retrieval_alpha.unsqueeze(-1), dim=1)

        base_fusion_inputs = {
            "retrieval": torch.stack([retrieval["retrieval_confidence"].to(torch.float32), composite_mean, composite_std, compatibility_mean], dim=-1),
            "state": state_features,
            "feature_availability": availability[:, :2],
            "chemistry": query_encoded["meta_embedding"],
        }
        fusion_out = self.base_fusion_router(base_fusion_inputs)
        base_stack = torch.stack([fm_delta, rag_delta, pair_delta], dim=1)
        base_delta_raw = torch.sum(base_stack * fusion_out["weights"].unsqueeze(-1), dim=1)
        trend_delta = self._linear_trend_delta(query["soh_seq"].to(torch.float32), self.horizon)
        trend_blend = float(self.base_trend_blend) * (1.0 - retrieval["retrieval_confidence"].to(torch.float32).clamp(0.0, 1.0))
        trend_blend = trend_blend.unsqueeze(-1).clamp(0.0, 1.0)
        base_delta = base_delta_raw * (1.0 - trend_blend) + trend_delta * trend_blend

        retrieval_features = torch.stack(
            [
                retrieval["retrieval_confidence"].to(torch.float32),
                composite_mean,
                composite_std,
                top1_distance,
                availability.mean(dim=-1),
                chemistry_match_ratio,
            ],
            dim=-1,
        )

        ref_state = self._state_features(
            retrieval["ref_anchor_soh"].reshape(batch_size * top_k).to(torch.float32),
            retrieval["ref_soh_seq"].reshape(batch_size * top_k, *retrieval["ref_soh_seq"].shape[2:]).to(torch.float32),
        )
        ref_anchor_physics = retrieval["ref_anchor_physics_features"].reshape(batch_size * top_k, -1).to(torch.float32)
        ref_operation_seq = retrieval["ref_operation_seq"].reshape(batch_size * top_k, *retrieval["ref_operation_seq"].shape[2:]).to(torch.float32)
        zero_future_summary = torch.zeros(batch_size * top_k, self.future_operation_dim, device=device, dtype=torch.float32)
        ref_operation_group = torch.cat([ref_operation_seq.mean(dim=1), ref_operation_seq.std(dim=1), zero_future_summary], dim=-1)
        ref_retrieval_features = torch.zeros(batch_size * top_k, 6, device=device, dtype=torch.float32)
        ref_retrieval_features[:, 0] = 1.0
        ref_retrieval_features[:, 4] = 1.0
        if "ref_chemistry_id" in retrieval:
            ref_retrieval_features[:, 5] = (ref_chemistry_id.reshape(-1) == chemistry_ids[:, None].expand(-1, top_k).reshape(-1)).to(torch.float32)
        ref_prior_payload = self.semantic_router.semantic_prior_from_groups(
            {
                "soh_state": ref_state,
                "qv_polarization": ref_anchor_physics[..., : min(8, self.physics_dim)],
                "partial_charge_shape": ref_anchor_physics[..., 8:],
                "operation_stress": ref_operation_group,
                "retrieval": ref_retrieval_features,
            }
        )
        ref_semantic_prior = ref_prior_payload["semantic_prior"].view(batch_size, top_k, -1)
        rag_semantic_prior = torch.sum(ref_semantic_prior * retrieval_alpha.unsqueeze(-1), dim=1)
        neighbor_vote = self._neighbor_vote(retrieval, batch_size, device)
        neighbor_vote = neighbor_vote + rag_semantic_prior
        neighbor_vote = neighbor_vote / neighbor_vote.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        router_inputs = {
            "soh_state": state_features,
            "qv_polarization": query["anchor_physics_features"].to(torch.float32)[..., : min(8, self.physics_dim)],
            "partial_charge_shape": query["anchor_physics_features"].to(torch.float32)[..., 8:],
            "operation_stress": operation_group,
            "retrieval": retrieval_features,
            "neighbor_vote": neighbor_vote,
        }
        router_out = self.semantic_router(router_inputs, rag_semantic_prior=rag_semantic_prior)

        baseline_detached = base_delta.detach()
        baseline_summary = torch.stack([baseline_detached.mean(dim=-1), baseline_detached.std(dim=-1), baseline_detached[:, -1]], dim=-1)
        expert_seq = query["expert_seq"].to(torch.float32) if "expert_seq" in query else self._fallback_expert_seq(query)
        expert_context = torch.cat(
            [
                state_features,
                query["anchor_physics_features"].to(torch.float32),
                query_encoded["meta_embedding"],
                torch.stack(
                    [
                        retrieval["retrieval_confidence"].to(torch.float32),
                        composite_mean,
                        composite_std,
                        availability.mean(dim=-1),
                        compatibility_mean,
                    ],
                    dim=-1,
                ),
                baseline_summary,
                query["future_ops"].to(torch.float32).mean(dim=1),
            ],
            dim=-1,
        )
        chemistry_outputs = []
        for chemistry in self.chemistry_families:
            mode_outputs = []
            for mode_name in self.expert_names:
                expert = self.expert_bank[chemistry][mode_name]
                mode_outputs.append(
                    expert(
                        expert_seq=expert_seq,
                        expert_context=expert_context,
                        baseline_delta_detached=baseline_detached,
                        future_ops=query["future_ops"].to(torch.float32),
                        future_ops_mask=query["future_ops_mask"].to(torch.float32),
                    )
                )
            chemistry_outputs.append(torch.stack(mode_outputs, dim=1))
        all_expert_outputs = torch.stack(chemistry_outputs, dim=1)
        chemistry_one_hot = F.one_hot(chemistry_ids, num_classes=len(self.chemistry_families)).to(all_expert_outputs.dtype)
        expert_outputs = torch.sum(all_expert_outputs * chemistry_one_hot[:, :, None, None], dim=1)
        raw_moe_residual = torch.sum(expert_outputs * router_out["weights"].unsqueeze(-1), dim=1)
        retrieval_residual_gate = retrieval["retrieval_confidence"].to(torch.float32).clamp(0.0, 1.0)
        retrieval_residual_gate = float(self.residual_confidence_floor) + (1.0 - float(self.residual_confidence_floor)) * retrieval_residual_gate
        stage_progress_gate = torch.sigmoid((float(self.residual_stage_mid_soh) - anchor_soh.to(torch.float32)) * float(self.residual_stage_sharpness))
        stage_progress_gate = float(self.residual_stage_floor) + (1.0 - float(self.residual_stage_floor)) * stage_progress_gate
        residual_gate = (retrieval_residual_gate * stage_progress_gate).clamp(0.0, 1.0)
        if self.residual_max_abs > 0:
            # Residual experts are corrective specialists, not a second full
            # forecaster. Bound their amplitude and shrink it when RAG
            # confidence is modest so they cannot override the smooth base
            # trajectory with a large unsupported correction.
            residual_bound = float(self.residual_max_abs) * residual_gate.unsqueeze(-1)
            moe_residual = residual_bound * torch.tanh(raw_moe_residual / residual_bound.clamp_min(1e-6))
        else:
            residual_bound = torch.full_like(raw_moe_residual, float("inf"))
            moe_residual = raw_moe_residual
        pred_delta = base_delta + moe_residual
        pred_soh = anchor_soh.unsqueeze(-1) + pred_delta

        top1_topk_gap = composite_distance[:, 0].masked_fill(retrieval_mask[:, 0] <= 0, 0.0) - composite_mean
        return {
            "pred_delta": pred_delta,
            "pred_soh": pred_soh,
            "fm_delta": fm_delta,
            "rag_delta": rag_delta,
            "pair_delta": pair_delta,
            "base_delta": base_delta,
            "base_delta_raw": base_delta_raw,
            "trend_delta": trend_delta,
            "trend_blend": trend_blend.squeeze(-1),
            "moe_residual": moe_residual,
            "raw_moe_residual": raw_moe_residual,
            "residual_gate": residual_gate,
            "residual_retrieval_gate": retrieval_residual_gate,
            "residual_stage_gate": stage_progress_gate,
            "residual_bound": residual_bound,
            "pair_residual": pair_residual,
            "retrieval_alpha": retrieval_alpha,
            "retrieval_confidence": retrieval["retrieval_confidence"].to(torch.float32),
            "expert_weights": router_out["weights"],
            "expert_logits": router_out["logits"],
            "expert_router_contributions": router_out["contributions"],
            "semantic_concepts": router_out["concept_scores"],
            "semantic_prior": router_out["semantic_prior"],
            "rag_semantic_prior": router_out["rag_semantic_prior"],
            "final_router_prior": router_out["final_prior"],
            "chemistry_branch_id": chemistry_ids,
            "selected_mode_residuals": expert_outputs,
            "fusion_weights": fusion_out["weights"],
            "base_fusion_weights": fusion_out["weights"],
            "fusion_router_contributions": fusion_out["contributions"],
            "diagnostics": {
                "retrieval_distance_mean": composite_mean,
                "retrieval_distance_std": composite_std,
                "top1_topk_gap": top1_topk_gap,
            },
        }
