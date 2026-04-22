from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from forecasting.routers import BranchFusionRouter, PhysicalDegradationRouter


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(values.dtype)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    summed = (values * weight).sum(dim=dim)
    denom = weight.sum(dim=dim).clamp_min(1e-6)
    return summed / denom


class CycleStatsEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cycle_stats: torch.Tensor, soh_seq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([cycle_stats, soh_seq], dim=-1)
        _, hidden = self.gru(x)
        return self.dropout(hidden[-1])


class QVMapEncoder(nn.Module):
    def __init__(self, hidden_dim: int, qv_width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 5), padding=(1, 2)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 4)),
        )
        self.proj = nn.Linear(16 * 4, hidden_dim)

    def forward(self, qv_maps: torch.Tensor, qv_masks: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, width = qv_maps.shape
        x = qv_maps.view(batch * steps, 1, channels, width)
        feat = self.net(x).reshape(batch, steps, -1)
        feat = self.proj(feat)
        cycle_mask = (qv_masks.sum(dim=-1) > 0).to(qv_maps.dtype)
        return _masked_mean(feat, cycle_mask, dim=1)


class SequenceCurveEncoder(nn.Module):
    def __init__(self, length: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
        )
        self.proj = nn.Linear(16 * 4, hidden_dim)

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        batch, steps, length = values.shape
        x = values.view(batch * steps, 1, length)
        feat = self.net(x).reshape(batch, steps, -1)
        feat = self.proj(feat)
        return _masked_mean(feat, masks, dim=1)


class FutureOperationEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, future_ops: torch.Tensor, future_ops_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = future_ops * future_ops_mask
        outputs, hidden = self.gru(x)
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


class BatterySOHForecaster(nn.Module):
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
        relaxation_points: int = 30,
        tsfm_dim: int = 0,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        meta_embedding_dim: int = 16,
        expert_names: List[str] | None = None,
        top_k_experts: int = 2,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.cycle_feature_dim = int(cycle_feature_dim)
        self.physics_dim = int(physics_dim)
        self.operation_dim = int(operation_dim)
        self.future_operation_dim = int(future_operation_dim)
        self.meta_dim = int(meta_dim)
        self.tsfm_dim = int(tsfm_dim)
        self.expert_names = list(expert_names or ["shared", "LFP", "NCM", "NCA", "slow_linear", "accelerating", "high_polarization", "relaxation_impedance"])

        self.cycle_encoder = CycleStatsEncoder(self.cycle_feature_dim + 1, hidden_dim, dropout)
        self.qv_encoder = QVMapEncoder(hidden_dim, qv_width)
        self.partial_charge_encoder = SequenceCurveEncoder(partial_charge_points, hidden_dim)
        self.relaxation_encoder = SequenceCurveEncoder(relaxation_points, hidden_dim)
        self.future_op_encoder = FutureOperationEncoder(self.future_operation_dim, hidden_dim)
        self.meta_embedding = nn.Embedding(512, meta_embedding_dim)

        state_dim = 3
        query_repr_dim = hidden_dim * 4 + hidden_dim + meta_embedding_dim + tsfm_dim + state_dim + physics_dim
        pair_input_dim = query_repr_dim * 4 + self.horizon + 8 + 1
        expert_input_dim = hidden_dim * 5 + meta_embedding_dim + tsfm_dim + state_dim + self.horizon * 3

        self.generalist_head = MLPHead(hidden_dim + meta_embedding_dim + tsfm_dim, self.horizon, hidden_dim, dropout)
        self.pairwise_branch = MLPHead(pair_input_dim, self.horizon, hidden_dim, dropout)
        self.expert_bank = nn.ModuleList([MLPHead(expert_input_dim, self.horizon, hidden_dim, dropout) for _ in self.expert_names])

        self.physical_router = PhysicalDegradationRouter(
            group_dims={
                "state": 3,
                "curve": physics_dim * 4,
                "partial_charge": 4,
                "relaxation": 4,
                "operation": operation_dim * 2 + future_operation_dim,
                "chemistry": meta_embedding_dim,
                "retrieval": 5,
                "availability": 3,
            },
            num_experts=len(self.expert_names),
            top_k_experts=top_k_experts,
        )
        self.fusion_router = BranchFusionRouter(
            group_dims={
                "retrieval": 4,
                "state": 3,
                "availability": 2,
                "chemistry": meta_embedding_dim,
            },
            num_branches=4,
        )

    def _meta_encode(self, meta_ids: torch.Tensor) -> torch.Tensor:
        emb = self.meta_embedding(meta_ids.long().clamp(0, 511))
        return emb.mean(dim=1)

    def _state_features(self, anchor_soh: torch.Tensor, soh_seq: torch.Tensor) -> torch.Tensor:
        seq = soh_seq.squeeze(-1)
        slope = seq[:, -1] - seq[:, -2] if seq.size(1) >= 2 else torch.zeros_like(anchor_soh)
        if seq.size(1) >= 3:
            curvature = seq[:, -1] - 2 * seq[:, -2] + seq[:, -3]
        else:
            curvature = torch.zeros_like(anchor_soh)
        return torch.stack([anchor_soh, slope, curvature], dim=-1)

    def _availability_features(self, partial_mask: torch.Tensor, relax_mask: torch.Tensor, physics_mask: torch.Tensor) -> torch.Tensor:
        physics_avail = physics_mask.mean(dim=(1, 2))
        partial_avail = partial_mask.mean(dim=1)
        relax_avail = relax_mask.mean(dim=1)
        return torch.stack([physics_avail, partial_avail, relax_avail], dim=-1)

    def _curve_group_features(self, physics_features: torch.Tensor, physics_mask: torch.Tensor, anchor_physics: torch.Tensor) -> torch.Tensor:
        masked = physics_features * physics_mask
        mean_feat = physics_features.mean(dim=1)
        std_feat = physics_features.std(dim=1)
        delta_feat = physics_features[:, -1] - physics_features[:, 0]
        return torch.cat([anchor_physics, mean_feat, std_feat, delta_feat], dim=-1)

    def _partial_group_features(self, partial_charge: torch.Tensor, partial_mask: torch.Tensor) -> torch.Tensor:
        anchor = partial_charge[:, -1]
        return torch.stack(
            [
                anchor.mean(dim=-1),
                anchor.std(dim=-1),
                anchor[:, -1],
                partial_mask.mean(dim=-1),
            ],
            dim=-1,
        )

    def _relax_group_features(self, relaxation: torch.Tensor, relaxation_mask: torch.Tensor) -> torch.Tensor:
        anchor = relaxation[:, -1]
        return torch.stack(
            [
                anchor.mean(dim=-1),
                anchor.std(dim=-1),
                anchor[:, -1] - anchor[:, 0],
                relaxation_mask.mean(dim=-1),
            ],
            dim=-1,
        )

    def _operation_group_features(self, operation_seq: torch.Tensor, future_ops: torch.Tensor) -> torch.Tensor:
        return torch.cat([operation_seq.mean(dim=1), operation_seq.std(dim=1), future_ops.mean(dim=1)], dim=-1)

    def _encode_modalities(
        self,
        cycle_stats: torch.Tensor,
        soh_seq: torch.Tensor,
        qv_maps: torch.Tensor,
        qv_masks: torch.Tensor,
        partial_charge: torch.Tensor,
        partial_charge_mask: torch.Tensor,
        relaxation: torch.Tensor,
        relaxation_mask: torch.Tensor,
        future_ops: torch.Tensor,
        future_ops_mask: torch.Tensor,
        anchor_physics_features: torch.Tensor,
        meta_ids: torch.Tensor,
        tsfm_embedding: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        h_cycle = self.cycle_encoder(cycle_stats, soh_seq)
        h_qv = self.qv_encoder(qv_maps, qv_masks)
        h_pc = self.partial_charge_encoder(partial_charge, partial_charge_mask)
        h_relax = self.relaxation_encoder(relaxation, relaxation_mask)
        h_future_ops, future_horizon_context = self.future_op_encoder(future_ops, future_ops_mask)
        meta_embedding = self._meta_encode(meta_ids)
        return {
            "h_cycle": h_cycle,
            "h_qv": h_qv,
            "h_pc": h_pc,
            "h_relax": h_relax,
            "h_future_ops": h_future_ops,
            "future_horizon_context": future_horizon_context,
            "meta_embedding": meta_embedding,
            "tsfm_embedding": tsfm_embedding,
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
            relaxation=query["relaxation"].to(torch.float32),
            relaxation_mask=query["relaxation_mask"].to(torch.float32),
            future_ops=query["future_ops"].to(torch.float32),
            future_ops_mask=query["future_ops_mask"].to(torch.float32),
            anchor_physics_features=query["anchor_physics_features"].to(torch.float32),
            meta_ids=query["meta_ids"].to(device),
            tsfm_embedding=query["tsfm_embedding"].to(torch.float32),
        )

        state_features = self._state_features(anchor_soh, query["soh_seq"].to(torch.float32))
        availability_features = self._availability_features(
            query["partial_charge_mask"].to(torch.float32),
            query["relaxation_mask"].to(torch.float32),
            query["physics_feature_masks"].to(torch.float32),
        )
        curve_group = self._curve_group_features(
            query["physics_features"].to(torch.float32),
            query["physics_feature_masks"].to(torch.float32),
            query["anchor_physics_features"].to(torch.float32),
        )
        partial_group = self._partial_group_features(query["partial_charge"].to(torch.float32), query["partial_charge_mask"].to(torch.float32))
        relax_group = self._relax_group_features(query["relaxation"].to(torch.float32), query["relaxation_mask"].to(torch.float32))
        operation_group = self._operation_group_features(query["operation_seq"].to(torch.float32), query["future_ops"].to(torch.float32))

        rag_delta = torch.sum(
            retrieval["ref_future_delta_soh"].to(torch.float32) * retrieval["retrieval_alpha"].to(torch.float32).unsqueeze(-1),
            dim=1,
        )

        fm_input = torch.cat(
            [query_encoded["h_cycle"], query_encoded["meta_embedding"], query_encoded["tsfm_embedding"]],
            dim=-1,
        )
        fm_delta = self.generalist_head(fm_input)

        batch_size, top_k = retrieval["ref_future_delta_soh"].shape[:2]
        ref_qv = retrieval["ref_qv_maps"].to(torch.float32)
        ref_qv_mask = (ref_qv.abs().sum(dim=-1) > 0).to(torch.float32)
        ref_partial = retrieval["ref_partial_charge"].to(torch.float32)
        ref_partial_mask = (ref_partial.abs().sum(dim=-1) > 0).to(torch.float32)
        ref_relax = retrieval["ref_relaxation"].to(torch.float32)
        ref_relax_mask = (ref_relax.abs().sum(dim=-1) > 0).to(torch.float32)
        ref_encoded = self._encode_modalities(
            cycle_stats=retrieval["ref_cycle_stats"].reshape(batch_size * top_k, *retrieval["ref_cycle_stats"].shape[2:]).to(torch.float32),
            soh_seq=torch.zeros(batch_size * top_k, query["soh_seq"].shape[1], 1, device=device, dtype=torch.float32),
            qv_maps=ref_qv.reshape(batch_size * top_k, *ref_qv.shape[2:]),
            qv_masks=ref_qv_mask.reshape(batch_size * top_k, *ref_qv_mask.shape[2:]),
            partial_charge=ref_partial.reshape(batch_size * top_k, *ref_partial.shape[2:]),
            partial_charge_mask=ref_partial_mask.reshape(batch_size * top_k, ref_partial_mask.shape[2]),
            relaxation=ref_relax.reshape(batch_size * top_k, *ref_relax.shape[2:]),
            relaxation_mask=ref_relax_mask.reshape(batch_size * top_k, ref_relax_mask.shape[2]),
            future_ops=query["future_ops"].repeat_interleave(top_k, dim=0).to(torch.float32),
            future_ops_mask=query["future_ops_mask"].repeat_interleave(top_k, dim=0).to(torch.float32),
            anchor_physics_features=retrieval["ref_anchor_physics_features"].reshape(batch_size * top_k, -1).to(torch.float32),
            meta_ids=retrieval["ref_meta_ids"].reshape(batch_size * top_k, -1).to(device),
            tsfm_embedding=retrieval["ref_tsfm_embedding"].reshape(batch_size * top_k, -1).to(torch.float32),
        )
        query_repr = torch.cat(
            [
                query_encoded["h_cycle"],
                query_encoded["h_qv"],
                query_encoded["h_pc"],
                query_encoded["h_relax"],
                query_encoded["h_future_ops"],
                query_encoded["meta_embedding"],
                query_encoded["tsfm_embedding"],
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
                ref_encoded["h_relax"],
                ref_encoded["h_future_ops"],
                ref_encoded["meta_embedding"],
                ref_encoded["tsfm_embedding"],
                state_features.repeat_interleave(top_k, dim=0),
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
                retrieval["ref_future_delta_soh"].to(torch.float32),
                retrieval["component_distances"].to(torch.float32),
                retrieval["reference_compatibility_score"].to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_residual = self.pairwise_branch(pair_features)
        pair_delta_per_ref = retrieval["ref_future_delta_soh"].to(torch.float32) + pair_residual
        pair_delta = torch.sum(pair_delta_per_ref * retrieval["retrieval_alpha"].to(torch.float32).unsqueeze(-1), dim=1)

        router_inputs = {
            "state": state_features,
            "curve": curve_group,
            "partial_charge": partial_group,
            "relaxation": relax_group,
            "operation": operation_group,
            "chemistry": query_encoded["meta_embedding"],
            "retrieval": torch.stack(
                [
                    retrieval["retrieval_confidence"].to(torch.float32),
                    retrieval["component_distances"].to(torch.float32).mean(dim=(1, 2)),
                    retrieval["component_distances"].to(torch.float32).std(dim=(1, 2)),
                    retrieval["retrieval_alpha"].to(torch.float32).max(dim=1).values,
                    retrieval["reference_compatibility_score"].to(torch.float32).mean(dim=1),
                ],
                dim=-1,
            ),
            "availability": availability_features,
        }
        router_out = self.physical_router(router_inputs)

        expert_input = torch.cat(
            [
                query_encoded["h_cycle"],
                query_encoded["h_qv"],
                query_encoded["h_pc"],
                query_encoded["h_relax"],
                query_encoded["h_future_ops"],
                query_encoded["meta_embedding"],
                query_encoded["tsfm_embedding"],
                state_features,
                fm_delta,
                rag_delta,
                pair_delta,
            ],
            dim=-1,
        )
        expert_outputs = torch.stack([expert(expert_input) for expert in self.expert_bank], dim=1)
        moe_delta = torch.sum(expert_outputs * router_out["weights"].unsqueeze(-1), dim=1)

        retrieval_dist_mean = retrieval["component_distances"].to(torch.float32).mean(dim=(1, 2))
        retrieval_dist_std = retrieval["component_distances"].to(torch.float32).std(dim=(1, 2))
        top1_topk_gap = retrieval["component_distances"].to(torch.float32)[:, 0].mean(dim=-1) - retrieval_dist_mean
        fusion_inputs = {
            "retrieval": torch.stack(
                [
                    retrieval["retrieval_confidence"].to(torch.float32),
                    retrieval_dist_mean,
                    retrieval_dist_std,
                    top1_topk_gap,
                ],
                dim=-1,
            ),
            "state": state_features,
            "availability": torch.stack(
                [
                    query["physics_feature_masks"].to(torch.float32).mean(dim=(1, 2)),
                    query["future_ops_mask"].to(torch.float32).mean(dim=(1, 2)),
                ],
                dim=-1,
            ),
            "chemistry": query_encoded["meta_embedding"],
        }
        fusion_out = self.fusion_router(fusion_inputs)
        branch_stack = torch.stack([fm_delta, rag_delta, pair_delta, moe_delta], dim=1)
        pred_delta = torch.sum(branch_stack * fusion_out["weights"].unsqueeze(-1), dim=1)
        pred_soh = anchor_soh.unsqueeze(-1) + pred_delta

        return {
            "pred_delta": pred_delta,
            "pred_soh": pred_soh,
            "fm_delta": fm_delta,
            "rag_delta": rag_delta,
            "pair_delta": pair_delta,
            "moe_delta": moe_delta,
            "pair_residual": pair_residual,
            "retrieval_alpha": retrieval["retrieval_alpha"].to(torch.float32),
            "retrieval_confidence": retrieval["retrieval_confidence"].to(torch.float32),
            "expert_weights": router_out["weights"],
            "expert_logits": router_out["logits"],
            "expert_router_contributions": router_out["contributions"],
            "fusion_weights": fusion_out["weights"],
            "fusion_router_contributions": fusion_out["contributions"],
            "diagnostics": {
                "retrieval_distance_mean": retrieval_dist_mean,
                "retrieval_distance_std": retrieval_dist_std,
                "top1_topk_gap": top1_topk_gap,
            },
        }
