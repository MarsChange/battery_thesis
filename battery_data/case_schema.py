from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class CaseSample:
    case_id: int
    cell_uid: str
    source_dataset: str
    raw_cell_id: str
    split: str
    domain_label: str
    window_start: int
    window_end: int
    target_start: int
    target_end: int
    cycle_idx_start: int
    cycle_idx_end: int
    target_cycle_idx_start: int
    target_cycle_idx_end: int
    chemistry_family: str
    temperature_bucket: str
    charge_rate_bucket: str
    discharge_policy_family: str
    nominal_capacity_bucket: str
    voltage_window_bucket: str
    full_or_partial: str
    anchor_soh: float
    anchor_capacity: float
    recent_soh_slope: float
    recent_soh_curvature: float
    throughput_recent: float
    degradation_stage: str
    target_delta_soh: np.ndarray
    target_soh: np.ndarray
    cycle_stats: np.ndarray
    soh_seq: np.ndarray
    qv_maps: np.ndarray
    qv_masks: np.ndarray
    partial_charge_curves: np.ndarray
    partial_charge_masks: np.ndarray
    relaxation_curves: np.ndarray
    relaxation_masks: np.ndarray
    physics_features: np.ndarray
    physics_feature_masks: np.ndarray
    anchor_physics_features: np.ndarray
    operation_seq: np.ndarray
    future_operation_seq: np.ndarray
    future_operation_mask: np.ndarray
    tsfm_embedding: np.ndarray | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    feature_names: Dict[str, Any] = field(default_factory=dict)
    missing_mask: Dict[str, Any] = field(default_factory=dict)

    def to_row_dict(self) -> Dict[str, Any]:
        scalar_fields = {
            "case_id": int(self.case_id),
            "cell_uid": self.cell_uid,
            "source_dataset": self.source_dataset,
            "raw_cell_id": self.raw_cell_id,
            "split": self.split,
            "domain_label": self.domain_label,
            "window_start": int(self.window_start),
            "window_end": int(self.window_end),
            "target_start": int(self.target_start),
            "target_end": int(self.target_end),
            "cycle_idx_start": int(self.cycle_idx_start),
            "cycle_idx_end": int(self.cycle_idx_end),
            "target_cycle_idx_start": int(self.target_cycle_idx_start),
            "target_cycle_idx_end": int(self.target_cycle_idx_end),
            "chemistry_family": self.chemistry_family,
            "temperature_bucket": self.temperature_bucket,
            "charge_rate_bucket": self.charge_rate_bucket,
            "discharge_policy_family": self.discharge_policy_family,
            "nominal_capacity_bucket": self.nominal_capacity_bucket,
            "voltage_window_bucket": self.voltage_window_bucket,
            "full_or_partial": self.full_or_partial,
            "anchor_soh": float(self.anchor_soh),
            "anchor_capacity": float(self.anchor_capacity),
            "recent_soh_slope": float(self.recent_soh_slope),
            "recent_soh_curvature": float(self.recent_soh_curvature),
            "throughput_recent": float(self.throughput_recent),
            "degradation_stage": self.degradation_stage,
            "target_horizon": int(np.asarray(self.target_delta_soh).shape[0]),
            "lookback_length": int(np.asarray(self.soh_seq).shape[0]),
            "qv_width": int(np.asarray(self.qv_maps).shape[-1]),
            "relaxation_points": int(np.asarray(self.relaxation_curves).shape[-1]),
            "physics_dim": int(np.asarray(self.physics_features).shape[-1]),
            "operation_dim": int(np.asarray(self.operation_seq).shape[-1]),
            "future_operation_dim": int(np.asarray(self.future_operation_seq).shape[-1]),
            "has_tsfm_embedding": bool(self.tsfm_embedding is not None),
            "partial_charge_availability": float(np.asarray(self.partial_charge_masks, dtype=np.float32).mean()) if np.asarray(self.partial_charge_masks).size else 0.0,
            "relaxation_availability": float(np.asarray(self.relaxation_masks, dtype=np.float32).mean()) if np.asarray(self.relaxation_masks).size else 0.0,
            "qv_availability": float(np.asarray(self.qv_masks, dtype=np.float32).mean()) if np.asarray(self.qv_masks).size else 0.0,
            "physics_feature_availability": float(np.asarray(self.physics_feature_masks, dtype=np.float32).mean()) if np.asarray(self.physics_feature_masks).size else 0.0,
            "metadata_json": json.dumps(self.metadata, ensure_ascii=True, sort_keys=True),
            "feature_names_json": json.dumps(self.feature_names, ensure_ascii=True, sort_keys=True),
            "missing_mask_json": json.dumps(self.missing_mask, ensure_ascii=True, sort_keys=True),
        }
        return scalar_fields
