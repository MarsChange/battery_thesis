"""retrieval.multistage_retriever

实现用于电池 SOH 任务的多阶段历史案例检索器。
该检索器遵守 `configs/retrieval_features.yaml` 中的显式开关：
1. Stage 0: hard filtering，确保 target_query 不进入 reference DB，且排除 self-retrieval。
2. Stage 1: coarse retrieval，默认用 handcrafted battery embedding 召回候选。
3. Stage 2: physics-aware reranking，计算命名距离分量并生成 composite_distance。
4. Stage 3: optional diversity selection，用 MMR 控制 top-k 多样性。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from retrieval.physics_distance import (
    ALL_COMPONENT_NAMES,
    CORE_COMPONENT_NAMES,
    OPTIONAL_COMPONENT_NAMES,
    QV_CHANNEL_TO_INDEX,
    curve_shape_distance,
    compute_composite_distance,
    compute_metadata_distance,
    compute_operation_distance,
    compute_physics_distance,
    compute_qv_shape_distance,
    compute_retrieval_confidence,
    compute_soh_state_distance,
    degradation_stage_distance,
    soh_history_shape_distance,
)

try:
    from retrieval.index import FAISSIndex
except Exception:  # pragma: no cover - fallback path is enough for tests without faiss.
    FAISSIndex = None


CORE_COMPONENTS = list(CORE_COMPONENT_NAMES)
COMPONENT_NAMES = list(ALL_COMPONENT_NAMES)
OPTIONAL_COMPONENTS = list(OPTIONAL_COMPONENT_NAMES)
RETRIEVAL_CACHE_SCHEMA_VERSION = "qv_window_factor_experts_v1"


def _read_case_rows(case_bank_dir: Path) -> pd.DataFrame:
    parquet_path = case_bank_dir / "case_rows.parquet"
    csv_path = case_bank_dir / "case_rows.csv"
    if parquet_path.exists():
        try:
            return _normalize_case_rows(pd.read_parquet(parquet_path))
        except Exception:
            if csv_path.exists():
                return _normalize_case_rows(pd.read_csv(csv_path))
            raise
    if csv_path.exists():
        return _normalize_case_rows(pd.read_csv(csv_path))
    raise FileNotFoundError(f"Missing case rows at {parquet_path} or {csv_path}")


def _normalize_case_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Use plain Python metadata scalars for retrieval loops.

    Retrieval computes distances for many query/reference pairs. Pyarrow-backed
    parquet string scalars are slow in that loop, so string-like columns are
    converted once when the case rows are loaded. Numeric columns remain numeric.
    """

    frame = rows.copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].fillna("unknown").astype(str).astype(object)
    return frame


def _read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _softmax_masked(distances: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    logits = -np.asarray(distances, dtype=np.float32) / max(float(temperature), 1e-6)
    mask_arr = np.asarray(mask, dtype=np.float32)
    if logits.size == 0:
        return logits.astype(np.float32)
    logits = np.where(mask_arr > 0, logits, -1e9)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits) * mask_arr
    denom = float(weights.sum())
    if denom <= 0:
        return np.zeros_like(logits, dtype=np.float32)
    return (weights / denom).astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(arr_a) * np.linalg.norm(arr_b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(arr_a, arr_b) / denom)


def component_matrix_to_named_list(component_distances: np.ndarray) -> list[dict[str, float]]:
    """把 `[K, C]` 距离矩阵转换成带名称的字典列表。"""

    matrix = np.asarray(component_distances, dtype=np.float32)
    result = []
    for row in matrix.tolist():
        result.append({name: float(value) for name, value in zip(COMPONENT_NAMES, row)})
    return result


@dataclass
class RetrievalResult:
    query_case_id: int
    neighbor_case_ids: np.ndarray
    retrieval_mask: np.ndarray
    d_soh_state: np.ndarray
    d_qv_shape: np.ndarray
    d_physics: np.ndarray
    d_operation: np.ndarray
    d_metadata: np.ndarray
    composite_distance: np.ndarray
    retrieval_confidence: float
    retrieval_alpha: np.ndarray
    neighbor_future_delta_soh: np.ndarray
    neighbor_metadata: list[dict[str, object]] = field(default_factory=list)
    explain_json: Dict[str, object] = field(default_factory=dict)
    stage_distance: np.ndarray | None = None
    missing_penalty: np.ndarray | None = None
    reference_compatibility_score: np.ndarray | None = None
    mmr_diversity_score: np.ndarray | None = None

    @property
    def component_distances(self) -> np.ndarray:
        return np.stack(
            [
                self.d_soh_state,
                self.d_qv_shape,
                self.d_physics,
                self.d_operation,
                self.d_metadata,
            ],
            axis=-1,
        ).astype(np.float32)

    @property
    def composite_distances(self) -> np.ndarray:
        return self.composite_distance

    @property
    def explain(self) -> Dict[str, object]:
        return self.explain_json


class MultiStageBatteryRetriever:
    def __init__(
        self,
        case_bank_dir: str | Path,
        retrieval_config_path: str | Path = "configs/retrieval_features.yaml",
        db_splits: List[str] | None = None,
        query_splits: List[str] | None = None,
        top_m: int | None = None,
        top_k: int | None = None,
        same_cell_policy: str | None = None,
        exclude_query_case: bool = True,
        allow_cross_chemistry: bool | None = None,
        metric: str = "cosine",
        stage1_embedding_name: str | None = None,
        retrieval_temperature: float = 0.1,
        hard_filter: Dict[str, object] | None = None,
        mmr: Dict[str, object] | None = None,
        rerank_weights: Dict[str, float] | None = None,
        qv_channel_weights: Dict[str, float] | None = None,
    ):
        import yaml

        self.case_bank_dir = Path(case_bank_dir)
        self.retrieval_config_path = Path(retrieval_config_path)
        if not self.retrieval_config_path.exists() and self.retrieval_config_path.name == "rag_retrieval_features.yaml":
            # Backward-compatible path mapping only. The active mainline config
            # is configs/retrieval_features.yaml and contains no external
            # sequence embedding distance.
            fallback_path = Path("configs/retrieval_features.yaml")
            if fallback_path.exists():
                self.retrieval_config_path = fallback_path
        self.rag_config = yaml.safe_load(self.retrieval_config_path.read_text())
        self.case_rows = _read_case_rows(self.case_bank_dir).sort_values("case_id").reset_index(drop=True)
        self.case_rows["case_id"] = self.case_rows["case_id"].astype(int)
        self.case_id_to_index = {int(case_id): row_idx for row_idx, case_id in enumerate(self.case_rows["case_id"].tolist())}
        self.metric = str(metric)

        final_topk_cfg = dict(self.rag_config.get("final_topk", {}) or {})
        stage1_cfg = dict(self.rag_config.get("stage1_retrieval", {}) or {})
        hard_filter = dict(hard_filter or {})
        mmr = dict(mmr or {})

        self.db_splits = list(db_splits or ["source_train"])
        self.query_splits = list(query_splits or [])
        self.top_m = int(top_m if top_m is not None else stage1_cfg.get("top_m", 200))
        self.top_k = int(top_k if top_k is not None else final_topk_cfg.get("top_k", 8))
        self.same_cell_policy = str(same_cell_policy or hard_filter.get("same_cell_policy") or final_topk_cfg.get("same_cell_policy", "exclude"))
        self.exclude_query_case = bool(exclude_query_case)
        self.allow_cross_chemistry = bool(
            allow_cross_chemistry if allow_cross_chemistry is not None else final_topk_cfg.get("allow_cross_chemistry", True)
        )
        self.retrieval_temperature = float(retrieval_temperature)
        self.use_mmr = bool(mmr.get("use_mmr", final_topk_cfg.get("use_mmr", True)))
        self.mmr_lambda = float(mmr.get("mmr_lambda", final_topk_cfg.get("mmr_lambda", 0.75)))
        self.max_neighbors_per_cell = int(mmr.get("max_neighbors_per_cell", final_topk_cfg.get("max_neighbors_per_cell", 2)))
        self.max_neighbors_per_domain = mmr.get("max_neighbors_per_domain", final_topk_cfg.get("max_neighbors_per_domain"))
        self.stage1_embedding_name = "handcrafted"

        if rerank_weights:
            for distance_name, weight in rerank_weights.items():
                if distance_name in dict(self.rag_config.get("distance_components", {}) or {}):
                    self.rag_config["distance_components"][distance_name]["weight"] = float(weight)
        if qv_channel_weights:
            qv_cfg = dict(self.rag_config.get("distance_components", {}).get("d_qv_shape", {}) or {})
            qv_cfg["channel_weights"] = dict(qv_cfg.get("channel_weights", {}) or {})
            qv_cfg["channel_weights"].update({key: float(value) for key, value in qv_channel_weights.items()})
            self.rag_config["distance_components"]["d_qv_shape"] = qv_cfg
        self.core_component_names = list(CORE_COMPONENTS)
        self.optional_component_names = list(OPTIONAL_COMPONENTS)
        if "distance_components" in self.rag_config:
            self.enabled_component_names = [
                name
                for name in COMPONENT_NAMES
                if bool(dict(self.rag_config.get("distance_components", {}).get(name, {}) or {}).get("enabled", False))
            ]
        else:
            self.enabled_component_names = [name for name in COMPONENT_NAMES if bool(self.rag_config.get(name, False))]

        self.feature_names = dict(_read_json(self.case_bank_dir / "feature_names.json", {}))
        self.arrays = self._load_case_arrays()
        self.db_mask = (
            self.case_rows["split"].astype(str).isin(self.db_splits).to_numpy()
            & self.case_rows["split"].astype(str).ne("target_query").to_numpy()
        )
        self.db_indices = np.flatnonzero(self.db_mask).astype(np.int64)
        self.db_case_ids = self.case_rows.loc[self.db_indices, "case_id"].to_numpy(dtype=np.int64)

        self.handcrafted_retrieval_embedding = self._build_handcrafted_embeddings()
        self.handcrafted_embeddings = self.handcrafted_retrieval_embedding
        self.stage1_embeddings = self._build_stage1_embeddings()
        self.db_stage1_embeddings = self.stage1_embeddings[self.db_indices]
        self._prepare_fast_retrieval_tables()
        self._build_index()

    def _prepare_fast_retrieval_tables(self) -> None:
        """Precompute NumPy lookup tables used by bulk cache construction."""

        self._row_case_ids = self.case_rows["case_id"].to_numpy(dtype=np.int64)
        self._row_cell_uid = self.case_rows["cell_uid"].astype(str).to_numpy(dtype=object)
        self._row_split = self.case_rows["split"].astype(str).to_numpy(dtype=object)
        self._row_chemistry = self.case_rows["chemistry_family"].astype(str).to_numpy(dtype=object)
        self._row_domain = self.case_rows["domain_label"].astype(str).to_numpy(dtype=object)
        self._row_voltage_window = self.case_rows["voltage_window_bucket"].astype(str).to_numpy(dtype=object)
        self._row_degradation_stage = self.case_rows["degradation_stage"].astype(str).to_numpy(dtype=object)
        self._row_window_end = self.case_rows["window_end"].fillna(0).to_numpy(dtype=np.int64)
        self._row_target_horizon = self.case_rows["target_horizon"].fillna(0).to_numpy(dtype=np.int64)
        self._future_delta_finite = np.isfinite(np.asarray(self.arrays["future_delta"], dtype=np.float32)).all(axis=1)
        self._state_matrix = self.case_rows[["anchor_soh", "recent_soh_slope", "recent_soh_curvature"]].fillna(0.0).to_numpy(dtype=np.float32)
        self._soh_history_delta_matrix, self._soh_history_slope_matrix = self._build_soh_history_shape_matrices()
        self._qv_summary_matrix, self._qv_summary_mask = self._build_qv_summary_matrix()
        self._physics_matrix = np.asarray(self.arrays["anchor_physics_features"], dtype=np.float32)
        self._physics_mask_matrix = np.asarray(self.arrays["physics_feature_masks"][:, -1, :], dtype=np.float32)
        self._metadata_summary_matrix = np.asarray(self.handcrafted_retrieval_embedding[:, -12:], dtype=np.float32)

    def _build_soh_history_shape_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        soh_seq = np.asarray(self.arrays["soh_seq"], dtype=np.float32)
        anchors = soh_seq[:, -1:]
        anchors = np.where(np.isfinite(anchors), anchors, np.nanmean(soh_seq, axis=1, keepdims=True))
        delta = (soh_seq - anchors).astype(np.float32)
        slope = np.diff(soh_seq, axis=1).astype(np.float32)
        return delta, slope

    def _build_qv_summary_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        cycle_last = np.asarray(self.arrays["cycle_stats"][:, -1, :], dtype=np.float32)
        values = []
        masks = []
        for name, fallback in [
            ("qv_dqdv_peak_value", "dq_dv_peak_value"),
            ("qv_dqdv_peak_voltage", "dq_dv_peak_position"),
            ("qv_dqdv_area", None),
            ("qv_capacity_span", "q_total"),
        ]:
            feature_idx = self._cycle_feature_index(name)
            if feature_idx is None and fallback is not None:
                feature_idx = self._cycle_feature_index(fallback)
            if feature_idx is not None and feature_idx < cycle_last.shape[1]:
                values.append(cycle_last[:, feature_idx])
                masks.append(np.ones(cycle_last.shape[0], dtype=np.float32))
            else:
                values.append(np.zeros(cycle_last.shape[0], dtype=np.float32))
                masks.append(np.zeros(cycle_last.shape[0], dtype=np.float32))
        return np.stack(values, axis=1).astype(np.float32), np.stack(masks, axis=1).astype(np.float32)

    def _load_case_arrays(self) -> Dict[str, np.ndarray | None]:
        arrays: Dict[str, np.ndarray | None] = {
            "future_delta": np.load(self.case_bank_dir / "case_future_delta_soh.npy"),
            "future_soh": np.load(self.case_bank_dir / "case_future_soh.npy"),
            "cycle_stats": np.load(self.case_bank_dir / "case_cycle_stats.npy"),
            "soh_seq": np.load(self.case_bank_dir / "case_soh_seq.npy"),
            "qv_maps": np.load(self.case_bank_dir / "case_qv_maps.npy"),
            "qv_masks": np.load(self.case_bank_dir / "case_qv_masks.npy"),
            "partial_charge": np.load(self.case_bank_dir / "case_partial_charge.npy"),
            "partial_charge_mask": np.load(self.case_bank_dir / "case_partial_charge_mask.npy"),
            "physics_features": np.load(self.case_bank_dir / "case_physics_features.npy"),
            "physics_feature_masks": np.load(self.case_bank_dir / "case_physics_feature_masks.npy"),
            "anchor_physics_features": np.load(self.case_bank_dir / "case_anchor_physics_features.npy"),
            "operation_seq": np.load(self.case_bank_dir / "case_operation_seq.npy"),
            "future_ops": np.load(self.case_bank_dir / "case_future_ops.npy"),
            "future_ops_mask": np.load(self.case_bank_dir / "case_future_ops_mask.npy"),
        }
        return arrays

    def _build_index(self) -> None:
        if self.db_stage1_embeddings.size == 0 or FAISSIndex is None:
            self.index = None
            return
        self.index = FAISSIndex(dim=int(self.db_stage1_embeddings.shape[1]), metric=self.metric)
        self.index.add(np.asarray(self.db_stage1_embeddings, dtype=np.float32))

    def retrieval_config_hash(self) -> str:
        payload = {
            "cache_schema_version": RETRIEVAL_CACHE_SCHEMA_VERSION,
            "retrieval_config_path": str(self.retrieval_config_path),
            "rag_config": self.rag_config,
            "db_splits": self.db_splits,
            "query_splits": self.query_splits,
            "top_m": self.top_m,
            "top_k": self.top_k,
            "same_cell_policy": self.same_cell_policy,
            "exclude_query_case": self.exclude_query_case,
            "allow_cross_chemistry": self.allow_cross_chemistry,
            "metric": self.metric,
            "stage1_embedding_name": self.stage1_embedding_name,
            "retrieval_temperature": self.retrieval_temperature,
            "use_mmr": self.use_mmr,
            "mmr_lambda": self.mmr_lambda,
            "max_neighbors_per_cell": self.max_neighbors_per_cell,
            "max_neighbors_per_domain": self.max_neighbors_per_domain,
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _reference_quality_payload(
        neighbor_future_delta_soh: np.ndarray,
        retrieval_alpha: np.ndarray,
        retrieval_mask: np.ndarray,
        neighbor_cell_uids: Sequence[str] | None = None,
    ) -> Dict[str, float]:
        """Summarize whether selected top-k reference futures agree.

        Only historical reference labels are used here; the query future target
        is never accessed. The resulting diagnostics make confidence lower when
        the top-k references have similar composite distances but disagree about
        the future delta-SOH trajectory.
        """

        mask = np.asarray(retrieval_mask, dtype=np.float32) > 0
        if not np.any(mask):
            return {"future_delta_dispersion": 0.0, "alpha_entropy": 1.0, "unique_cell_ratio": 0.0}
        futures = np.asarray(neighbor_future_delta_soh, dtype=np.float32)[mask]
        alpha = np.asarray(retrieval_alpha, dtype=np.float32)[mask]
        alpha_sum = float(np.sum(alpha))
        if alpha_sum <= 0:
            alpha = np.full(futures.shape[0], 1.0 / max(futures.shape[0], 1), dtype=np.float32)
        else:
            alpha = alpha / alpha_sum
        weighted_mean = np.sum(futures * alpha[:, None], axis=0)
        weighted_var = np.sum(((futures - weighted_mean[None, :]) ** 2) * alpha[:, None], axis=0)
        future_delta_dispersion = float(np.sqrt(np.maximum(weighted_var, 0.0)).mean())
        if alpha.size <= 1:
            alpha_entropy = 0.0
        else:
            alpha_entropy = float(-np.sum(alpha * np.log(np.clip(alpha, 1e-8, 1.0))) / np.log(alpha.size))
        if neighbor_cell_uids:
            valid_cells = [str(cell) for idx, cell in enumerate(neighbor_cell_uids) if idx < len(mask) and bool(mask[idx])]
            unique_cell_ratio = float(len(set(valid_cells)) / max(len(valid_cells), 1))
        else:
            unique_cell_ratio = 1.0
        return {
            "future_delta_dispersion": future_delta_dispersion,
            "alpha_entropy": alpha_entropy,
            "unique_cell_ratio": unique_cell_ratio,
        }

    def _cycle_feature_index(self, name: str) -> int | None:
        names = list(self.feature_names.get("cycle_stats", []))
        try:
            return names.index(name)
        except ValueError:
            return None

    def _qv_summary_stats(self, idx: int) -> Dict[str, float]:
        cycle_last = np.asarray(self.arrays["cycle_stats"][idx, -1], dtype=np.float32)
        stats: Dict[str, float] = {}
        for name in [
            "qv_dqdv_peak_value",
            "qv_dqdv_peak_voltage",
            "qv_dqdv_area",
            "qv_capacity_span",
            "qv_power_energy_proxy",
            "dq_dv_peak_value",
            "dq_dv_peak_position",
            "q_total",
        ]:
            feature_idx = self._cycle_feature_index(name)
            if feature_idx is not None and feature_idx < cycle_last.shape[0]:
                stats[name] = float(cycle_last[feature_idx])
        if "qv_dqdv_peak_value" not in stats and "dq_dv_peak_value" in stats:
            stats["qv_dqdv_peak_value"] = stats["dq_dv_peak_value"]
        if "qv_dqdv_peak_voltage" not in stats and "dq_dv_peak_position" in stats:
            stats["qv_dqdv_peak_voltage"] = stats["dq_dv_peak_position"]
        if "qv_capacity_span" not in stats and "q_total" in stats:
            stats["qv_capacity_span"] = stats["q_total"]
        return stats

    def _state_dict(self, idx: int) -> Dict[str, object]:
        row = self.case_rows.iloc[idx]
        lookback = max(int(row.get("lookback_length", self.arrays["soh_seq"].shape[1])), 1)
        return {
            "anchor_soh": float(row.get("anchor_soh", 0.0)),
            "recent_soh_slope": float(row.get("recent_soh_slope", 0.0)),
            "recent_soh_curvature": float(row.get("recent_soh_curvature", 0.0)),
            "degradation_stage": str(row.get("degradation_stage", "unknown")),
            "normalized_cycle_index": float(int(row.get("window_end", 0)) / max(lookback, 1)),
            "soh_seq": np.asarray(self.arrays["soh_seq"][idx], dtype=np.float32),
        }

    def _qv_dict(self, idx: int) -> Dict[str, object]:
        return {
            "qv_map": np.asarray(self.arrays["qv_maps"][idx, -1], dtype=np.float32),
            "qv_mask": np.asarray(self.arrays["qv_masks"][idx, -1], dtype=np.float32),
            "qv_summary_stats": self._qv_summary_stats(idx),
        }

    def _physics_dict(self, idx: int) -> Dict[str, object]:
        return {
            "physics_features": np.asarray(self.arrays["anchor_physics_features"][idx], dtype=np.float32),
            "physics_feature_mask": np.asarray(self.arrays["physics_feature_masks"][idx, -1], dtype=np.float32),
        }

    def _operation_feature_dict(self, idx: int) -> Dict[str, object]:
        """Return a disabled compatibility payload for the legacy d_operation field.

        Raw operating signals used for retrieval are represented in `_metadata_dict`
        as charge/discharge current, temperature and normalized-capacity-change
        sequences.
        """

        return {"operation_mask": {}}

    def _metadata_dict(self, idx: int) -> Dict[str, object]:
        row = self.case_rows.iloc[idx]
        qv_maps = np.asarray(self.arrays["qv_maps"][idx], dtype=np.float32)
        qv_masks = np.asarray(self.arrays["qv_masks"][idx], dtype=np.float32)
        op_seq = np.asarray(self.arrays["operation_seq"][idx], dtype=np.float32)
        op_names = list(self.feature_names.get("operation", []))
        op_name_to_series = {name: op_seq[:, pos] for pos, name in enumerate(op_names[: op_seq.shape[1]])}
        soh_seq = np.asarray(self.arrays["soh_seq"][idx], dtype=np.float32).reshape(-1)

        def _qv_channel_sequence(channel_idx: int, *, absolute: bool = False) -> np.ndarray:
            if qv_maps.ndim != 3 or qv_maps.shape[1] <= channel_idx:
                return np.zeros(0, dtype=np.float32)
            values = qv_maps[:, channel_idx, :]
            if absolute:
                values = np.abs(values)
            channel_mask = (
                qv_masks[:, channel_idx] > 0
                if qv_masks.ndim == 2 and qv_masks.shape[1] > channel_idx
                else np.ones(values.shape[0], dtype=bool)
            )
            seq = np.full(values.shape[0], np.nan, dtype=np.float32)
            for cycle_pos in range(values.shape[0]):
                if not bool(channel_mask[cycle_pos]):
                    continue
                finite = values[cycle_pos][np.isfinite(values[cycle_pos])]
                if finite.size:
                    seq[cycle_pos] = float(np.mean(finite))
            return seq

        def _operation_sequence(name: str) -> np.ndarray:
            series = op_name_to_series.get(name)
            if series is None:
                return np.zeros(0, dtype=np.float32)
            return np.asarray(series, dtype=np.float32).reshape(-1)

        capacity_delta = np.zeros_like(soh_seq, dtype=np.float32)
        if soh_seq.size > 1:
            capacity_delta[1:] = np.diff(soh_seq)
            capacity_delta[~np.isfinite(capacity_delta)] = np.nan
        return {
            "chemistry_family": row.get("chemistry_family", "unknown"),
            "source_dataset": row.get("source_dataset", "unknown"),
            "domain_label": row.get("domain_label", "unknown"),
            "voltage_window_bucket": row.get("voltage_window_bucket", "unknown"),
            "nominal_capacity_bucket": row.get("nominal_capacity_bucket", "unknown"),
            "temperature_bucket": row.get("temperature_bucket", "unknown"),
            "charge_rate_bucket": row.get("charge_rate_bucket", "unknown"),
            "current_abs_seq": _qv_channel_sequence(QV_CHANNEL_TO_INDEX["Id_abs"], absolute=True),
            "temperature_seq": _operation_sequence("temp_mean"),
            "power_energy_seq": _qv_channel_sequence(QV_CHANNEL_TO_INDEX["Power_abs"], absolute=True),
            "normalized_capacity_delta_seq": capacity_delta,
        }

    def _metadata_numeric_summary(self, idx: int) -> np.ndarray:
        qv_maps = np.asarray(self.arrays["qv_maps"][idx], dtype=np.float32)
        qv_masks = np.asarray(self.arrays["qv_masks"][idx], dtype=np.float32)
        op_seq = np.asarray(self.arrays["operation_seq"][idx], dtype=np.float32)
        op_names = list(self.feature_names.get("operation", []))
        op_name_to_pos = {name: pos for pos, name in enumerate(op_names[: op_seq.shape[1]])}
        soh_seq = np.asarray(self.arrays["soh_seq"][idx], dtype=np.float32).reshape(-1)

        def _qv_mean_sequence(channel_idx: int) -> np.ndarray:
            if qv_maps.ndim != 3 or qv_maps.shape[1] <= channel_idx:
                return np.zeros(0, dtype=np.float32)
            values = np.abs(qv_maps[:, channel_idx, :]).astype(np.float32)
            channel_mask = (
                qv_masks[:, channel_idx] > 0
                if qv_masks.ndim == 2 and qv_masks.shape[1] > channel_idx
                else np.ones(values.shape[0], dtype=bool)
            )
            values = np.where(np.isfinite(values), values, np.nan)
            seq = np.nanmean(values, axis=1).astype(np.float32)
            seq[~channel_mask] = np.nan
            return seq

        def _operation_sequence(name: str) -> np.ndarray:
            pos = op_name_to_pos.get(name)
            if pos is None:
                return np.zeros(0, dtype=np.float32)
            return np.asarray(op_seq[:, pos], dtype=np.float32).reshape(-1)

        capacity_delta = np.zeros_like(soh_seq, dtype=np.float32)
        if soh_seq.size > 1:
            capacity_delta[1:] = np.diff(soh_seq)
            capacity_delta[~np.isfinite(capacity_delta)] = np.nan
        metadata = {
            "current_abs_seq": _qv_mean_sequence(QV_CHANNEL_TO_INDEX["Id_abs"]),
            "temperature_seq": _operation_sequence("temp_mean"),
            "power_energy_seq": _qv_mean_sequence(QV_CHANNEL_TO_INDEX["Power_abs"]),
            "normalized_capacity_delta_seq": capacity_delta,
        }
        values: list[float] = []
        for name in ["current_abs_seq", "temperature_seq", "power_energy_seq", "normalized_capacity_delta_seq"]:
            seq = np.asarray(metadata.get(name, []), dtype=np.float32).reshape(-1)
            finite = seq[np.isfinite(seq)]
            if finite.size:
                values.extend([float(np.mean(finite)), float(np.std(finite)), float(finite[-1])])
            else:
                values.extend([0.0, 0.0, 0.0])
        return np.asarray(values, dtype=np.float32)

    def _metadata_numeric_matrix(self, chunk_size: int = 1024) -> np.ndarray:
        """Build Stage-1 numeric metadata summaries for all cases.

        The 12 dimensions are mean/std/last for absolute-current sequence,
        temperature sequence, power/energy proxy sequence and normalized SOH
        delta sequence. This is equivalent to `_metadata_numeric_summary`, but
        uses chunked NumPy operations to avoid slow per-row parquet/pandas access.
        """

        qv_maps = np.asarray(self.arrays["qv_maps"], dtype=np.float32)
        qv_masks = np.asarray(self.arrays["qv_masks"], dtype=np.float32)
        op_seq = np.asarray(self.arrays["operation_seq"], dtype=np.float32)
        soh_seq = np.asarray(self.arrays["soh_seq"], dtype=np.float32)
        n_cases = int(soh_seq.shape[0])
        result = np.zeros((n_cases, 12), dtype=np.float32)
        op_names = list(self.feature_names.get("operation", []))
        op_name_to_pos = {name: pos for pos, name in enumerate(op_names[: op_seq.shape[2]])}
        temp_pos = op_name_to_pos.get("temp_mean")

        def _summarize_sequence(seq: np.ndarray) -> np.ndarray:
            finite = np.isfinite(seq)
            count = finite.sum(axis=1).astype(np.float32)
            safe_seq = np.where(finite, seq, 0.0).astype(np.float32)
            mean = safe_seq.sum(axis=1) / np.maximum(count, 1.0)
            centered = np.where(finite, seq - mean[:, None], 0.0)
            std = np.sqrt((centered * centered).sum(axis=1) / np.maximum(count, 1.0)).astype(np.float32)
            reversed_seq = np.where(finite, seq, np.nan)[:, ::-1]
            last = np.zeros(seq.shape[0], dtype=np.float32)
            has_any = finite.any(axis=1)
            if np.any(has_any):
                last_pos = np.argmax(np.isfinite(reversed_seq[has_any]), axis=1)
                last[has_any] = reversed_seq[has_any, last_pos].astype(np.float32)
            return np.stack([mean, std, last], axis=1).astype(np.float32)

        for start in range(0, n_cases, int(chunk_size)):
            end = min(start + int(chunk_size), n_cases)
            chunk_maps = qv_maps[start:end]
            chunk_masks = qv_masks[start:end]
            sequences: list[np.ndarray] = []
            for channel_idx in [QV_CHANNEL_TO_INDEX["Id_abs"]]:
                values = np.abs(chunk_maps[:, :, channel_idx, :]).astype(np.float32)
                values = np.where(np.isfinite(values), values, np.nan)
                seq = np.nanmean(values, axis=2).astype(np.float32)
                if chunk_masks.ndim == 3 and chunk_masks.shape[2] > channel_idx:
                    seq = np.where(chunk_masks[:, :, channel_idx] > 0, seq, np.nan).astype(np.float32)
                sequences.append(seq)

            if temp_pos is not None:
                sequences.append(op_seq[start:end, :, temp_pos].astype(np.float32))
            else:
                sequences.append(np.full((end - start, soh_seq.shape[1]), np.nan, dtype=np.float32))

            power_idx = QV_CHANNEL_TO_INDEX["Power_abs"]
            if qv_maps.shape[2] > power_idx:
                power_values = np.abs(chunk_maps[:, :, power_idx, :]).astype(np.float32)
                power_values = np.where(np.isfinite(power_values), power_values, np.nan)
                power_seq = np.nanmean(power_values, axis=2).astype(np.float32)
                if chunk_masks.ndim == 3 and chunk_masks.shape[2] > power_idx:
                    power_seq = np.where(chunk_masks[:, :, power_idx] > 0, power_seq, np.nan).astype(np.float32)
                sequences.append(power_seq)
            else:
                sequences.append(np.full((end - start, soh_seq.shape[1]), np.nan, dtype=np.float32))

            capacity_delta = np.zeros_like(soh_seq[start:end], dtype=np.float32)
            if capacity_delta.shape[1] > 1:
                capacity_delta[:, 1:] = np.diff(soh_seq[start:end], axis=1)
                capacity_delta[~np.isfinite(capacity_delta)] = np.nan
            sequences.append(capacity_delta)
            result[start:end] = np.concatenate([_summarize_sequence(seq) for seq in sequences], axis=1)
        return result

    def _feature_availability_ratio(self, query_idx: int, ref_idx: int) -> float:
        q_qv = np.asarray(self.arrays["qv_masks"][query_idx, -1], dtype=np.float32)
        r_qv = np.asarray(self.arrays["qv_masks"][ref_idx, -1], dtype=np.float32)
        q_phy = np.asarray(self.arrays["physics_feature_masks"][query_idx, -1], dtype=np.float32)
        r_phy = np.asarray(self.arrays["physics_feature_masks"][ref_idx, -1], dtype=np.float32)
        q_metadata = self._metadata_dict(query_idx)
        r_metadata = self._metadata_dict(ref_idx)
        metadata_keys = ["current_abs_seq", "temperature_seq", "power_energy_seq", "normalized_capacity_delta_seq"]
        metadata_common = np.asarray(
            [
                1.0
                if np.isfinite(np.asarray(q_metadata.get(key, []), dtype=np.float32)).any()
                and np.isfinite(np.asarray(r_metadata.get(key, []), dtype=np.float32)).any()
                else 0.0
                for key in metadata_keys
            ],
            dtype=np.float32,
        )
        availability = np.concatenate(
            [
                ((q_qv > 0) & (r_qv > 0)).astype(np.float32),
                ((q_phy > 0) & (r_phy > 0)).astype(np.float32),
                metadata_common,
            ]
        )
        return float(availability.mean()) if availability.size else 0.0

    def _reference_compatibility_score(self, query_idx: int, ref_idx: int, component_distances: Dict[str, float], missing_penalty: float) -> float:
        query_row = self.case_rows.iloc[query_idx]
        ref_row = self.case_rows.iloc[ref_idx]
        chemistry_bonus = 0.1 if str(query_row["chemistry_family"]) == str(ref_row["chemistry_family"]) else -0.05
        domain_bonus = 0.05 if str(query_row["domain_label"]) == str(ref_row["domain_label"]) else 0.0
        stage_bonus = 0.1 if str(query_row["degradation_stage"]) == str(ref_row["degradation_stage"]) else -0.05
        core_values = [
            float(component_distances[name])
            for name in CORE_COMPONENTS
            if np.isfinite(float(component_distances.get(name, np.nan)))
        ]
        core_mean = float(np.mean(core_values)) if core_values else 1.0
        raw_score = 1.0 - core_mean + chemistry_bonus + domain_bonus + stage_bonus - 0.2 * float(missing_penalty)
        return float(np.clip(raw_score, 0.0, 1.0))

    def _build_handcrafted_embeddings(self) -> np.ndarray:
        cache_path = self.case_bank_dir / "case_handcrafted_retrieval_embedding.npy"
        history_points = int(dict(self.rag_config.get("soh_state", {}) or {}).get("stage1_history_points", 16))
        expected_dim = 3 + history_points + 4 + int(np.asarray(self.arrays["anchor_physics_features"]).shape[1]) + 12
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (len(self.case_rows), expected_dim):
                return np.asarray(cached, dtype=np.float32)

        state = self.case_rows[["anchor_soh", "recent_soh_slope", "recent_soh_curvature"]].fillna(0.0).to_numpy(dtype=np.float32)
        soh_seq = np.asarray(self.arrays["soh_seq"], dtype=np.float32)
        anchored_history = soh_seq - soh_seq[:, -1:]
        if history_points > 0:
            source_positions = np.linspace(0, max(soh_seq.shape[1] - 1, 0), soh_seq.shape[1], dtype=np.float32)
            target_positions = np.linspace(0, max(soh_seq.shape[1] - 1, 0), history_points, dtype=np.float32)
            history_vec = np.stack(
                [
                    np.interp(target_positions, source_positions, np.nan_to_num(row, nan=0.0)).astype(np.float32)
                    for row in anchored_history
                ],
                axis=0,
            )
        else:
            history_vec = np.zeros((anchored_history.shape[0], 0), dtype=np.float32)
        cycle_last = np.asarray(self.arrays["cycle_stats"][:, -1, :], dtype=np.float32)
        qv_columns = []
        for name, fallback in [
            ("qv_dqdv_peak_value", "dq_dv_peak_value"),
            ("qv_dqdv_peak_voltage", "dq_dv_peak_position"),
            ("qv_dqdv_area", None),
            ("qv_capacity_span", "q_total"),
        ]:
            feature_idx = self._cycle_feature_index(name)
            if feature_idx is None and fallback is not None:
                feature_idx = self._cycle_feature_index(fallback)
            if feature_idx is not None and feature_idx < cycle_last.shape[1]:
                qv_columns.append(cycle_last[:, feature_idx])
            else:
                qv_columns.append(np.zeros(cycle_last.shape[0], dtype=np.float32))
        qv_vec = np.stack(qv_columns, axis=1).astype(np.float32)
        physics = np.asarray(self.arrays["anchor_physics_features"], dtype=np.float32)
        metadata_numeric = self._metadata_numeric_matrix()
        embedding = np.concatenate([state, history_vec, qv_vec, physics, metadata_numeric], axis=1).astype(np.float32)
        np.save(cache_path, embedding)
        return embedding

    def _build_stage1_embeddings(self) -> np.ndarray:
        raw = np.asarray(self.handcrafted_embeddings, dtype=np.float32)
        if raw.size == 0 or len(self.db_indices) == 0:
            return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        db_values = raw[self.db_indices]
        center = np.nanmedian(db_values, axis=0)
        q25 = np.nanpercentile(db_values, 25, axis=0)
        q75 = np.nanpercentile(db_values, 75, axis=0)
        scale = q75 - q25
        fallback_scale = np.nanstd(db_values, axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback_scale)
        scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
        standardized = (raw - center[None, :]) / scale[None, :]
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=8.0, neginf=-8.0)
        return np.clip(standardized, -8.0, 8.0).astype(np.float32)

    def _coarse_candidates(self, query_idx: int) -> np.ndarray:
        if len(self.db_case_ids) == 0:
            return np.zeros(0, dtype=np.int64)
        query_embedding = np.asarray(self.stage1_embeddings[query_idx : query_idx + 1], dtype=np.float32)
        fetch = min(max(self.top_m * 4, self.top_k * 8), len(self.db_case_ids))
        if self.index is not None:
            _, index_positions = self.index.search(query_embedding, fetch)
            return self.db_case_ids[index_positions[0]].astype(np.int64)
        distances = np.linalg.norm(self.db_stage1_embeddings - query_embedding[0], axis=1)
        return self.db_case_ids[np.argsort(distances)[:fetch]].astype(np.int64)

    def _hard_filter_candidate_ids(self, query_idx: int, candidate_case_ids: np.ndarray) -> np.ndarray:
        query_row = self.case_rows.iloc[query_idx]
        query_case_id = int(query_row["case_id"])
        query_horizon = int(query_row["target_horizon"])
        valid_ids = []
        for case_id in np.asarray(candidate_case_ids, dtype=np.int64).tolist():
            ref_idx = self.case_id_to_index[int(case_id)]
            ref_row = self.case_rows.iloc[ref_idx]
            if self.exclude_query_case and int(ref_row["case_id"]) == query_case_id:
                continue
            if str(ref_row["split"]) == "target_query":
                continue
            if int(ref_row["target_horizon"]) != query_horizon:
                continue
            ref_future = np.asarray(self.arrays["future_delta"][ref_idx], dtype=np.float32)
            if ref_future.size == 0 or not np.isfinite(ref_future).all():
                continue
            if self.same_cell_policy == "exclude" and str(ref_row["cell_uid"]) == str(query_row["cell_uid"]):
                continue
            if self.same_cell_policy == "past_only" and str(ref_row["cell_uid"]) == str(query_row["cell_uid"]):
                if int(ref_row["window_end"]) > int(query_row["window_end"]):
                    continue
            if not self.allow_cross_chemistry and str(ref_row["chemistry_family"]) != str(query_row["chemistry_family"]):
                continue
            valid_ids.append(int(case_id))
        return np.asarray(valid_ids, dtype=np.int64)

    def _distance_bundle(self, query_idx: int, ref_idx: int) -> Dict[str, float]:
        query_state = self._state_dict(query_idx)
        ref_state = self._state_dict(ref_idx)
        query_qv = self._qv_dict(query_idx)
        ref_qv = self._qv_dict(ref_idx)
        query_physics = self._physics_dict(query_idx)
        ref_physics = self._physics_dict(ref_idx)
        query_operation = self._operation_feature_dict(query_idx)
        ref_operation = self._operation_feature_dict(ref_idx)
        query_metadata = self._metadata_dict(query_idx)
        ref_metadata = self._metadata_dict(ref_idx)
        d_soh_state = compute_soh_state_distance(
            query_state,
            ref_state,
            self.rag_config,
        )
        d_qv_shape = compute_qv_shape_distance(
            query_qv,
            ref_qv,
            self.rag_config,
        )
        d_physics = compute_physics_distance(
            query_physics,
            ref_physics,
            self.rag_config,
        )
        d_operation = (
            compute_operation_distance(query_operation, ref_operation, self.rag_config)
            if "d_operation" in self.enabled_component_names
            else np.nan
        )
        d_metadata = compute_metadata_distance(
            query_metadata,
            ref_metadata,
            self.rag_config,
        )
        components = {
            "d_soh_state": float(d_soh_state),
            "d_qv_shape": float(d_qv_shape),
            "d_physics": float(d_physics),
            "d_operation": float(d_operation),
            "d_metadata": float(d_metadata),
        }
        composite_distance = compute_composite_distance(components, self.rag_config)
        stage_distance = degradation_stage_distance(
            str(query_state.get("degradation_stage", "unknown")),
            str(ref_state.get("degradation_stage", "unknown")),
        )
        feature_availability_ratio = self._feature_availability_ratio(query_idx, ref_idx)
        missing_penalty = float(1.0 - feature_availability_ratio)
        reference_compatibility_score = self._reference_compatibility_score(query_idx, ref_idx, components, missing_penalty)
        return {
            **components,
            "composite_distance": float(composite_distance),
            "stage_distance": float(stage_distance),
            "missing_penalty": float(missing_penalty),
            "feature_availability_ratio": float(feature_availability_ratio),
            "reference_compatibility_score": float(reference_compatibility_score),
            "chemistry_match": float(self.case_rows.iloc[query_idx]["chemistry_family"] == self.case_rows.iloc[ref_idx]["chemistry_family"]),
            "domain_match": float(self.case_rows.iloc[query_idx]["domain_label"] == self.case_rows.iloc[ref_idx]["domain_label"]),
        }

    def _apply_mmr(self, candidate_case_ids: np.ndarray, composite_distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_mmr or len(candidate_case_ids) <= self.top_k:
            scores = np.ones(min(len(candidate_case_ids), self.top_k), dtype=np.float32)
            return candidate_case_ids[: self.top_k], scores

        selected: List[int] = []
        mmr_scores: List[float] = []
        cell_counter: Dict[str, int] = {}
        domain_counter: Dict[str, int] = {}
        candidate_order = list(np.argsort(composite_distance))

        while candidate_order and len(selected) < self.top_k:
            best_case = None
            best_score = None
            for pos in candidate_order:
                case_id = int(candidate_case_ids[pos])
                ref_idx = self.case_id_to_index[case_id]
                ref_row = self.case_rows.iloc[ref_idx]
                cell_uid = str(ref_row["cell_uid"])
                domain_label = str(ref_row["domain_label"])
                if cell_counter.get(cell_uid, 0) >= self.max_neighbors_per_cell:
                    continue
                if self.max_neighbors_per_domain is not None and domain_counter.get(domain_label, 0) >= int(self.max_neighbors_per_domain):
                    continue
                relevance = -float(composite_distance[pos])
                diversity = 0.0
                if selected:
                    similarities = []
                    ref_embedding = self.stage1_embeddings[ref_idx]
                    for selected_case in selected:
                        sel_idx = self.case_id_to_index[selected_case]
                        similarities.append(_cosine_similarity(ref_embedding, self.stage1_embeddings[sel_idx]))
                    diversity = max(similarities) if similarities else 0.0
                score = self.mmr_lambda * relevance - (1.0 - self.mmr_lambda) * diversity
                if best_score is None or score > best_score:
                    best_case = case_id
                    best_score = score
            if best_case is None:
                break
            selected.append(int(best_case))
            mmr_scores.append(float(best_score))
            chosen_row = self.case_rows.iloc[self.case_id_to_index[int(best_case)]]
            cell_counter[str(chosen_row["cell_uid"])] = cell_counter.get(str(chosen_row["cell_uid"]), 0) + 1
            domain_counter[str(chosen_row["domain_label"])] = domain_counter.get(str(chosen_row["domain_label"]), 0) + 1
            candidate_order = [pos for pos in candidate_order if int(candidate_case_ids[pos]) != int(best_case)]
        return np.asarray(selected, dtype=np.int64), np.asarray(mmr_scores, dtype=np.float32)

    def _fast_hard_filter_indices(self, query_idx: int, candidate_case_ids: np.ndarray) -> np.ndarray:
        if candidate_case_ids.size == 0:
            return np.zeros(0, dtype=np.int64)
        candidate_indices = np.asarray([self.case_id_to_index[int(case_id)] for case_id in candidate_case_ids.tolist()], dtype=np.int64)
        mask = (
            self._row_split[candidate_indices] != "target_query"
        ) & (self._row_target_horizon[candidate_indices] == self._row_target_horizon[query_idx]) & self._future_delta_finite[candidate_indices]
        if self.exclude_query_case:
            mask &= self._row_case_ids[candidate_indices] != self._row_case_ids[query_idx]
        if self.same_cell_policy == "exclude":
            mask &= self._row_cell_uid[candidate_indices] != self._row_cell_uid[query_idx]
        elif self.same_cell_policy == "past_only":
            same_cell = self._row_cell_uid[candidate_indices] == self._row_cell_uid[query_idx]
            mask &= (~same_cell) | (self._row_window_end[candidate_indices] <= self._row_window_end[query_idx])
        if not self.allow_cross_chemistry:
            mask &= self._row_chemistry[candidate_indices] == self._row_chemistry[query_idx]
        return candidate_indices[mask][: self.top_m]

    def _fast_component_arrays(self, query_idx: int, ref_indices: np.ndarray) -> Dict[str, np.ndarray]:
        if ref_indices.size == 0:
            return {name: np.zeros(0, dtype=np.float32) for name in COMPONENT_NAMES + ["composite_distance"]}

        q_state = self._state_matrix[query_idx]
        r_state = self._state_matrix[ref_indices]
        state_scale = np.asarray([0.1, 0.01, 0.005], dtype=np.float32)
        d_state_numeric = np.sqrt(np.mean(((r_state - q_state[None, :]) / state_scale[None, :]) ** 2, axis=1))
        d_stage = (self._row_degradation_stage[ref_indices] != self._row_degradation_stage[query_idx]).astype(np.float32)
        soh_cfg = dict(self.rag_config.get("soh_state", {}) or {})
        recency_power = float(soh_cfg.get("recency_weight_power", 1.5))
        history_scale = max(float(soh_cfg.get("history_shape_scale", 0.015)), 1e-8)
        slope_scale = max(float(soh_cfg.get("history_slope_scale", 0.0008)), 1e-8)
        q_hist = self._soh_history_delta_matrix[query_idx]
        r_hist = self._soh_history_delta_matrix[ref_indices]
        hist_mask = np.isfinite(r_hist) & np.isfinite(q_hist[None, :])
        hist_weights = np.linspace(1.0 / q_hist.shape[0], 1.0, q_hist.shape[0], dtype=np.float32) ** max(recency_power, 0.0)
        hist_weights = hist_weights / max(float(np.mean(hist_weights)), 1e-6)
        hist_diff = ((r_hist - q_hist[None, :]) / history_scale) ** 2
        hist_weighted = hist_diff * hist_mask * hist_weights[None, :]
        hist_denom = np.maximum((hist_mask * hist_weights[None, :]).sum(axis=1), 1e-6)
        d_history_shape = np.sqrt(hist_weighted.sum(axis=1) / hist_denom).astype(np.float32)

        q_slope = self._soh_history_slope_matrix[query_idx]
        r_slope = self._soh_history_slope_matrix[ref_indices]
        slope_mask = np.isfinite(r_slope) & np.isfinite(q_slope[None, :])
        slope_weights = np.linspace(1.0 / max(q_slope.shape[0], 1), 1.0, q_slope.shape[0], dtype=np.float32) ** max(recency_power, 0.0)
        slope_weights = slope_weights / max(float(np.mean(slope_weights)), 1e-6)
        slope_diff = ((r_slope - q_slope[None, :]) / slope_scale) ** 2
        slope_weighted = slope_diff * slope_mask * slope_weights[None, :]
        slope_denom = np.maximum((slope_mask * slope_weights[None, :]).sum(axis=1), 1e-6)
        d_history_slope = np.sqrt(slope_weighted.sum(axis=1) / slope_denom).astype(np.float32)

        current_weight = max(float(soh_cfg.get("current_state_weight", 0.30)), 0.0)
        history_weight = max(float(soh_cfg.get("history_shape_weight", 0.50)), 0.0)
        slope_weight = max(float(soh_cfg.get("history_slope_weight", 0.15)), 0.0)
        stage_weight = max(float(soh_cfg.get("stage_weight", 0.05)), 0.0)
        soh_weight_sum = max(current_weight + history_weight + slope_weight + stage_weight, 1e-6)
        d_soh_state = (
            current_weight * d_state_numeric
            + history_weight * d_history_shape
            + slope_weight * d_history_slope
            + stage_weight * d_stage
        ) / soh_weight_sum
        d_soh_state = d_soh_state.astype(np.float32)

        qv_maps = np.asarray(self.arrays["qv_maps"], dtype=np.float32)
        qv_masks = np.asarray(self.arrays["qv_masks"], dtype=np.float32)
        q_map = qv_maps[query_idx, -1]
        r_maps = qv_maps[ref_indices, -1]
        q_mask = qv_masks[query_idx, -1]
        r_masks = qv_masks[ref_indices, -1]
        qv_cfg = dict(self.rag_config.get("qv_shape", {}) or {})
        qv_weights = dict(qv_cfg.get("channel_weights", {}) or {"Qd": 0.35, "dQdV": 0.65})
        qv_terms = np.zeros(ref_indices.size, dtype=np.float32)
        qv_weight_sum = np.zeros(ref_indices.size, dtype=np.float32)
        for channel_name in list(qv_cfg.get("channels", ["Qd", "dQdV"])):
            channel_idx = QV_CHANNEL_TO_INDEX[channel_name]
            common = (q_mask[channel_idx] > 0) & (r_masks[:, channel_idx] > 0)
            if not np.any(common):
                continue
            q_curve = q_map[channel_idx].astype(np.float32)
            r_curves = r_maps[:, channel_idx, :].astype(np.float32)
            finite = np.isfinite(r_curves) & np.isfinite(q_curve[None, :])
            safe_r = np.where(finite, r_curves, np.nan)
            safe_q = np.where(np.isfinite(q_curve), q_curve, np.nan)
            q_mean = np.nanmean(safe_q).astype(np.float32)
            q_std = max(float(np.nanstd(safe_q)), 1e-4)
            ref_mean = np.nanmean(safe_r, axis=1).astype(np.float32)
            ref_std = np.maximum(np.nanstd(safe_r, axis=1).astype(np.float32), 1e-4)
            q_shape = (q_curve[None, :] - q_mean) / q_std
            ref_shape = (r_curves - ref_mean[:, None]) / ref_std[:, None]
            shape_diff = np.where(finite, (ref_shape - q_shape) ** 2, 0.0)
            denom = np.maximum(finite.sum(axis=1), 1)
            shape_dist = np.sqrt(shape_diff.sum(axis=1) / denom).astype(np.float32)
            level_scale = np.maximum.reduce([np.abs(ref_mean), np.full_like(ref_mean, abs(float(q_mean))), ref_std, np.full_like(ref_mean, q_std), np.full_like(ref_mean, 1e-4)])
            level_dist = np.abs(ref_mean - float(q_mean)) / level_scale
            channel_dist = (0.85 * shape_dist + 0.15 * level_dist).astype(np.float32)
            weight = float(qv_weights.get(channel_name, 1.0))
            qv_terms[common] += weight * channel_dist[common]
            qv_weight_sum[common] += weight
        q_summary = self._qv_summary_matrix[query_idx]
        r_summary = self._qv_summary_matrix[ref_indices]
        common_summary = (self._qv_summary_mask[query_idx][None, :] > 0) & (self._qv_summary_mask[ref_indices] > 0)
        if np.any(common_summary):
            scale = np.maximum(np.maximum(np.abs(r_summary), np.abs(q_summary[None, :])), 1e-4)
            summary_diff = ((r_summary - q_summary[None, :]) / scale) ** 2
            denom = np.maximum(common_summary.sum(axis=1), 1)
            summary_dist = np.sqrt((summary_diff * common_summary).sum(axis=1) / denom).astype(np.float32)
            has_summary = common_summary.any(axis=1)
            qv_terms[has_summary] += summary_dist[has_summary]
            qv_weight_sum[has_summary] += 1.0
        d_qv_shape = np.where(qv_weight_sum > 0, qv_terms / np.maximum(qv_weight_sum, 1e-6), 1.0).astype(np.float32)

        q_phys = self._physics_matrix[query_idx, :4]
        r_phys = self._physics_matrix[ref_indices, :4]
        q_phys_mask = self._physics_mask_matrix[query_idx, :4]
        r_phys_mask = self._physics_mask_matrix[ref_indices, :4]
        common_phys = (q_phys_mask[None, :] > 0) & (r_phys_mask > 0)
        phys_scale = np.maximum(np.maximum(np.abs(r_phys), np.abs(q_phys[None, :])), 1e-4)
        phys_diff = ((r_phys - q_phys[None, :]) / phys_scale) ** 2
        phys_denom = np.maximum(common_phys.sum(axis=1), 1)
        d_physics = np.where(common_phys.any(axis=1), np.sqrt((phys_diff * common_phys).sum(axis=1) / phys_denom), 1.0).astype(np.float32)

        penalties = dict(self.rag_config.get("metadata_penalties", {}) or {})
        meta_pairs = [
            (self._row_chemistry, "chemistry_family_mismatch", 1.0),
            (self._row_domain, "domain_label_mismatch", 0.5),
            (self._row_voltage_window, "voltage_window_mismatch", 0.5),
        ]
        categorical_score = np.zeros(ref_indices.size, dtype=np.float32)
        categorical_weight = 0.0
        for values, penalty_name, default_penalty in meta_pairs:
            penalty = float(penalties.get(penalty_name, default_penalty))
            categorical_score += penalty * (values[ref_indices] != values[query_idx]).astype(np.float32)
            categorical_weight += penalty
        categorical_distance = categorical_score / max(categorical_weight, 1e-6)
        q_meta = self._metadata_summary_matrix[query_idx]
        r_meta = self._metadata_summary_matrix[ref_indices]
        meta_scale = np.maximum(np.maximum(np.abs(r_meta), np.abs(q_meta[None, :])), 1e-4)
        numeric_distance = np.sqrt(np.mean(((r_meta - q_meta[None, :]) / meta_scale) ** 2, axis=1)).astype(np.float32)
        mix = dict(self.rag_config.get("metadata_distance_weights", {}) or {})
        categorical_mix = float(mix.get("categorical", 0.4))
        numeric_mix = float(mix.get("raw_numeric", 0.6))
        d_metadata = ((categorical_mix * categorical_distance + numeric_mix * numeric_distance) / max(categorical_mix + numeric_mix, 1e-6)).astype(np.float32)

        d_operation = np.full(ref_indices.size, np.nan, dtype=np.float32)
        components = {
            "d_soh_state": d_soh_state,
            "d_qv_shape": d_qv_shape,
            "d_physics": d_physics,
            "d_operation": d_operation,
            "d_metadata": d_metadata,
        }
        weights = []
        names = []
        for name in CORE_COMPONENTS:
            cfg = self.rag_config
            if "distance_components" in cfg:
                enabled = bool(dict(cfg.get("distance_components", {}).get(name, {}) or {}).get("enabled", False))
                weight = float(dict(cfg.get("distance_components", {}).get(name, {}) or {}).get("weight", 0.0))
            else:
                enabled = bool(cfg.get(name, False))
                weight = float(dict(cfg.get("weights", {}) or {}).get(name, 0.0))
            if enabled and weight > 0 and np.isfinite(components[name]).any():
                weights.append(weight)
                names.append(name)
        composite = np.zeros(ref_indices.size, dtype=np.float32)
        weight_sum = float(sum(weights))
        if weight_sum > 0:
            for name, weight in zip(names, weights):
                values = components[name]
                composite += (weight / weight_sum) * np.where(np.isfinite(values), values, 0.0).astype(np.float32)
        components["composite_distance"] = composite.astype(np.float32)
        return components

    def _empty_result(self, query_case_id: int) -> RetrievalResult:
        horizon = int(self.arrays["future_delta"].shape[1])
        zeros = np.zeros(self.top_k, dtype=np.float32)
        return RetrievalResult(
            query_case_id=int(query_case_id),
            neighbor_case_ids=np.full(self.top_k, -1, dtype=np.int64),
            retrieval_mask=zeros.copy(),
            d_soh_state=zeros.copy(),
            d_qv_shape=zeros.copy(),
            d_physics=zeros.copy(),
            d_operation=np.full(self.top_k, np.nan, dtype=np.float32),
            d_metadata=zeros.copy(),
            composite_distance=np.full(self.top_k, np.inf, dtype=np.float32),
            retrieval_confidence=0.0,
            retrieval_alpha=zeros.copy(),
            neighbor_future_delta_soh=np.zeros((self.top_k, horizon), dtype=np.float32),
            neighbor_metadata=[{} for _ in range(self.top_k)],
            explain_json={"warning": "No valid neighbors after hard filtering."},
            stage_distance=zeros.copy(),
            missing_penalty=zeros.copy(),
            reference_compatibility_score=zeros.copy(),
            mmr_diversity_score=zeros.copy(),
        )

    def _retrieve_fast(self, query_case_id: int) -> RetrievalResult:
        query_idx = self.case_id_to_index[int(query_case_id)]
        ref_indices = self._fast_hard_filter_indices(query_idx, self._coarse_candidates(query_idx))
        if ref_indices.size == 0:
            return self._empty_result(query_case_id)
        distances = self._fast_component_arrays(query_idx, ref_indices)
        order = np.argsort(distances["composite_distance"])[: max(self.top_m, self.top_k)]
        ref_indices = ref_indices[order]
        distances = {name: values[order] for name, values in distances.items()}

        selected_positions: list[int] = []
        cell_counter: Dict[str, int] = {}
        domain_counter: Dict[str, int] = {}
        for pos, ref_idx in enumerate(ref_indices.tolist()):
            cell_uid = str(self._row_cell_uid[ref_idx])
            domain_label = str(self._row_domain[ref_idx])
            if self.use_mmr and cell_counter.get(cell_uid, 0) >= self.max_neighbors_per_cell:
                continue
            if self.use_mmr and self.max_neighbors_per_domain is not None and domain_counter.get(domain_label, 0) >= int(self.max_neighbors_per_domain):
                continue
            selected_positions.append(pos)
            cell_counter[cell_uid] = cell_counter.get(cell_uid, 0) + 1
            domain_counter[domain_label] = domain_counter.get(domain_label, 0) + 1
            if len(selected_positions) >= self.top_k:
                break
        if not selected_positions:
            selected_positions = list(range(min(self.top_k, ref_indices.size)))

        selected_ref_indices = ref_indices[np.asarray(selected_positions, dtype=np.int64)]
        k = int(min(selected_ref_indices.size, self.top_k))
        horizon = int(self.arrays["future_delta"].shape[1])
        neighbor_case_ids = np.full(self.top_k, -1, dtype=np.int64)
        retrieval_mask = np.zeros(self.top_k, dtype=np.float32)
        composite_distance = np.full(self.top_k, np.inf, dtype=np.float32)
        d_soh_state = np.zeros(self.top_k, dtype=np.float32)
        d_qv_shape = np.zeros(self.top_k, dtype=np.float32)
        d_physics = np.zeros(self.top_k, dtype=np.float32)
        d_operation = np.full(self.top_k, np.nan, dtype=np.float32)
        d_metadata = np.zeros(self.top_k, dtype=np.float32)
        stage_distance = np.zeros(self.top_k, dtype=np.float32)
        missing_penalty = np.zeros(self.top_k, dtype=np.float32)
        reference_compatibility_score = np.zeros(self.top_k, dtype=np.float32)
        mmr_diversity_score = np.ones(self.top_k, dtype=np.float32)
        neighbor_future_delta_soh = np.zeros((self.top_k, horizon), dtype=np.float32)
        retrieval_alpha = np.zeros(self.top_k, dtype=np.float32)
        neighbor_metadata = [{} for _ in range(self.top_k)]

        if k > 0:
            pos_arr = np.asarray(selected_positions[:k], dtype=np.int64)
            neighbor_case_ids[:k] = self._row_case_ids[selected_ref_indices[:k]]
            retrieval_mask[:k] = 1.0
            d_soh_state[:k] = distances["d_soh_state"][pos_arr]
            d_qv_shape[:k] = distances["d_qv_shape"][pos_arr]
            d_physics[:k] = distances["d_physics"][pos_arr]
            d_operation[:k] = distances["d_operation"][pos_arr]
            d_metadata[:k] = distances["d_metadata"][pos_arr]
            composite_distance[:k] = distances["composite_distance"][pos_arr]
            stage_distance[:k] = (self._row_degradation_stage[selected_ref_indices[:k]] != self._row_degradation_stage[query_idx]).astype(np.float32)
            reference_compatibility_score[:k] = np.clip(1.0 - composite_distance[:k], 0.0, 1.0)
            retrieval_alpha[:k] = _softmax_masked(composite_distance[:k], retrieval_mask[:k], self.retrieval_temperature)
            neighbor_future_delta_soh[:k] = np.asarray(self.arrays["future_delta"][selected_ref_indices[:k]], dtype=np.float32)
            for pos, ref_idx in enumerate(selected_ref_indices[:k].tolist()):
                neighbor_metadata[pos] = {
                    "case_id": int(self._row_case_ids[ref_idx]),
                    "cell_uid": str(self._row_cell_uid[ref_idx]),
                    "chemistry_family": str(self._row_chemistry[ref_idx]),
                    "domain_label": str(self._row_domain[ref_idx]),
                    "source_dataset": str(self.case_rows.iloc[ref_idx].get("source_dataset", "unknown")),
                    "degradation_stage": str(self._row_degradation_stage[ref_idx]),
                    "anchor_soh": float(self._state_matrix[ref_idx, 0]),
                    "recent_soh_slope": float(self._state_matrix[ref_idx, 1]),
                    "voltage_window_bucket": str(self._row_voltage_window[ref_idx]),
                }
        confidence_payload = {
            "composite_distance": composite_distance[:k],
            "feature_availability_ratio": 1.0,
            "chemistry_match_rate": float(np.mean(self._row_chemistry[selected_ref_indices[:k]] == self._row_chemistry[query_idx])) if k else 0.0,
            "domain_match_rate": float(np.mean(self._row_domain[selected_ref_indices[:k]] == self._row_domain[query_idx])) if k else 0.0,
        }
        confidence_payload.update(
            self._reference_quality_payload(
                neighbor_future_delta_soh[:k],
                retrieval_alpha[:k],
                retrieval_mask[:k],
                [str(value) for value in self._row_cell_uid[selected_ref_indices[:k]].tolist()],
            )
        )
        retrieval_confidence = compute_retrieval_confidence(confidence_payload, self.rag_config) if k else 0.0
        return RetrievalResult(
            query_case_id=int(query_case_id),
            neighbor_case_ids=neighbor_case_ids,
            retrieval_mask=retrieval_mask,
            d_soh_state=d_soh_state,
            d_qv_shape=d_qv_shape,
            d_physics=d_physics,
            d_operation=d_operation,
            d_metadata=d_metadata,
            composite_distance=composite_distance,
            retrieval_confidence=float(retrieval_confidence),
            retrieval_alpha=retrieval_alpha,
            neighbor_future_delta_soh=neighbor_future_delta_soh,
            neighbor_metadata=neighbor_metadata,
            explain_json={"fast_cache_path": True, "enabled_distance_components": self.enabled_component_names},
            stage_distance=stage_distance,
            missing_penalty=missing_penalty,
            reference_compatibility_score=reference_compatibility_score,
            mmr_diversity_score=mmr_diversity_score,
        )

    def retrieve(self, query_case_id: int) -> RetrievalResult:
        query_idx = self.case_id_to_index[int(query_case_id)]
        candidate_case_ids = self._hard_filter_candidate_ids(query_idx, self._coarse_candidates(query_idx))
        if candidate_case_ids.size == 0:
            horizon = int(self.arrays["future_delta"].shape[1])
            zeros = np.zeros(self.top_k, dtype=np.float32)
            return RetrievalResult(
                query_case_id=int(query_case_id),
                neighbor_case_ids=np.full(self.top_k, -1, dtype=np.int64),
                retrieval_mask=zeros.copy(),
                d_soh_state=zeros.copy(),
                d_qv_shape=zeros.copy(),
                d_physics=zeros.copy(),
                d_operation=zeros.copy(),
                d_metadata=zeros.copy(),
                composite_distance=np.full(self.top_k, np.inf, dtype=np.float32),
                retrieval_confidence=0.0,
                retrieval_alpha=zeros.copy(),
                neighbor_future_delta_soh=np.zeros((self.top_k, horizon), dtype=np.float32),
                neighbor_metadata=[{} for _ in range(self.top_k)],
                explain_json={
                    "core_component_names": CORE_COMPONENTS,
                    "optional_component_names": OPTIONAL_COMPONENTS,
                    "enabled_distance_components": self.enabled_component_names,
                    "warning": "No valid neighbors after hard filtering.",
                },
                stage_distance=zeros.copy(),
                missing_penalty=zeros.copy(),
                reference_compatibility_score=zeros.copy(),
                mmr_diversity_score=zeros.copy(),
            )

        candidate_case_ids = candidate_case_ids[: self.top_m]
        bundles = []
        for case_id in candidate_case_ids.tolist():
            ref_idx = self.case_id_to_index[int(case_id)]
            bundles.append(self._distance_bundle(query_idx, ref_idx))

        bundle_frame = pd.DataFrame(bundles)
        order = np.argsort(bundle_frame["composite_distance"].to_numpy(dtype=np.float32))
        candidate_case_ids = candidate_case_ids[order]
        bundle_frame = bundle_frame.iloc[order].reset_index(drop=True)

        selected_case_ids, mmr_scores = self._apply_mmr(candidate_case_ids, bundle_frame["composite_distance"].to_numpy(dtype=np.float32))
        if selected_case_ids.size == 0:
            selected_case_ids = candidate_case_ids[: self.top_k]
            mmr_scores = np.ones(len(selected_case_ids), dtype=np.float32)

        selected_positions = [int(np.flatnonzero(candidate_case_ids == case_id)[0]) for case_id in selected_case_ids.tolist()]
        selected_frame = bundle_frame.iloc[selected_positions].reset_index(drop=True)
        k = min(len(selected_case_ids), self.top_k)
        horizon = int(self.arrays["future_delta"].shape[1])

        neighbor_case_ids = np.full(self.top_k, -1, dtype=np.int64)
        retrieval_mask = np.zeros(self.top_k, dtype=np.float32)
        composite_distance = np.full(self.top_k, np.inf, dtype=np.float32)
        named_distance_arrays = {
            name: (np.full(self.top_k, np.nan, dtype=np.float32) if name in OPTIONAL_COMPONENTS else np.zeros(self.top_k, dtype=np.float32))
            for name in COMPONENT_NAMES
        }
        stage_distance = np.zeros(self.top_k, dtype=np.float32)
        missing_penalty = np.zeros(self.top_k, dtype=np.float32)
        reference_compatibility_score = np.zeros(self.top_k, dtype=np.float32)
        mmr_diversity_score = np.zeros(self.top_k, dtype=np.float32)
        retrieval_alpha = np.zeros(self.top_k, dtype=np.float32)
        neighbor_future_delta_soh = np.zeros((self.top_k, horizon), dtype=np.float32)
        neighbor_metadata = [{} for _ in range(self.top_k)]

        if k > 0:
            neighbor_case_ids[:k] = selected_case_ids[:k]
            retrieval_mask[:k] = 1.0
            composite_distance[:k] = selected_frame["composite_distance"].to_numpy(dtype=np.float32)[:k]
            for name in COMPONENT_NAMES:
                named_distance_arrays[name][:k] = selected_frame[name].to_numpy(dtype=np.float32)[:k]
            stage_distance[:k] = selected_frame["stage_distance"].to_numpy(dtype=np.float32)[:k]
            missing_penalty[:k] = selected_frame["missing_penalty"].to_numpy(dtype=np.float32)[:k]
            reference_compatibility_score[:k] = selected_frame["reference_compatibility_score"].to_numpy(dtype=np.float32)[:k]
            mmr_diversity_score[: min(len(mmr_scores), self.top_k)] = mmr_scores[: self.top_k]
            retrieval_alpha[:k] = _softmax_masked(composite_distance[:k], retrieval_mask[:k], self.retrieval_temperature)
            row_indices = [self.case_id_to_index[int(case_id)] for case_id in selected_case_ids[:k].tolist()]
            neighbor_future_delta_soh[:k] = np.asarray(self.arrays["future_delta"][row_indices], dtype=np.float32)
            neighbor_metadata[:k] = [
                self.case_rows.iloc[row_idx][
                    [
                        "case_id",
                        "cell_uid",
                        "chemistry_family",
                        "domain_label",
                        "source_dataset",
                        "degradation_stage",
                        "anchor_soh",
                        "recent_soh_slope",
                        "voltage_window_bucket",
                    ]
                ].to_dict()
                for row_idx in row_indices
            ]

        confidence_payload = {
            "composite_distance": composite_distance[:k],
            "feature_availability_ratio": float(selected_frame["feature_availability_ratio"].mean()) if k > 0 else 0.0,
            "chemistry_match_rate": float(selected_frame["chemistry_match"].mean()) if k > 0 else 0.0,
            "domain_match_rate": float(selected_frame["domain_match"].mean()) if k > 0 else 0.0,
        }
        confidence_payload.update(
            self._reference_quality_payload(
                neighbor_future_delta_soh[:k],
                retrieval_alpha[:k],
                retrieval_mask[:k],
                [str(item.get("cell_uid", "")) for item in neighbor_metadata[:k]],
            )
        )
        retrieval_confidence = compute_retrieval_confidence(confidence_payload, self.rag_config) if k > 0 else 0.0

        explain_json = {
            "core_component_names": CORE_COMPONENTS,
            "optional_component_names": OPTIONAL_COMPONENTS,
            "enabled_distance_components": self.enabled_component_names,
            "candidate_count_after_filter": int(len(candidate_case_ids)),
            "reference_quality": {
                key: float(value)
                for key, value in self._reference_quality_payload(
                    neighbor_future_delta_soh[:k],
                    retrieval_alpha[:k],
                    retrieval_mask[:k],
                    [str(item.get("cell_uid", "")) for item in neighbor_metadata[:k]],
                ).items()
            },
            "selected_neighbors": [
                {
                    "case_id": int(neighbor_case_ids[pos]),
                    "metadata": neighbor_metadata[pos],
                    **{name: float(named_distance_arrays[name][pos]) for name in COMPONENT_NAMES},
                    "composite_distance": float(composite_distance[pos]),
                    "stage_distance": float(stage_distance[pos]),
                    "missing_penalty": float(missing_penalty[pos]),
                    "reference_compatibility_score": float(reference_compatibility_score[pos]),
                    "mmr_diversity_score": float(mmr_diversity_score[pos]),
                }
                for pos in range(k)
            ],
            "feature_switches": self.rag_config,
        }

        return RetrievalResult(
            query_case_id=int(query_case_id),
            neighbor_case_ids=neighbor_case_ids,
            retrieval_mask=retrieval_mask,
            d_soh_state=named_distance_arrays["d_soh_state"],
            d_qv_shape=named_distance_arrays["d_qv_shape"],
            d_physics=named_distance_arrays["d_physics"],
            d_operation=named_distance_arrays["d_operation"],
            d_metadata=named_distance_arrays["d_metadata"],
            composite_distance=composite_distance,
            retrieval_confidence=float(retrieval_confidence),
            retrieval_alpha=retrieval_alpha,
            neighbor_future_delta_soh=neighbor_future_delta_soh,
            neighbor_metadata=neighbor_metadata,
            explain_json=explain_json,
            stage_distance=stage_distance,
            missing_penalty=missing_penalty,
            reference_compatibility_score=reference_compatibility_score,
            mmr_diversity_score=mmr_diversity_score,
        )

    def build_cache(self, query_case_ids: Iterable[int], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        query_case_ids = [int(case_id) for case_id in query_case_ids]
        results = []
        total = len(query_case_ids)
        for pos, case_id in enumerate(query_case_ids, start=1):
            results.append(self._retrieve_fast(case_id))
            if pos == 1 or pos % 500 == 0 or pos == total:
                print(f"[retrieval-cache] {output_path.name}: {pos}/{total}", flush=True)
        np.savez_compressed(
            output_path,
            query_case_ids=np.asarray(query_case_ids, dtype=np.int64),
            neighbor_case_ids=np.stack([result.neighbor_case_ids for result in results]).astype(np.int64),
            retrieval_mask=np.stack([result.retrieval_mask for result in results]).astype(np.float32),
            retrieval_alpha=np.stack([result.retrieval_alpha for result in results]).astype(np.float32),
            retrieval_confidence=np.asarray([result.retrieval_confidence for result in results], dtype=np.float32),
            composite_distance=np.stack([result.composite_distance for result in results]).astype(np.float32),
            component_distances=np.stack([result.component_distances for result in results]).astype(np.float32),
            d_soh_state=np.stack([result.d_soh_state for result in results]).astype(np.float32),
            d_qv_shape=np.stack([result.d_qv_shape for result in results]).astype(np.float32),
            d_physics=np.stack([result.d_physics for result in results]).astype(np.float32),
            d_operation=np.stack([result.d_operation for result in results]).astype(np.float32),
            d_metadata=np.stack([result.d_metadata for result in results]).astype(np.float32),
            stage_distance=np.stack([np.asarray(result.stage_distance, dtype=np.float32) for result in results]).astype(np.float32),
            missing_penalty=np.stack([np.asarray(result.missing_penalty, dtype=np.float32) for result in results]).astype(np.float32),
            reference_compatibility_score=np.stack([np.asarray(result.reference_compatibility_score, dtype=np.float32) for result in results]).astype(np.float32),
            mmr_diversity_score=np.stack([np.asarray(result.mmr_diversity_score, dtype=np.float32) for result in results]).astype(np.float32),
            component_names=np.asarray(COMPONENT_NAMES, dtype=object),
            config_hash=np.asarray([self.retrieval_config_hash()], dtype=object),
        )
        return output_path
