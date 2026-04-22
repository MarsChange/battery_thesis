from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from retrieval.physics_distance import (
    compute_retrieval_confidence,
    degradation_stage_distance,
    metadata_distance,
    normalized_l2,
    operation_distance,
    physics_feature_distance,
    qv_map_distance,
    soh_state_distance,
)

try:
    from retrieval.index import FAISSIndex
except Exception:  # pragma: no cover - fallback path only matters without faiss.
    FAISSIndex = None


COMPONENT_NAMES = ["tsfm", "soh", "qv", "physics", "operation", "metadata", "stage", "missing"]


def _read_case_rows(case_bank_dir: Path) -> pd.DataFrame:
    parquet_path = case_bank_dir / "case_rows.parquet"
    csv_path = case_bank_dir / "case_rows.csv"
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing case rows at {parquet_path} or {csv_path}")


@dataclass
class RetrievalResult:
    query_case_id: int
    neighbor_case_ids: np.ndarray
    retrieval_mask: np.ndarray
    composite_distances: np.ndarray
    component_distances: np.ndarray
    retrieval_alpha: np.ndarray
    retrieval_confidence: float
    neighbor_future_delta_soh: np.ndarray
    reference_compatibility_score: np.ndarray
    explain: Dict[str, object]


def _softmax_masked(logits: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    if logits.size == 0:
        return logits
    scaled = -logits / max(float(temperature), 1e-6)
    scaled = np.where(mask > 0, scaled, -1e9)
    scaled = scaled - float(np.max(scaled))
    weight = np.exp(scaled) * mask
    total = float(weight.sum())
    return (weight / total).astype(np.float32) if total > 0 else np.zeros_like(logits, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


class MultiStageBatteryRetriever:
    def __init__(
        self,
        case_bank_dir: str | Path,
        db_splits: List[str],
        metric: str = "cosine",
        stage1_embedding_name: str = "tsfm",
        top_m: int = 200,
        top_k: int = 8,
        rerank_weights: Dict[str, float] | None = None,
        hard_filter: Dict[str, object] | None = None,
        mmr: Dict[str, object] | None = None,
        retrieval_temperature: float = 0.1,
        qv_channel_weights: Dict[str, float] | None = None,
    ):
        self.case_bank_dir = Path(case_bank_dir)
        self.case_rows = _read_case_rows(self.case_bank_dir).sort_values("case_id").reset_index(drop=True)
        self.case_rows["case_id"] = self.case_rows["case_id"].astype(int)
        self.metric = metric
        self.stage1_embedding_name = stage1_embedding_name
        self.top_m = int(top_m)
        self.top_k = int(top_k)
        self.retrieval_temperature = float(retrieval_temperature)
        self.rerank_weights = {
            "tsfm": 0.2,
            "soh_state": 0.2,
            "qv_map": 0.25,
            "physics_features": 0.2,
            "operation": 0.05,
            "metadata": 0.05,
            "stage": 0.05,
            "missing": 0.05,
            **(rerank_weights or {}),
        }
        self.hard_filter = {"same_cell_policy": "exclude", **(hard_filter or {})}
        self.mmr = {
            "use_mmr": True,
            "mmr_lambda": 0.75,
            "max_neighbors_per_cell": 2,
            "max_neighbors_per_domain": None,
            **(mmr or {}),
        }
        self.qv_channel_weights = qv_channel_weights or {
            "Vc": 0.10,
            "Vd": 0.20,
            "Ic": 0.05,
            "Id": 0.05,
            "DeltaV": 0.35,
            "R": 0.25,
        }

        self.db_splits = list(db_splits)
        self.db_mask = np.asarray(self.case_rows["split"].astype(str).isin(self.db_splits).to_numpy(), dtype=bool).copy()
        self.db_mask &= self.case_rows["split"].astype(str).ne("target_query").to_numpy()
        self.db_case_ids = self.case_rows.loc[self.db_mask, "case_id"].to_numpy(dtype=np.int64)
        self.case_id_to_index = {int(case_id): idx for idx, case_id in enumerate(self.case_rows["case_id"].tolist())}

        self.arrays = self._load_case_arrays()
        self.stage1_embeddings = self._load_stage1_embeddings()
        self._build_index()

    def _load_case_arrays(self) -> Dict[str, np.ndarray]:
        arrays = {
            "future_delta": np.load(self.case_bank_dir / "case_future_delta_soh.npy"),
            "future_soh": np.load(self.case_bank_dir / "case_future_soh.npy"),
            "cycle_stats": np.load(self.case_bank_dir / "case_cycle_stats.npy"),
            "soh_seq": np.load(self.case_bank_dir / "case_soh_seq.npy"),
            "qv_maps": np.load(self.case_bank_dir / "case_qv_maps.npy"),
            "qv_masks": np.load(self.case_bank_dir / "case_qv_masks.npy"),
            "partial_charge": np.load(self.case_bank_dir / "case_partial_charge.npy"),
            "partial_charge_mask": np.load(self.case_bank_dir / "case_partial_charge_mask.npy"),
            "relaxation": np.load(self.case_bank_dir / "case_relaxation.npy"),
            "relaxation_mask": np.load(self.case_bank_dir / "case_relaxation_mask.npy"),
            "physics_features": np.load(self.case_bank_dir / "case_physics_features.npy"),
            "physics_feature_masks": np.load(self.case_bank_dir / "case_physics_feature_masks.npy"),
            "anchor_physics_features": np.load(self.case_bank_dir / "case_anchor_physics_features.npy"),
            "operation_seq": np.load(self.case_bank_dir / "case_operation_seq.npy"),
            "future_ops": np.load(self.case_bank_dir / "case_future_ops.npy"),
            "future_ops_mask": np.load(self.case_bank_dir / "case_future_ops_mask.npy"),
        }
        tsfm_path = self.case_bank_dir / "case_tsfm_embeddings.npy"
        arrays["tsfm_embeddings"] = np.load(tsfm_path) if tsfm_path.exists() else None
        return arrays

    def _load_stage1_embeddings(self) -> np.ndarray:
        if self.stage1_embedding_name == "tsfm" and self.arrays["tsfm_embeddings"] is not None:
            return np.asarray(self.arrays["tsfm_embeddings"], dtype=np.float32)
        if self.stage1_embedding_name == "physics":
            return np.concatenate(
                [
                    np.asarray(self.arrays["anchor_physics_features"], dtype=np.float32),
                    np.asarray(self.arrays["soh_seq"][:, -4:], dtype=np.float32),
                ],
                axis=-1,
            )
        # hybrid / fallback
        return np.concatenate(
            [
                np.asarray(self.arrays["anchor_physics_features"], dtype=np.float32),
                np.asarray(self.arrays["cycle_stats"][:, -1], dtype=np.float32),
                np.asarray(self.arrays["soh_seq"][:, -4:], dtype=np.float32),
            ],
            axis=-1,
        )

    def _build_index(self) -> None:
        db_embeddings = self.stage1_embeddings[self.db_case_ids]
        self.db_embeddings = np.asarray(db_embeddings, dtype=np.float32)
        if FAISSIndex is not None and self.db_embeddings.size > 0:
            self.index = FAISSIndex(dim=self.db_embeddings.shape[1], metric=self.metric)
            self.index.add(self.db_embeddings)
        else:
            self.index = None

    def retrieval_config_hash(self) -> str:
        payload = {
            "db_splits": self.db_splits,
            "metric": self.metric,
            "stage1_embedding_name": self.stage1_embedding_name,
            "top_m": self.top_m,
            "top_k": self.top_k,
            "rerank_weights": self.rerank_weights,
            "hard_filter": self.hard_filter,
            "mmr": self.mmr,
            "retrieval_temperature": self.retrieval_temperature,
        }
        text = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def _coarse_candidates(self, query_idx: int) -> np.ndarray:
        if len(self.db_case_ids) == 0:
            return np.zeros(0, dtype=np.int64)
        query_embedding = np.asarray(self.stage1_embeddings[query_idx : query_idx + 1], dtype=np.float32)
        fetch = min(max(self.top_m * 4, self.top_k * 8), len(self.db_case_ids))
        if self.index is not None:
            _, index_positions = self.index.search(query_embedding, fetch)
            candidate_case_ids = self.db_case_ids[index_positions[0]]
        else:
            distances = np.linalg.norm(self.db_embeddings - query_embedding[0], axis=1)
            candidate_case_ids = self.db_case_ids[np.argsort(distances)[:fetch]]
        return np.asarray(candidate_case_ids, dtype=np.int64)

    def _hard_filter_candidate_ids(self, query_idx: int, candidate_case_ids: np.ndarray) -> np.ndarray:
        query_row = self.case_rows.iloc[query_idx]
        query_case_id = int(query_row["case_id"])
        same_cell_policy = str(self.hard_filter.get("same_cell_policy", "exclude"))
        query_horizon = int(query_row["target_horizon"])
        valid_ids = []
        for case_id in candidate_case_ids.tolist():
            ref_idx = self.case_id_to_index[int(case_id)]
            ref_row = self.case_rows.iloc[ref_idx]
            if int(ref_row["case_id"]) == query_case_id:
                continue
            if int(ref_row["target_horizon"]) != query_horizon:
                continue
            if str(ref_row["split"]) == "target_query":
                continue
            if same_cell_policy == "exclude" and str(ref_row["cell_uid"]) == str(query_row["cell_uid"]):
                continue
            if same_cell_policy == "past_only" and str(ref_row["cell_uid"]) == str(query_row["cell_uid"]):
                if int(ref_row["window_end"]) > int(query_row["window_end"]):
                    continue
            valid_ids.append(int(case_id))
        return np.asarray(valid_ids, dtype=np.int64)

    def _meta_dict(self, row: pd.Series) -> Dict[str, object]:
        return {
            "chemistry_family": row.get("chemistry_family"),
            "temperature_bucket": row.get("temperature_bucket"),
            "charge_rate_bucket": row.get("charge_rate_bucket"),
            "discharge_policy_family": row.get("discharge_policy_family"),
            "nominal_capacity_bucket": row.get("nominal_capacity_bucket"),
            "voltage_window_bucket": row.get("voltage_window_bucket"),
            "full_or_partial": row.get("full_or_partial"),
            "domain_label": row.get("domain_label"),
        }

    def _state_dict(self, row: pd.Series) -> Dict[str, float]:
        return {
            "anchor_soh": float(row.get("anchor_soh", 0.0)),
            "recent_soh_slope": float(row.get("recent_soh_slope", 0.0)),
            "recent_soh_curvature": float(row.get("recent_soh_curvature", 0.0)),
            "throughput_recent": float(row.get("throughput_recent", 0.0)),
        }

    def _operation_summary(self, idx: int) -> np.ndarray:
        op_seq = np.asarray(self.arrays["operation_seq"][idx], dtype=np.float32)
        future = np.asarray(self.arrays["future_ops"][idx], dtype=np.float32)
        return np.concatenate([op_seq.mean(axis=0), op_seq.std(axis=0), future.mean(axis=0)], axis=0).astype(np.float32)

    def _missing_penalty(self, query_idx: int, ref_idx: int) -> float:
        q_avail = np.concatenate(
            [
                self.arrays["qv_masks"][query_idx, -1].reshape(-1),
                self.arrays["physics_feature_masks"][query_idx, -1].reshape(-1),
                self.arrays["future_ops_mask"][query_idx].reshape(-1),
            ],
            axis=0,
        )
        r_avail = np.concatenate(
            [
                self.arrays["qv_masks"][ref_idx, -1].reshape(-1),
                self.arrays["physics_feature_masks"][ref_idx, -1].reshape(-1),
                self.arrays["future_ops_mask"][ref_idx].reshape(-1),
            ],
            axis=0,
        )
        common = float(((q_avail > 0) & (r_avail > 0)).mean())
        return float(1.0 - common)

    def _component_distances(self, query_idx: int, ref_idx: int) -> Tuple[np.ndarray, float]:
        query_row = self.case_rows.iloc[query_idx]
        ref_row = self.case_rows.iloc[ref_idx]

        d_tsfm = normalized_l2(self.stage1_embeddings[query_idx], self.stage1_embeddings[ref_idx])
        d_soh = soh_state_distance(self._state_dict(query_row), self._state_dict(ref_row))
        d_qv = qv_map_distance(
            self.arrays["qv_maps"][query_idx, -1],
            self.arrays["qv_maps"][ref_idx, -1],
            self.arrays["qv_masks"][query_idx, -1],
            self.arrays["qv_masks"][ref_idx, -1],
            self.qv_channel_weights,
        )
        d_physics = physics_feature_distance(
            self.arrays["physics_features"][query_idx, -1],
            self.arrays["physics_features"][ref_idx, -1],
            self.arrays["physics_feature_masks"][query_idx, -1],
            self.arrays["physics_feature_masks"][ref_idx, -1],
        )
        d_operation = operation_distance(self._operation_summary(query_idx), self._operation_summary(ref_idx))
        d_metadata = metadata_distance(
            self._meta_dict(query_row),
            self._meta_dict(ref_row),
            {
                "chemistry_family": 0.35,
                "domain_label": 0.20,
                "temperature_bucket": 0.10,
                "charge_rate_bucket": 0.10,
                "discharge_policy_family": 0.10,
                "full_or_partial": 0.05,
                "nominal_capacity_bucket": 0.05,
                "voltage_window_bucket": 0.05,
            },
        )
        d_stage = degradation_stage_distance(str(query_row["degradation_stage"]), str(ref_row["degradation_stage"]))
        d_missing = self._missing_penalty(query_idx, ref_idx)
        components = np.asarray([d_tsfm, d_soh, d_qv, d_physics, d_operation, d_metadata, d_stage, d_missing], dtype=np.float32)
        composite = (
            self.rerank_weights["tsfm"] * d_tsfm
            + self.rerank_weights["soh_state"] * d_soh
            + self.rerank_weights["qv_map"] * d_qv
            + self.rerank_weights["physics_features"] * d_physics
            + self.rerank_weights["operation"] * d_operation
            + self.rerank_weights["metadata"] * d_metadata
            + self.rerank_weights["stage"] * d_stage
            + self.rerank_weights["missing"] * d_missing
        )
        return components, float(composite)

    def _reference_compatibility_score(self, query_idx: int, ref_idx: int, components: np.ndarray) -> float:
        query_row = self.case_rows.iloc[query_idx]
        ref_row = self.case_rows.iloc[ref_idx]
        chemistry_bonus = 0.1 if str(query_row["chemistry_family"]) == str(ref_row["chemistry_family"]) else -0.05
        stage_bonus = 0.1 if str(query_row["degradation_stage"]) == str(ref_row["degradation_stage"]) else -0.05
        domain_bonus = 0.05 if str(query_row["domain_label"]) == str(ref_row["domain_label"]) else 0.0
        base = 1.0 - float(np.mean(components[:5]))
        return float(np.clip(base + chemistry_bonus + stage_bonus + domain_bonus - 0.2 * components[-1], 0.0, 1.0))

    def _apply_mmr(self, query_idx: int, candidate_case_ids: np.ndarray, distances: np.ndarray) -> np.ndarray:
        if not bool(self.mmr.get("use_mmr", True)) or len(candidate_case_ids) <= self.top_k:
            return candidate_case_ids[: self.top_k]
        lambda_ = float(self.mmr.get("mmr_lambda", 0.75))
        max_per_cell = int(self.mmr.get("max_neighbors_per_cell", 2))
        max_per_domain = self.mmr.get("max_neighbors_per_domain")
        selected: List[int] = []
        cell_counter: Dict[str, int] = {}
        domain_counter: Dict[str, int] = {}
        candidate_order = list(np.argsort(distances))

        while candidate_order and len(selected) < self.top_k:
            best_case = None
            best_score = None
            for pos in candidate_order:
                case_id = int(candidate_case_ids[pos])
                ref_idx = self.case_id_to_index[case_id]
                ref_row = self.case_rows.iloc[ref_idx]
                cell_uid = str(ref_row["cell_uid"])
                domain_label = str(ref_row["domain_label"])
                if cell_counter.get(cell_uid, 0) >= max_per_cell:
                    continue
                if max_per_domain is not None and domain_counter.get(domain_label, 0) >= int(max_per_domain):
                    continue
                relevance = -float(distances[pos])
                diversity = 0.0
                if selected:
                    similarities = []
                    for selected_case in selected:
                        sel_idx = self.case_id_to_index[selected_case]
                        similarities.append(_cosine_similarity(self.stage1_embeddings[ref_idx], self.stage1_embeddings[sel_idx]))
                    diversity = max(similarities) if similarities else 0.0
                mmr_score = lambda_ * relevance - (1.0 - lambda_) * diversity
                if best_score is None or mmr_score > best_score:
                    best_case = case_id
                    best_score = mmr_score
            if best_case is None:
                break
            selected.append(int(best_case))
            chosen_row = self.case_rows.iloc[self.case_id_to_index[int(best_case)]]
            cell_counter[str(chosen_row["cell_uid"])] = cell_counter.get(str(chosen_row["cell_uid"]), 0) + 1
            domain_counter[str(chosen_row["domain_label"])] = domain_counter.get(str(chosen_row["domain_label"]), 0) + 1
            candidate_order = [pos for pos in candidate_order if int(candidate_case_ids[pos]) != int(best_case)]
        return np.asarray(selected, dtype=np.int64)

    def retrieve(self, query_case_id: int) -> RetrievalResult:
        query_idx = self.case_id_to_index[int(query_case_id)]
        candidate_case_ids = self._coarse_candidates(query_idx)
        candidate_case_ids = self._hard_filter_candidate_ids(query_idx, candidate_case_ids)
        if candidate_case_ids.size == 0:
            return RetrievalResult(
                query_case_id=int(query_case_id),
                neighbor_case_ids=np.full(self.top_k, -1, dtype=np.int64),
                retrieval_mask=np.zeros(self.top_k, dtype=np.float32),
                composite_distances=np.full(self.top_k, np.inf, dtype=np.float32),
                component_distances=np.zeros((self.top_k, len(COMPONENT_NAMES)), dtype=np.float32),
                retrieval_alpha=np.zeros(self.top_k, dtype=np.float32),
                retrieval_confidence=0.0,
                neighbor_future_delta_soh=np.zeros((self.top_k, self.arrays["future_delta"].shape[1]), dtype=np.float32),
                reference_compatibility_score=np.zeros(self.top_k, dtype=np.float32),
                explain={"component_names": COMPONENT_NAMES, "warning": "No valid neighbors after hard filter"},
            )

        candidate_case_ids = candidate_case_ids[: self.top_m]
        components = []
        composite = []
        compat = []
        for case_id in candidate_case_ids.tolist():
            ref_idx = self.case_id_to_index[int(case_id)]
            comp, comp_dist = self._component_distances(query_idx, ref_idx)
            components.append(comp)
            composite.append(comp_dist)
            compat.append(self._reference_compatibility_score(query_idx, ref_idx, comp))
        components = np.asarray(components, dtype=np.float32)
        composite = np.asarray(composite, dtype=np.float32)
        compat = np.asarray(compat, dtype=np.float32)
        order = np.argsort(composite)
        candidate_case_ids = candidate_case_ids[order]
        components = components[order]
        composite = composite[order]
        compat = compat[order]

        selected_case_ids = self._apply_mmr(query_idx, candidate_case_ids, composite)
        if selected_case_ids.size == 0:
            selected_case_ids = candidate_case_ids[: self.top_k]
        selected_positions = [int(np.flatnonzero(candidate_case_ids == case_id)[0]) for case_id in selected_case_ids.tolist()]
        selected_components = components[selected_positions]
        selected_composite = composite[selected_positions]
        selected_compat = compat[selected_positions]

        k = min(len(selected_case_ids), self.top_k)
        neighbor_case_ids = np.full(self.top_k, -1, dtype=np.int64)
        retrieval_mask = np.zeros(self.top_k, dtype=np.float32)
        composite_distances = np.full(self.top_k, np.inf, dtype=np.float32)
        component_distances = np.zeros((self.top_k, len(COMPONENT_NAMES)), dtype=np.float32)
        retrieval_alpha = np.zeros(self.top_k, dtype=np.float32)
        compatibility = np.zeros(self.top_k, dtype=np.float32)
        neighbor_future_delta = np.zeros((self.top_k, self.arrays["future_delta"].shape[1]), dtype=np.float32)

        if k > 0:
            neighbor_case_ids[:k] = selected_case_ids[:k]
            retrieval_mask[:k] = 1.0
            composite_distances[:k] = selected_composite[:k]
            component_distances[:k] = selected_components[:k]
            compatibility[:k] = selected_compat[:k]
            retrieval_alpha[:k] = _softmax_masked(selected_composite[:k], retrieval_mask[:k], self.retrieval_temperature)
            neighbor_future_delta[:k] = self.arrays["future_delta"][selected_case_ids[:k]]

        confidence = compute_retrieval_confidence(component_distances[:k]) if k > 0 else 0.0
        explain = {
            "component_names": COMPONENT_NAMES,
            "candidate_count_after_filter": int(len(candidate_case_ids)),
            "selected_metadata": [
                self.case_rows.iloc[self.case_id_to_index[int(case_id)]][
                    ["case_id", "cell_uid", "chemistry_family", "domain_label", "degradation_stage", "anchor_soh", "recent_soh_slope"]
                ].to_dict()
                for case_id in neighbor_case_ids[:k].tolist()
            ],
            "selected_component_distances": component_distances[:k].tolist(),
        }
        return RetrievalResult(
            query_case_id=int(query_case_id),
            neighbor_case_ids=neighbor_case_ids,
            retrieval_mask=retrieval_mask,
            composite_distances=composite_distances,
            component_distances=component_distances,
            retrieval_alpha=retrieval_alpha,
            retrieval_confidence=float(confidence),
            neighbor_future_delta_soh=neighbor_future_delta,
            reference_compatibility_score=compatibility,
            explain=explain,
        )

    def build_cache(self, query_case_ids: Iterable[int], output_path: str | Path) -> Path:
        output_path = Path(output_path)
        query_case_ids = [int(case_id) for case_id in query_case_ids]
        all_results = [self.retrieve(case_id) for case_id in query_case_ids]
        np.savez_compressed(
            output_path,
            query_case_ids=np.asarray(query_case_ids, dtype=np.int64),
            neighbor_case_ids=np.stack([result.neighbor_case_ids for result in all_results]).astype(np.int64),
            retrieval_mask=np.stack([result.retrieval_mask for result in all_results]).astype(np.float32),
            composite_distances=np.stack([result.composite_distances for result in all_results]).astype(np.float32),
            component_distances=np.stack([result.component_distances for result in all_results]).astype(np.float32),
            retrieval_alpha=np.stack([result.retrieval_alpha for result in all_results]).astype(np.float32),
            retrieval_confidence=np.asarray([result.retrieval_confidence for result in all_results], dtype=np.float32),
            reference_compatibility_score=np.stack([result.reference_compatibility_score for result in all_results]).astype(np.float32),
            config_hash=np.asarray([self.retrieval_config_hash()], dtype=object),
            component_names=np.asarray(COMPONENT_NAMES, dtype=object),
        )
        return output_path
