from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from retrieval.multistage_retriever import COMPONENT_NAMES, MultiStageBatteryRetriever


DEFAULT_META_FIELDS = [
    "chemistry_family",
    "temperature_bucket",
    "charge_rate_bucket",
    "discharge_policy_family",
    "nominal_capacity_bucket",
    "voltage_window_bucket",
    "full_or_partial",
    "domain_label",
    "degradation_stage",
    "source_dataset",
]

CHEMISTRY_TO_ID = {"LFP": 0, "NCM": 1, "NCA": 2}
RESIDUAL_FACTOR_NAMES = [
    "high_temperature_expert",
    "high_current_expert",
    "high_cycle_expert",
    "high_power_expert",
]


def chemistry_family_to_id(value: object) -> int:
    text = str(value or "unknown").upper()
    if "LFP" in text:
        return CHEMISTRY_TO_ID["LFP"]
    if "NCM" in text:
        return CHEMISTRY_TO_ID["NCM"]
    if "NCA" in text:
        return CHEMISTRY_TO_ID["NCA"]
    return CHEMISTRY_TO_ID["LFP"]


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
    """Convert metadata columns to plain Python objects for fast row access.

    Pandas can preserve parquet string columns as pyarrow-backed scalars. That
    is memory efficient, but very slow for the row-wise indexing used by case
    bank retrieval/evaluation. Numeric columns are kept numeric.
    """

    frame = rows.copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].fillna("unknown").astype(str).astype(object)
    return frame


def _source_train_quantile(values: np.ndarray, rows: pd.DataFrame, quantile: float, fallback: float) -> float:
    """Fit robust factor thresholds on source_train only to avoid query leakage."""

    values = np.asarray(values, dtype=np.float32)
    split = rows["split"].astype(str).to_numpy()
    mask = (split == "source_train") & np.isfinite(values)
    if not bool(mask.any()):
        mask = np.isfinite(values)
    if not bool(mask.any()):
        return float(fallback)
    return float(np.quantile(values[mask], quantile))


def _robust_factor_score(values: np.ndarray, rows: pd.DataFrame, low_q: float = 0.50, high_q: float = 0.90) -> np.ndarray:
    """Map a raw stress feature to [0, 1] using source_train quantiles."""

    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = _source_train_quantile(values, rows, low_q, fallback=0.0)
    high = _source_train_quantile(values, rows, high_q, fallback=low + 1.0)
    denom = max(high - low, 1e-6)
    return np.clip((values - low) / denom, 0.0, 1.0).astype(np.float32)


def compute_residual_factor_scores(rows: pd.DataFrame, expert_seq: np.ndarray | None) -> np.ndarray:
    """Compute factor scores for residual specialist training.

    Output columns follow `RESIDUAL_FACTOR_NAMES`:
    high-temperature, high-current, high-cycle-aging, and high-power stress.

    Temperature is anchored to a room-temperature baseline: 25-30 C is treated
    as near-zero high-temperature evidence, while hotter windows receive larger
    scores. Current and power use source_train robust quantiles because their
    absolute scales differ across public datasets and protocols.
    """

    num_cases = int(len(rows))
    if expert_seq is None:
        return np.zeros((num_cases, len(RESIDUAL_FACTOR_NAMES)), dtype=np.float32)

    seq = np.nan_to_num(np.asarray(expert_seq, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if seq.ndim != 3 or seq.shape[0] != num_cases:
        return np.zeros((num_cases, len(RESIDUAL_FACTOR_NAMES)), dtype=np.float32)

    def _seq_column(index: int) -> np.ndarray:
        if seq.shape[-1] <= index:
            return np.zeros(num_cases, dtype=np.float32)
        return seq[:, :, index].mean(axis=1).astype(np.float32)

    def _seq_column_max(index: int) -> np.ndarray:
        if seq.shape[-1] <= index:
            return np.zeros(num_cases, dtype=np.float32)
        return seq[:, :, index].max(axis=1).astype(np.float32)

    # EXPERT_SEQ_FEATURES indices from battery_data.build_case_bank:
    # 5 temperature_mean, 7 temperature_max, 8 current_abs_mean,
    # 10 current_abs_max, 11 power_energy_proxy, 12 cycle_aging_index,
    # 13 soh_below_0p9.
    temp_level = np.maximum(_seq_column(5), _seq_column_max(7))
    current_level = np.maximum(_seq_column(8), _seq_column_max(10))
    power_level = np.abs(_seq_column(11))
    cycle_aging = _seq_column(12)
    soh_below = _seq_column(13)

    anchor_soh = pd.to_numeric(rows.get("anchor_soh", pd.Series([1.0] * num_cases)), errors="coerce").fillna(1.0).to_numpy(dtype=np.float32)
    high_temperature = np.clip((temp_level - 30.0) / 15.0, 0.0, 1.0).astype(np.float32)
    high_current = _robust_factor_score(current_level, rows)
    high_power = _robust_factor_score(power_level, rows)
    deep_soh = np.clip((0.90 - anchor_soh) / 0.15, 0.0, 1.0).astype(np.float32)
    high_cycle = np.maximum.reduce([np.clip(cycle_aging, 0.0, 1.0), np.clip(soh_below / 0.15, 0.0, 1.0), deep_soh]).astype(np.float32)

    return np.stack([high_temperature, high_current, high_cycle, high_power], axis=-1).astype(np.float32)


class BatterySOHForecastDataset(Dataset):
    def __init__(
        self,
        case_bank_dir: str | Path,
        splits: List[str],
        retrieval_cfg: Dict[str, object] | None = None,
        meta_fields: List[str] | None = None,
        auto_build_retrieval_cache: bool = True,
    ):
        self.case_bank_dir = Path(case_bank_dir)
        self.case_rows = _read_case_rows(self.case_bank_dir).sort_values("case_id").reset_index(drop=True)
        self.case_rows["case_id"] = self.case_rows["case_id"].astype(int)
        self.case_id_to_row_index = {int(case_id): row_idx for row_idx, case_id in enumerate(self.case_rows["case_id"].tolist())}
        self.meta_fields = list(meta_fields or DEFAULT_META_FIELDS)
        self.retrieval_cfg = retrieval_cfg or {}

        self.arrays = self._load_arrays()
        if self.arrays.get("residual_factor_scores") is None:
            self.arrays["residual_factor_scores"] = compute_residual_factor_scores(self.case_rows, self.arrays.get("expert_seq"))
        self.meta_vocab = self._build_meta_vocab(self.case_rows, self.meta_fields)
        self.meta_ids_all = self._encode_meta_ids(self.case_rows, self.meta_fields, self.meta_vocab)

        split_mask = self.case_rows["split"].astype(str).isin(splits).to_numpy()
        self.indices = np.flatnonzero(split_mask).astype(np.int64)
        self.selected_case_ids = self.case_rows.loc[self.indices, "case_id"].to_numpy(dtype=np.int64)
        self.query_case_id_to_local = {int(case_id): pos for pos, case_id in enumerate(self.selected_case_ids.tolist())}

        if retrieval_cfg:
            self.retrieval_cache = self._load_or_build_retrieval_cache(auto_build_retrieval_cache=auto_build_retrieval_cache)
        else:
            self.retrieval_cache = None

    def _load_arrays(self) -> Dict[str, np.ndarray]:
        def _load_array(filename: str) -> np.ndarray:
            # copy-on-write mmap avoids reloading multi-GB case-bank arrays for
            # train/val/OOF/few-shot Dataset instances while keeping slices
            # writable enough for PyTorch collation.
            return np.load(self.case_bank_dir / filename, mmap_mode="c")

        arrays = {
            "cycle_stats": _load_array("case_cycle_stats.npy"),
            "soh_seq": _load_array("case_soh_seq.npy"),
            "qv_maps": _load_array("case_qv_maps.npy"),
            "qv_masks": _load_array("case_qv_masks.npy"),
            "partial_charge": _load_array("case_partial_charge.npy"),
            "partial_charge_mask": _load_array("case_partial_charge_mask.npy"),
            "physics_features": _load_array("case_physics_features.npy"),
            "physics_feature_masks": _load_array("case_physics_feature_masks.npy"),
            "anchor_physics_features": _load_array("case_anchor_physics_features.npy"),
            "operation_seq": _load_array("case_operation_seq.npy"),
            "future_ops": _load_array("case_future_ops.npy"),
            "future_ops_mask": _load_array("case_future_ops_mask.npy"),
            "future_delta_soh": _load_array("case_future_delta_soh.npy"),
            "future_soh": _load_array("case_future_soh.npy"),
        }
        expert_seq_path = self.case_bank_dir / "case_expert_seq.npy"
        arrays["expert_seq"] = np.load(expert_seq_path, mmap_mode="c") if expert_seq_path.exists() else None
        baseline_path = self.case_bank_dir / "case_baseline_delta_oof.npy"
        residual_path = self.case_bank_dir / "case_residual_target_oof.npy"
        arrays["baseline_delta_oof"] = np.load(baseline_path, mmap_mode="c") if baseline_path.exists() else None
        arrays["residual_target_oof"] = np.load(residual_path, mmap_mode="c") if residual_path.exists() else None
        factor_path = self.case_bank_dir / "case_residual_factor_scores.npy"
        arrays["residual_factor_scores"] = np.load(factor_path, mmap_mode="c") if factor_path.exists() else None
        return arrays

    @staticmethod
    def _build_meta_vocab(rows: pd.DataFrame, meta_fields: List[str]) -> Dict[str, Dict[str, int]]:
        vocab = {}
        for field in meta_fields:
            values = sorted({str(value) for value in rows[field].fillna("unknown").astype(str).to_numpy(dtype=object).tolist()})
            vocab[field] = {value: idx + 1 for idx, value in enumerate(values)}
            vocab[field]["<unk>"] = 0
        return vocab

    @staticmethod
    def _encode_meta_ids(rows: pd.DataFrame, meta_fields: List[str], meta_vocab: Dict[str, Dict[str, int]]) -> np.ndarray:
        encoded = np.zeros((len(rows), len(meta_fields)), dtype=np.int64)
        for field_pos, field in enumerate(meta_fields):
            values = rows[field].fillna("unknown").astype(str).to_numpy(dtype=object)
            vocab = meta_vocab[field]
            unknown_id = int(vocab["<unk>"])
            encoded[:, field_pos] = np.asarray([int(vocab.get(str(value), unknown_id)) for value in values], dtype=np.int64)
        return encoded

    def _retrieval_cache_path(self, retriever: MultiStageBatteryRetriever) -> Path:
        split_tag = "-".join(sorted(self.case_rows.loc[self.indices, "split"].astype(str).unique().tolist()))
        return self.case_bank_dir / f"retrieval_cache_{retriever.retrieval_config_hash()}_{split_tag}.npz"

    def _load_or_build_retrieval_cache(self, auto_build_retrieval_cache: bool) -> Dict[str, np.ndarray]:
        retriever = MultiStageBatteryRetriever(
            case_bank_dir=self.case_bank_dir,
            retrieval_config_path=str(self.retrieval_cfg.get("retrieval_feature_config_path", "configs/retrieval_features.yaml")),
            db_splits=list(self.retrieval_cfg.get("db_splits", ["source_train"])),
            metric=str(self.retrieval_cfg.get("metric", "cosine")),
            stage1_embedding_name=str(self.retrieval_cfg.get("stage1_embedding", self.retrieval_cfg.get("stage1_embedding_name", "handcrafted"))),
            top_m=int(self.retrieval_cfg.get("top_m", 200)),
            top_k=int(self.retrieval_cfg.get("top_k", 8)),
            same_cell_policy=str(self.retrieval_cfg.get("same_cell_policy", "exclude")),
            allow_cross_chemistry=bool(self.retrieval_cfg.get("allow_cross_chemistry", True)),
            mmr={
                "use_mmr": bool(self.retrieval_cfg.get("use_mmr", True)),
                "mmr_lambda": float(self.retrieval_cfg.get("mmr_lambda", 0.75)),
                "max_neighbors_per_cell": int(self.retrieval_cfg.get("max_neighbors_per_cell", 2)),
                "max_neighbors_per_domain": self.retrieval_cfg.get("max_neighbors_per_domain"),
            },
            retrieval_temperature=float(self.retrieval_cfg.get("retrieval_temperature", 0.1)),
        )
        cache_path = self._retrieval_cache_path(retriever)
        if not cache_path.exists():
            if not auto_build_retrieval_cache:
                raise FileNotFoundError(f"Missing retrieval cache: {cache_path}")
            retriever.build_cache(self.selected_case_ids.tolist(), cache_path)
        cache = np.load(cache_path, allow_pickle=True)
        config_hash = str(cache["config_hash"][0])
        if config_hash != retriever.retrieval_config_hash():
            raise ValueError(f"Retrieval cache hash mismatch: {cache_path}")
        return {key: cache[key] for key in cache.files}

    def __len__(self) -> int:
        return int(len(self.indices))

    def _zero_retrieval_dict(self, horizon: int, top_k: int) -> Dict[str, np.ndarray]:
        zero_component = np.zeros((top_k, len(COMPONENT_NAMES)), dtype=np.float32)
        return {
            "neighbor_case_ids": np.full(top_k, -1, dtype=np.int64),
            "retrieval_mask": np.zeros(top_k, dtype=np.float32),
            "retrieval_alpha": np.zeros(top_k, dtype=np.float32),
            "retrieval_confidence": np.asarray(0.0, dtype=np.float32),
            "component_distances": zero_component,
            "composite_distance": np.full(top_k, np.inf, dtype=np.float32),
            "d_soh_state": np.zeros(top_k, dtype=np.float32),
            "d_qv_shape": np.zeros(top_k, dtype=np.float32),
            "d_physics": np.zeros(top_k, dtype=np.float32),
            "d_operation": np.zeros(top_k, dtype=np.float32),
            "d_metadata": np.zeros(top_k, dtype=np.float32),
            "reference_compatibility_score": np.zeros(top_k, dtype=np.float32),
            "ref_future_delta_soh": np.zeros((top_k, horizon), dtype=np.float32),
            "ref_cycle_stats": np.zeros((top_k,) + self.arrays["cycle_stats"].shape[1:], dtype=np.float32),
            "ref_soh_seq": np.zeros((top_k, self.arrays["soh_seq"].shape[1], 1), dtype=np.float32),
            "ref_anchor_soh": np.zeros(top_k, dtype=np.float32),
            "ref_chemistry_id": np.zeros(top_k, dtype=np.int64),
            "ref_qv_maps": np.zeros((top_k,) + self.arrays["qv_maps"].shape[1:], dtype=np.float32),
            "ref_partial_charge": np.zeros((top_k,) + self.arrays["partial_charge"].shape[1:], dtype=np.float32),
            "ref_physics_features": np.zeros((top_k,) + self.arrays["physics_features"].shape[1:], dtype=np.float32),
            "ref_anchor_physics_features": np.zeros((top_k,) + self.arrays["anchor_physics_features"].shape[1:], dtype=np.float32),
            "ref_operation_seq": np.zeros((top_k,) + self.arrays["operation_seq"].shape[1:], dtype=np.float32),
            "ref_meta_ids": np.zeros((top_k, len(self.meta_fields)), dtype=np.int64),
        }

    def __getitem__(self, item: int) -> Dict[str, Dict[str, np.ndarray | int | str]]:
        row_idx = int(self.indices[item])
        row = self.case_rows.iloc[row_idx]
        case_id = int(row["case_id"])
        horizon = int(self.arrays["future_delta_soh"].shape[1])

        query = {
            "case_id": case_id,
            "cell_uid": str(row["cell_uid"]),
            "split": str(row["split"]),
            "anchor_soh": np.asarray(float(row["anchor_soh"]), dtype=np.float32),
            "chemistry_id": np.asarray(chemistry_family_to_id(row.get("chemistry_family", "unknown")), dtype=np.int64),
            "target_delta_soh": np.asarray(self.arrays["future_delta_soh"][row_idx], dtype=np.float32),
            "target_soh": np.asarray(self.arrays["future_soh"][row_idx], dtype=np.float32),
            "cycle_stats": np.asarray(self.arrays["cycle_stats"][row_idx], dtype=np.float32),
            "soh_seq": np.asarray(self.arrays["soh_seq"][row_idx][:, None], dtype=np.float32),
            "qv_maps": np.asarray(self.arrays["qv_maps"][row_idx], dtype=np.float32),
            "qv_masks": np.asarray(self.arrays["qv_masks"][row_idx], dtype=np.float32),
            "partial_charge": np.asarray(self.arrays["partial_charge"][row_idx], dtype=np.float32),
            "partial_charge_mask": np.asarray(self.arrays["partial_charge_mask"][row_idx], dtype=np.float32),
            "physics_features": np.asarray(self.arrays["physics_features"][row_idx], dtype=np.float32),
            "physics_feature_masks": np.asarray(self.arrays["physics_feature_masks"][row_idx], dtype=np.float32),
            "anchor_physics_features": np.asarray(self.arrays["anchor_physics_features"][row_idx], dtype=np.float32),
            "operation_seq": np.asarray(self.arrays["operation_seq"][row_idx], dtype=np.float32),
            "future_ops": np.asarray(self.arrays["future_ops"][row_idx], dtype=np.float32),
            "future_ops_mask": np.asarray(self.arrays["future_ops_mask"][row_idx], dtype=np.float32),
            "meta_ids": np.asarray(self.meta_ids_all[row_idx], dtype=np.int64),
        }
        if self.arrays["expert_seq"] is not None:
            query["expert_seq"] = np.asarray(self.arrays["expert_seq"][row_idx], dtype=np.float32)
        if self.arrays["baseline_delta_oof"] is not None:
            query["baseline_delta_oof"] = np.asarray(self.arrays["baseline_delta_oof"][row_idx], dtype=np.float32)
        if self.arrays["residual_target_oof"] is not None:
            query["residual_target_oof"] = np.asarray(self.arrays["residual_target_oof"][row_idx], dtype=np.float32)
        factor_scores = self.arrays.get("residual_factor_scores")
        if factor_scores is not None:
            query["residual_factor_scores"] = np.asarray(factor_scores[row_idx], dtype=np.float32)

        if self.retrieval_cache is None:
            top_k = int(self.retrieval_cfg.get("top_k", 8)) if self.retrieval_cfg else 8
            retrieval = self._zero_retrieval_dict(horizon=horizon, top_k=top_k)
        else:
            local_idx = self.query_case_id_to_local[case_id]
            neighbor_case_ids = np.asarray(self.retrieval_cache["neighbor_case_ids"][local_idx], dtype=np.int64)
            retrieval_mask = np.asarray(self.retrieval_cache["retrieval_mask"][local_idx], dtype=np.float32)
            component_distances = np.asarray(self.retrieval_cache["component_distances"][local_idx], dtype=np.float32)
            ref_indices = np.where(neighbor_case_ids >= 0, neighbor_case_ids, 0).astype(np.int64)
            ref_row_indices = np.asarray([self.case_id_to_row_index[int(case_id)] for case_id in ref_indices.tolist()], dtype=np.int64)
            retrieval = {
                "neighbor_case_ids": neighbor_case_ids,
                "retrieval_mask": retrieval_mask,
                "retrieval_alpha": np.asarray(self.retrieval_cache["retrieval_alpha"][local_idx], dtype=np.float32),
                "retrieval_confidence": np.asarray(self.retrieval_cache["retrieval_confidence"][local_idx], dtype=np.float32),
                "component_distances": component_distances,
                "composite_distance": np.asarray(
                    self.retrieval_cache["composite_distance"][local_idx]
                    if "composite_distance" in self.retrieval_cache
                    else self.retrieval_cache["composite_distances"][local_idx],
                    dtype=np.float32,
                ),
                "d_soh_state": np.asarray(
                    self.retrieval_cache["d_soh_state"][local_idx] if "d_soh_state" in self.retrieval_cache else component_distances[:, 0],
                    dtype=np.float32,
                ),
                "d_qv_shape": np.asarray(
                    self.retrieval_cache["d_qv_shape"][local_idx] if "d_qv_shape" in self.retrieval_cache else component_distances[:, 1],
                    dtype=np.float32,
                ),
                "d_physics": np.asarray(
                    self.retrieval_cache["d_physics"][local_idx] if "d_physics" in self.retrieval_cache else component_distances[:, 2],
                    dtype=np.float32,
                ),
                "d_operation": np.asarray(
                    self.retrieval_cache["d_operation"][local_idx] if "d_operation" in self.retrieval_cache else component_distances[:, 3],
                    dtype=np.float32,
                ),
                "d_metadata": np.asarray(
                    self.retrieval_cache["d_metadata"][local_idx] if "d_metadata" in self.retrieval_cache else component_distances[:, 4],
                    dtype=np.float32,
                ),
                "reference_compatibility_score": np.asarray(self.retrieval_cache["reference_compatibility_score"][local_idx], dtype=np.float32),
                "ref_future_delta_soh": np.asarray(self.arrays["future_delta_soh"][ref_row_indices], dtype=np.float32),
                "ref_cycle_stats": np.asarray(self.arrays["cycle_stats"][ref_row_indices], dtype=np.float32),
                "ref_soh_seq": np.asarray(self.arrays["soh_seq"][ref_row_indices][:, :, None], dtype=np.float32),
                "ref_anchor_soh": np.asarray(self.case_rows.iloc[ref_row_indices]["anchor_soh"].to_numpy(dtype=np.float32), dtype=np.float32),
                "ref_chemistry_id": np.asarray(
                    [chemistry_family_to_id(value) for value in self.case_rows.iloc[ref_row_indices]["chemistry_family"].tolist()],
                    dtype=np.int64,
                ),
                "ref_qv_maps": np.asarray(self.arrays["qv_maps"][ref_row_indices], dtype=np.float32),
                "ref_partial_charge": np.asarray(self.arrays["partial_charge"][ref_row_indices], dtype=np.float32),
                "ref_physics_features": np.asarray(self.arrays["physics_features"][ref_row_indices], dtype=np.float32),
                "ref_anchor_physics_features": np.asarray(self.arrays["anchor_physics_features"][ref_row_indices], dtype=np.float32),
                "ref_operation_seq": np.asarray(self.arrays["operation_seq"][ref_row_indices], dtype=np.float32),
                "ref_meta_ids": np.asarray(self.meta_ids_all[ref_row_indices], dtype=np.int64),
            }
        return {"query": query, "retrieval": retrieval}
