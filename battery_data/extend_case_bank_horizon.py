"""Extend an existing case bank to a longer future SOH horizon.

The historical input window is unchanged. This utility reuses all named
history-side numerical features from an existing case bank and rebuilds only:

- `case_future_soh.npy`
- `case_future_delta_soh.npy`
- `case_future_ops.npy`
- `case_future_ops_mask.npy`
- target-horizon metadata in `case_rows.parquet`

It is intended for controlled experiments such as 32 observed historical cycles
to 64 future SOH steps, where re-extracting Q-V/partial-charge features from raw
cycle files would be unnecessary and slow.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


HISTORY_ARRAY_FILES = [
    "case_cycle_stats.npy",
    "case_soh_seq.npy",
    "case_qv_maps.npy",
    "case_qv_masks.npy",
    "case_partial_charge.npy",
    "case_partial_charge_mask.npy",
    "case_physics_features.npy",
    "case_physics_feature_masks.npy",
    "case_anchor_physics_features.npy",
    "case_operation_seq.npy",
    "case_expert_seq.npy",
]


def _read_rows(case_bank_dir: Path) -> pd.DataFrame:
    parquet_path = case_bank_dir / "case_rows.parquet"
    csv_path = case_bank_dir / "case_rows.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path).sort_values("case_id").reset_index(drop=True)
    if csv_path.exists():
        return pd.read_csv(csv_path).sort_values("case_id").reset_index(drop=True)
    raise FileNotFoundError(f"Missing case rows under {case_bank_dir}")


def _save_rows(rows: pd.DataFrame, output_dir: Path) -> None:
    try:
        rows.to_parquet(output_dir / "case_rows.parquet", index=False)
    except Exception:
        rows.to_csv(output_dir / "case_rows.csv", index=False)


def _copy_metadata_files(input_dir: Path, output_dir: Path) -> None:
    for name in [
        "feature_names.json",
        "case_expert_seq_feature_names.json",
        "normalization_stats.json",
    ]:
        src = input_dir / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def _collect_cell_truth_and_ops(
    rows: pd.DataFrame,
    soh_seq: np.ndarray,
    future_soh: np.ndarray,
    operation_seq: np.ndarray,
    future_ops: np.ndarray,
) -> tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, np.ndarray]]]:
    truth: Dict[str, Dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    ops: Dict[str, Dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    for row_idx, row in rows.iterrows():
        cell_uid = str(row["cell_uid"])
        hist_start = int(row["cycle_idx_start"])
        future_start = int(row["target_cycle_idx_start"])

        hist_soh = np.asarray(soh_seq[row_idx], dtype=np.float32).reshape(-1)
        hist_ops = np.asarray(operation_seq[row_idx], dtype=np.float32)
        for offset, value in enumerate(hist_soh):
            cycle_idx = hist_start + offset
            if np.isfinite(value):
                truth[cell_uid][cycle_idx].append(float(value))
            if offset < hist_ops.shape[0] and np.isfinite(hist_ops[offset]).any():
                ops[cell_uid][cycle_idx].append(np.nan_to_num(hist_ops[offset], nan=0.0).astype(np.float32))

        fut_soh = np.asarray(future_soh[row_idx], dtype=np.float32).reshape(-1)
        fut_ops = np.asarray(future_ops[row_idx], dtype=np.float32)
        for offset, value in enumerate(fut_soh):
            cycle_idx = future_start + offset
            if np.isfinite(value):
                truth[cell_uid][cycle_idx].append(float(value))
            if offset < fut_ops.shape[0] and np.isfinite(fut_ops[offset]).any():
                ops[cell_uid][cycle_idx].append(np.nan_to_num(fut_ops[offset], nan=0.0).astype(np.float32))

    truth_mean = {
        cell_uid: {cycle_idx: float(np.mean(values)) for cycle_idx, values in cycle_map.items()}
        for cell_uid, cycle_map in truth.items()
    }
    ops_mean = {
        cell_uid: {cycle_idx: np.mean(np.stack(values, axis=0), axis=0).astype(np.float32) for cycle_idx, values in cycle_map.items()}
        for cell_uid, cycle_map in ops.items()
    }
    return truth_mean, ops_mean


def _eligible_cases(
    rows: pd.DataFrame,
    truth_by_cell: Dict[str, Dict[int, float]],
    future_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    selected_indices: List[int] = []
    new_future_soh: List[np.ndarray] = []
    for row_idx, row in rows.iterrows():
        cell_uid = str(row["cell_uid"])
        start_cycle = int(row["target_cycle_idx_start"])
        cycle_values = []
        cell_truth = truth_by_cell.get(cell_uid, {})
        for offset in range(int(future_length)):
            cycle_idx = start_cycle + offset
            if cycle_idx not in cell_truth:
                cycle_values = []
                break
            cycle_values.append(float(cell_truth[cycle_idx]))
        if cycle_values:
            selected_indices.append(int(row_idx))
            new_future_soh.append(np.asarray(cycle_values, dtype=np.float32))
    if not selected_indices:
        raise ValueError(f"No cases have {future_length} consecutive future SOH values.")
    return np.asarray(selected_indices, dtype=np.int64), np.stack(new_future_soh, axis=0).astype(np.float32)


def _build_future_ops(
    rows: pd.DataFrame,
    selected_indices: np.ndarray,
    ops_by_cell: Dict[str, Dict[int, np.ndarray]],
    operation_seq: np.ndarray,
    future_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    op_dim = int(operation_seq.shape[-1])
    future_ops = np.zeros((len(selected_indices), int(future_length), op_dim), dtype=np.float32)
    future_mask = np.zeros_like(future_ops, dtype=np.float32)
    for out_idx, row_idx in enumerate(selected_indices.tolist()):
        row = rows.iloc[int(row_idx)]
        cell_uid = str(row["cell_uid"])
        start_cycle = int(row["target_cycle_idx_start"])
        fallback = np.nan_to_num(operation_seq[int(row_idx), -1], nan=0.0).astype(np.float32)
        cell_ops = ops_by_cell.get(cell_uid, {})
        for offset in range(int(future_length)):
            cycle_idx = start_cycle + offset
            value = cell_ops.get(cycle_idx)
            if value is None:
                future_ops[out_idx, offset] = fallback
            else:
                future_ops[out_idx, offset] = np.asarray(value, dtype=np.float32)
                future_mask[out_idx, offset] = np.isfinite(value).astype(np.float32)
    return np.nan_to_num(future_ops, nan=0.0).astype(np.float32), future_mask.astype(np.float32)


def _copy_selected_array(input_path: Path, output_path: Path, selected_indices: np.ndarray, chunk_size: int = 512) -> None:
    source = np.load(input_path, mmap_mode="r")
    destination = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=source.dtype,
        shape=(len(selected_indices), *source.shape[1:]),
    )
    for start in range(0, len(selected_indices), int(chunk_size)):
        end = min(start + int(chunk_size), len(selected_indices))
        destination[start:end] = source[selected_indices[start:end]]
    destination.flush()


def _fit_normalization_stats(rows: pd.DataFrame, output_dir: Path) -> Dict[str, object]:
    arrays = {
        "cycle_stats": np.load(output_dir / "case_cycle_stats.npy", mmap_mode="r"),
        "soh_seq": np.load(output_dir / "case_soh_seq.npy", mmap_mode="r"),
        "physics_features": np.load(output_dir / "case_physics_features.npy", mmap_mode="r"),
        "anchor_physics_features": np.load(output_dir / "case_anchor_physics_features.npy", mmap_mode="r"),
        "operation_seq": np.load(output_dir / "case_operation_seq.npy", mmap_mode="r"),
        "future_ops": np.load(output_dir / "case_future_ops.npy", mmap_mode="r"),
    }
    source_idx = rows.index[rows["split"].astype(str) == "source_train"].to_numpy(dtype=np.int64)
    stats: Dict[str, object] = {"fit_split": "source_train", "num_fit_cases": int(len(source_idx))}
    for name, values in arrays.items():
        arr = np.asarray(values[source_idx] if len(source_idx) else values, dtype=np.float32).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            stats[name] = {"mean": 0.0, "std": 1.0, "q05": 0.0, "q50": 0.0, "q95": 0.0}
        else:
            std = float(finite.std())
            stats[name] = {
                "mean": float(finite.mean()),
                "std": std if std > 1e-8 else 1.0,
                "q05": float(np.quantile(finite, 0.05)),
                "q50": float(np.quantile(finite, 0.50)),
                "q95": float(np.quantile(finite, 0.95)),
            }
    return stats


def extend_case_bank_horizon(
    input_dir: str | Path,
    output_dir: str | Path,
    future_length: int,
    overwrite: bool = False,
) -> Dict[str, object]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if output_path.exists() and overwrite:
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_path)
    soh_seq = np.load(input_path / "case_soh_seq.npy", mmap_mode="r")
    future_soh_old = np.load(input_path / "case_future_soh.npy", mmap_mode="r")
    operation_seq = np.load(input_path / "case_operation_seq.npy", mmap_mode="r")
    future_ops_old = np.load(input_path / "case_future_ops.npy", mmap_mode="r")
    truth_by_cell, ops_by_cell = _collect_cell_truth_and_ops(rows, soh_seq, future_soh_old, operation_seq, future_ops_old)
    selected_indices, future_soh = _eligible_cases(rows, truth_by_cell, int(future_length))

    selected_rows = rows.iloc[selected_indices].copy().reset_index(drop=True)
    selected_rows["case_id"] = np.arange(len(selected_rows), dtype=np.int64)
    selected_rows["target_horizon"] = int(future_length)
    selected_rows["target_end"] = selected_rows["target_start"].astype(int) + int(future_length)
    selected_rows["target_cycle_idx_end"] = selected_rows["target_cycle_idx_start"].astype(int) + int(future_length) - 1
    selected_rows["future_operation_dim"] = int(operation_seq.shape[-1])

    for name in HISTORY_ARRAY_FILES:
        _copy_selected_array(input_path / name, output_path / name, selected_indices)

    future_delta = future_soh - selected_rows["anchor_soh"].to_numpy(dtype=np.float32)[:, None]
    future_ops, future_ops_mask = _build_future_ops(rows, selected_indices, ops_by_cell, operation_seq, int(future_length))
    np.save(output_path / "case_future_soh.npy", future_soh.astype(np.float32))
    np.save(output_path / "case_future_delta_soh.npy", future_delta.astype(np.float32))
    np.save(output_path / "case_future_ops.npy", future_ops.astype(np.float32))
    np.save(output_path / "case_future_ops_mask.npy", future_ops_mask.astype(np.float32))

    _save_rows(selected_rows, output_path)
    _copy_metadata_files(input_path, output_path)
    normalization_stats = _fit_normalization_stats(selected_rows, output_path)
    (output_path / "normalization_stats.json").write_text(json.dumps(normalization_stats, indent=2, ensure_ascii=True))
    build_log = {
        "source_case_bank_dir": str(input_path),
        "future_length": int(future_length),
        "total_cases": int(len(selected_rows)),
        "cases_by_split": {str(k): int(v) for k, v in selected_rows["split"].value_counts().to_dict().items()},
        "cases_by_dataset": {str(k): int(v) for k, v in selected_rows["source_dataset"].value_counts().to_dict().items()},
        "cases_by_chemistry": {str(k): int(v) for k, v in selected_rows["chemistry_family"].value_counts().to_dict().items()},
        "future_ops_availability": float(future_ops_mask.mean()) if future_ops_mask.size else 0.0,
        "normalization_stats": normalization_stats,
    }
    (output_path / "case_bank_build_log.json").write_text(json.dumps(build_log, indent=2, ensure_ascii=True))
    pd.DataFrame([{"key": key, "value_json": json.dumps(value, ensure_ascii=True)} for key, value in build_log.items()]).to_csv(
        output_path / "case_bank_build_log.csv",
        index=False,
    )
    return build_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Extend a case bank to a longer future SOH horizon.")
    parser.add_argument("--input-dir", default="output/case_bank_stratified_fewshot")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--future-length", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = extend_case_bank_horizon(args.input_dir, args.output_dir, args.future_length, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
