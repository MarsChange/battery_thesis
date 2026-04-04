from __future__ import annotations

from typing import Dict, Iterable, List, Set

import numpy as np
import pandas as pd

from .domain_labeling import build_cell_level_domain_label
from .features import parse_json_list
from .schema import CanonicalCell


def build_cell_catalog(
    cells: Iterable[CanonicalCell],
    domain_rules: Dict[str, object] | None = None,
) -> pd.DataFrame:
    rows = []
    for cell in cells:
        representative = cell.cycles.iloc[min(len(cell.cycles) - 1, 0)]
        rep_meta = {
            "chemistry_family": representative.get("chemistry_family"),
            "temperature_bucket": representative.get("temperature_bucket"),
            "charge_rate_bucket": representative.get("charge_rate_bucket"),
            "discharge_policy_family": representative.get("discharge_policy_family"),
            "full_or_partial": representative.get("full_or_partial"),
            "nominal_capacity_bucket": representative.get("nominal_capacity_bucket"),
            "voltage_window_bucket": representative.get("voltage_window_bucket"),
            "missing_mask": parse_json_list(representative.get("missing_mask")),
        }
        rows.append(
            {
                "cell_uid": representative["cell_uid"],
                "source_dataset": cell.source_dataset,
                "raw_cell_id": cell.raw_cell_id,
                "file_path": cell.file_path,
                "n_cycles": int(len(cell.cycles)),
                "domain_label": build_cell_level_domain_label(rep_meta, domain_rules),
            }
        )
    return pd.DataFrame(rows).sort_values(["source_dataset", "raw_cell_id"]).reset_index(drop=True)


def _sample_ids(ids: List[str], count: int, rng: np.random.RandomState) -> Set[str]:
    if count <= 0 or not ids:
        return set()
    count = min(count, len(ids))
    chosen = rng.choice(ids, size=count, replace=False)
    return set(chosen.tolist())


def _validation_ids(
    source_df: pd.DataFrame,
    val_frac: float,
    rng: np.random.RandomState,
) -> Set[str]:
    val_ids: Set[str] = set()
    if val_frac <= 0:
        return val_ids
    for _, group in source_df.groupby("source_dataset"):
        ids = group["cell_uid"].tolist()
        if len(ids) <= 1:
            continue
        n_val = int(round(len(ids) * val_frac))
        if n_val == 0:
            n_val = 1
        if n_val >= len(ids):
            n_val = len(ids) - 1
        val_ids.update(_sample_ids(ids, n_val, rng))
    return val_ids


def build_split_manifest(
    cells: Iterable[CanonicalCell],
    split_cfg: Dict[str, object],
    domain_rules: Dict[str, object] | None = None,
) -> pd.DataFrame:
    catalog = build_cell_catalog(cells, domain_rules)
    if catalog.empty:
        raise ValueError("No cells were loaded; cannot build split manifest.")

    strategy = split_cfg.get("strategy", "dataset-held-out")
    rng = np.random.RandomState(int(split_cfg.get("random_seed", 7)))
    source_val_frac = float(split_cfg.get("source_val_frac", 0.2))
    few_shot_k_cells = int(split_cfg.get("few_shot_k_cells", 0))

    if strategy == "default-main":
        source_datasets = set(split_cfg.get("source_datasets", ["xjtu", "hust", "tju"]))
        target_datasets = set(split_cfg.get("target_datasets", ["mit"]))
        source_mask = catalog["source_dataset"].isin(source_datasets)
        target_mask = catalog["source_dataset"].isin(target_datasets)
    elif strategy == "dataset-held-out":
        held_out = set(split_cfg.get("held_out_datasets", split_cfg.get("target_datasets", ["mit"])))
        target_mask = catalog["source_dataset"].isin(held_out)
        source_mask = ~target_mask
    elif strategy == "leave-one-domain-out":
        held_out_domains = set(split_cfg.get("held_out_domains", []))
        if not held_out_domains:
            raise ValueError("leave-one-domain-out requires split.held_out_domains")
        target_mask = catalog["domain_label"].isin(held_out_domains)
        source_mask = ~target_mask
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")

    if not target_mask.any():
        raise ValueError("Split config produced an empty target set.")
    if not source_mask.any():
        raise ValueError("Split config produced an empty source set.")

    manifest = catalog.copy()
    manifest["split"] = "unassigned"

    source_df = manifest[source_mask].copy()
    target_df = manifest[target_mask].copy()

    val_ids = _validation_ids(source_df, source_val_frac, rng)
    support_ids = _sample_ids(target_df["cell_uid"].tolist(), few_shot_k_cells, rng)

    manifest.loc[source_mask, "split"] = "source_train"
    manifest.loc[manifest["cell_uid"].isin(val_ids), "split"] = "source_val"
    manifest.loc[target_mask, "split"] = "target_query"
    manifest.loc[manifest["cell_uid"].isin(support_ids), "split"] = "target_support"

    return manifest.sort_values(["split", "source_dataset", "raw_cell_id"]).reset_index(drop=True)


def assert_no_split_leakage(manifest: pd.DataFrame) -> None:
    duplicates = manifest.groupby("cell_uid")["split"].nunique()
    if (duplicates > 1).any():
        dup_ids = duplicates[duplicates > 1].index.tolist()
        raise AssertionError(f"Cells appear in multiple splits: {dup_ids[:5]}")
    source_train = set(manifest.loc[manifest["split"] == "source_train", "cell_uid"])
    target_query = set(manifest.loc[manifest["split"] == "target_query", "cell_uid"])
    overlap = source_train & target_query
    if overlap:
        raise AssertionError(f"source_train and target_query overlap: {sorted(overlap)[:5]}")
