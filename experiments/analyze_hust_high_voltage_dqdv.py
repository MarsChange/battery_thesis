"""Analyze HUST/LFP high-voltage dQ/dV reranking.

This script reuses an existing subset retrieval result and asks a focused
question: if HUST/LFP top-k candidates are reranked only by the dQ/dV curve
after a voltage threshold, does the top-1 neighbor become a better SOH-trajectory
match?
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_subset_rag_retrieval import (
    _dqdv_distance,
    _filter_dqdv_curve,
    _load_case_curve,
    _plot_dqdv_comparison,
    _plot_soh_comparison,
)


def _plot_high_voltage_dqdv_comparison(
    *,
    query_row: pd.Series,
    neighbor_rows: Sequence[pd.Series],
    curve_lookup,
    voltage_min: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.8, 5.2), dpi=180)

    query_curve = curve_lookup(int(query_row["case_id"]))
    if query_curve is not None:
        axis.plot(query_curve[0], query_curve[1], color="#dc2626", linewidth=2.4, label=f"Query | {query_row['cell_uid']}")
    else:
        axis.text(0.03, 0.95, f"Query V>={voltage_min:.1f}V dQ/dV unavailable", transform=axis.transAxes, va="top", ha="left", color="#dc2626")

    palette = ["#2563eb", "#16a34a", "#9333ea"]
    styles = ["--", "-", "-."]
    for rank, (neighbor_row, color, linestyle) in enumerate(zip(neighbor_rows, palette, styles), start=1):
        ref_curve = curve_lookup(int(neighbor_row["case_id"]))
        if ref_curve is None:
            continue
        label_suffix = f"{neighbor_row['cell_uid']} | {neighbor_row['source_dataset']}"
        axis.plot(ref_curve[0], ref_curve[1], color=color, linewidth=1.8, linestyle=linestyle, label=f"Top-{rank} | {label_suffix}")

    axis.axvline(voltage_min, color="#374151", linewidth=1.0, linestyle=":", label=f"V threshold = {voltage_min:.1f} V")
    axis.set_title(f"High-voltage dQ-dV comparison for cross-cell top-k retrieval (V >= {voltage_min:.1f} V)")
    axis.set_xlabel("Voltage (V)")
    axis.set_ylabel("dQ/dV")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    *,
    artifact_dir: Path,
    setting_name: str,
    voltage_min: float,
) -> dict[str, object]:
    case_bank = artifact_dir / "case_bank"
    pair_path = artifact_dir / "settings" / setting_name / "query_topk_similarity.csv"
    output_dir = artifact_dir / "hust_high_voltage_dqdv_analysis"
    condition_setting_name = f"{setting_name}_hust_vge_{str(voltage_min).replace('.', 'p')}_dqdv"
    condition_setting_dir = artifact_dir / "settings" / condition_setting_name
    condition_figure_dir = condition_setting_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_figure_dir.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(case_bank / "case_rows.parquet").sort_values("case_id").reset_index(drop=True)
    id_to_row = {int(row["case_id"]): row for _, row in rows.iterrows()}
    pair_df = pd.read_csv(pair_path)
    hust_df = pair_df.loc[pair_df["query_source_dataset"].astype(str) == "hust_lfp"].copy()

    curve_cache: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}

    def curve(case_id: int) -> tuple[np.ndarray, np.ndarray] | None:
        case_id = int(case_id)
        if case_id not in curve_cache:
            row = id_to_row.get(case_id)
            curve_cache[case_id] = None if row is None else _filter_dqdv_curve(_load_case_curve(row)[1], voltage_min=voltage_min)
        return curve_cache[case_id]

    records = []
    for _, row in hust_df.iterrows():
        record = row.to_dict()
        record["dqdv_high_voltage_distance"] = _dqdv_distance(
            curve(int(row["query_case_id"])),
            curve(int(row["neighbor_case_id"])),
            voltage_min=voltage_min,
        )
        records.append(record)
    scored_df = pd.DataFrame(records)

    reranked_rows = []
    for query_case_id, group in scored_df.groupby("query_case_id", sort=True):
        ranked = group.sort_values(["dqdv_high_voltage_distance", "composite_distance"], na_position="last").reset_index(drop=True)
        original_top1 = group.sort_values("neighbor_rank").iloc[0]
        new_top1 = ranked.iloc[0]
        for new_rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            record = row.to_dict()
            record["high_voltage_rank"] = int(new_rank)
            record["original_top1_case_id"] = int(original_top1["neighbor_case_id"])
            record["high_voltage_top1_case_id"] = int(new_top1["neighbor_case_id"])
            record["top1_changed_by_high_voltage"] = int(int(original_top1["neighbor_case_id"]) != int(new_top1["neighbor_case_id"]))
            reranked_rows.append(record)
    reranked_df = pd.DataFrame(reranked_rows)
    reranked_df.to_csv(output_dir / "hust_topk_high_voltage_dqdv_rerank.csv", index=False)
    condition_pair_df = reranked_df.sort_values(["query_case_id", "high_voltage_rank"]).copy()
    condition_pair_df["setting_name"] = condition_setting_name
    condition_pair_df["neighbor_rank"] = condition_pair_df["high_voltage_rank"].astype(int)
    condition_pair_df.to_csv(condition_setting_dir / "query_topk_similarity.csv", index=False)

    summary_rows = []
    for query_case_id, group in reranked_df.groupby("query_case_id", sort=True):
        original = group.sort_values("neighbor_rank").iloc[0]
        hv = group.sort_values("high_voltage_rank").iloc[0]
        summary_rows.append(
            {
                "query_case_id": int(query_case_id),
                "query_cell_uid": str(original["query_cell_uid"]),
                "original_top1_case_id": int(original["neighbor_case_id"]),
                "original_top1_cell_uid": str(original["neighbor_cell_uid"]),
                "original_top1_high_voltage_dqdv": float(original["dqdv_high_voltage_distance"]),
                "original_top1_full_soh_rmse": float(original["full_soh_rmse"]),
                "hv_top1_case_id": int(hv["neighbor_case_id"]),
                "hv_top1_cell_uid": str(hv["neighbor_cell_uid"]),
                "hv_top1_original_rank": int(hv["neighbor_rank"]),
                "hv_top1_high_voltage_dqdv": float(hv["dqdv_high_voltage_distance"]),
                "hv_top1_full_soh_rmse": float(hv["full_soh_rmse"]),
                "top1_changed": int(int(original["neighbor_case_id"]) != int(hv["neighbor_case_id"])),
                "full_soh_rmse_improved": int(float(hv["full_soh_rmse"]) < float(original["full_soh_rmse"])),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "hust_high_voltage_dqdv_summary.csv", index=False)
    comparison_df = pd.DataFrame(
        [
            {
                "setting_name": setting_name,
                "description": "original state_metadata_qv_physics top-1",
                "num_hust_queries": int(len(summary_df)),
                "top1_mean_full_soh_rmse": float(summary_df["original_top1_full_soh_rmse"].mean()),
                "top1_mean_high_voltage_dqdv": float(summary_df["original_top1_high_voltage_dqdv"].mean()),
                "top1_changed_from_baseline": 0,
                "top1_improved_vs_baseline": 0,
            },
            {
                "setting_name": condition_setting_name,
                "description": f"HUST/LFP top-k reranked by V>={voltage_min:.2f}V dQ/dV",
                "num_hust_queries": int(len(summary_df)),
                "top1_mean_full_soh_rmse": float(summary_df["hv_top1_full_soh_rmse"].mean()),
                "top1_mean_high_voltage_dqdv": float(summary_df["hv_top1_high_voltage_dqdv"].mean()),
                "top1_changed_from_baseline": int(summary_df["top1_changed"].sum()),
                "top1_improved_vs_baseline": int(summary_df["full_soh_rmse_improved"].sum()),
            },
        ]
    )
    comparison_df.to_csv(output_dir / "baseline_vs_high_voltage_condition.csv", index=False)

    condition_query_df = summary_df.rename(
        columns={
            "hv_top1_case_id": "top1_case_id",
            "hv_top1_cell_uid": "top1_cell_uid",
            "hv_top1_full_soh_rmse": "top1_full_soh_rmse",
            "hv_top1_high_voltage_dqdv": "top1_dqdv_high_voltage_distance",
        }
    ).copy()
    condition_query_df["setting_name"] = condition_setting_name
    condition_query_df["source_setting_name"] = setting_name
    condition_query_df["voltage_min"] = float(voltage_min)
    condition_query_df.to_csv(condition_setting_dir / "query_summary.csv", index=False)

    generated_figure_paths: list[str] = []
    for query_case_id, group in condition_pair_df.groupby("query_case_id", sort=True):
        query_row = id_to_row.get(int(query_case_id))
        if query_row is None:
            continue
        neighbor_rows = []
        for neighbor_case_id in group.sort_values("neighbor_rank")["neighbor_case_id"].astype(int).tolist()[:3]:
            neighbor_row = id_to_row.get(int(neighbor_case_id))
            if neighbor_row is not None:
                neighbor_rows.append(neighbor_row)
        if not neighbor_rows:
            continue
        stem = f"query_case_{int(query_case_id):05d}_{str(query_row['cell_uid'])}"
        soh_path = condition_figure_dir / f"{stem}_soh.png"
        dqdv_path = condition_figure_dir / f"{stem}_dqdv.png"
        high_voltage_dqdv_path = condition_figure_dir / f"{stem}_dqdv_vge_{str(voltage_min).replace('.', 'p')}.png"
        _plot_soh_comparison(query_row, neighbor_rows, soh_path)
        _plot_dqdv_comparison(query_row, neighbor_rows, dqdv_path)
        _plot_high_voltage_dqdv_comparison(
            query_row=query_row,
            neighbor_rows=neighbor_rows,
            curve_lookup=curve,
            voltage_min=voltage_min,
            output_path=high_voltage_dqdv_path,
        )
        generated_figure_paths.extend([str(soh_path), str(dqdv_path), str(high_voltage_dqdv_path)])

    figure, axis = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    x = np.arange(len(summary_df))
    axis.bar(x - 0.18, summary_df["original_top1_full_soh_rmse"], width=0.36, label="original top-1", color="#2563eb")
    axis.bar(x + 0.18, summary_df["hv_top1_full_soh_rmse"], width=0.36, label=f"{voltage_min:.1f}V+ dQ/dV top-1", color="#f97316")
    axis.set_xticks(x)
    axis.set_xticklabels(summary_df["query_cell_uid"], rotation=25, ha="right")
    axis.set_ylabel("Full-life SOH RMSE")
    axis.set_title(f"HUST/LFP top-1 after {voltage_min:.1f}V+ dQ/dV reranking within original top-k")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "hust_high_voltage_dqdv_top1_rmse.png", bbox_inches="tight")
    figure.savefig(condition_figure_dir / "hust_high_voltage_dqdv_top1_rmse.png", bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# HUST high-voltage dQ/dV rerank analysis",
        "",
        f"- Input setting: `{setting_name}`",
        f"- Voltage window: V >= {voltage_min:.2f} V",
        f"- HUST queries: {len(summary_df)}",
        f"- Top-1 changed by high-voltage dQ/dV: {int(summary_df['top1_changed'].sum())}/{len(summary_df)}",
        f"- Full-life SOH RMSE improved after high-voltage rerank: {int(summary_df['full_soh_rmse_improved'].sum())}/{len(summary_df)}",
        f"- Original mean top-1 RMSE: {summary_df['original_top1_full_soh_rmse'].mean():.6f}",
        f"- High-voltage rerank mean top-1 RMSE: {summary_df['hv_top1_full_soh_rmse'].mean():.6f}",
        "",
        "## Per-query",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- query {int(row['query_case_id'])} {row['query_cell_uid']}: "
            f"original {row['original_top1_cell_uid']} RMSE={row['original_top1_full_soh_rmse']:.6f}, "
            f"high-voltage {row['hv_top1_cell_uid']} original-rank={int(row['hv_top1_original_rank'])} "
            f"RMSE={row['hv_top1_full_soh_rmse']:.6f}, "
            f"high-voltage dQ/dV={row['hv_top1_high_voltage_dqdv']:.6f}"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    condition_lines = [
        f"# Retrieval Setting: {condition_setting_name}",
        "",
        "This is an additional HUST/LFP comparison setting derived from `state_metadata_qv_physics`.",
        f"It reranks the original top-k references using dQ/dV only in the high-voltage region `V >= {voltage_min:.2f} V`.",
        "",
        f"- Source setting: `{setting_name}`",
        f"- HUST queries: {len(summary_df)}",
        f"- Top-1 changed from baseline: {int(summary_df['top1_changed'].sum())}/{len(summary_df)}",
        f"- Full-life SOH RMSE improved: {int(summary_df['full_soh_rmse_improved'].sum())}/{len(summary_df)}",
        f"- Baseline mean top-1 RMSE: {summary_df['original_top1_full_soh_rmse'].mean():.6f}",
        f"- Condition mean top-1 RMSE: {summary_df['hv_top1_full_soh_rmse'].mean():.6f}",
        "",
        "Files:",
        "- `query_topk_similarity.csv`",
        "- `query_summary.csv`",
        "- `figures/hust_high_voltage_dqdv_top1_rmse.png`",
        f"- Per-query comparison figures: {len(generated_figure_paths)} files",
    ]
    (condition_setting_dir / "summary.md").write_text("\n".join(condition_lines), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "condition_setting_dir": str(condition_setting_dir),
        "comparison_csv": str(output_dir / "baseline_vs_high_voltage_condition.csv"),
        "num_hust_queries": int(len(summary_df)),
        "top1_changed": int(summary_df["top1_changed"].sum()),
        "full_soh_rmse_improved": int(summary_df["full_soh_rmse_improved"].sum()),
        "original_mean_rmse": float(summary_df["original_top1_full_soh_rmse"].mean()),
        "high_voltage_mean_rmse": float(summary_df["hv_top1_full_soh_rmse"].mean()),
        "generated_per_query_figures": int(len(generated_figure_paths)),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze HUST high-voltage dQ/dV reranking.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("tests/artifacts/subset_rag_cross_cell_retrieval"))
    parser.add_argument("--setting-name", type=str, default="state_metadata_qv_physics")
    parser.add_argument("--voltage-min", type=float, default=3.2)
    args = parser.parse_args(argv)
    print(run_analysis(artifact_dir=args.artifact_dir, setting_name=args.setting_name, voltage_min=args.voltage_min))


if __name__ == "__main__":
    main()
