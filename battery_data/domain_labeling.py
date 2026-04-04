from __future__ import annotations

from typing import Any, Dict, Iterable


DEFAULT_RULES = {
    "fastcharge_policies": {"fastcharge"},
    "multistage_policies": {"multistage"},
    "irregular_policies": {"irregular", "random_walk", "satellite"},
    "fallback_chemistry": "Unknown",
    "fallback_policy": "regular",
    "fallback_full": "full",
}


def _canonicalize_token(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text if text else fallback


def build_domain_label(
    metadata: Dict[str, Any],
    rules: Dict[str, Any] | None = None,
) -> str:
    cfg = dict(DEFAULT_RULES)
    if rules:
        cfg.update(rules)

    chemistry = _canonicalize_token(metadata.get("chemistry_family"), cfg["fallback_chemistry"])
    policy = _canonicalize_token(metadata.get("discharge_policy_family"), cfg["fallback_policy"])
    full_or_partial = _canonicalize_token(metadata.get("full_or_partial"), cfg["fallback_full"])

    fastcharge_policies = set(cfg["fastcharge_policies"])
    multistage_policies = set(cfg["multistage_policies"])
    irregular_policies = set(cfg["irregular_policies"])

    if policy in fastcharge_policies:
        return f"{chemistry}_fastcharge"
    if policy in multistage_policies:
        return f"{chemistry}_multistage"
    if policy in irregular_policies:
        return f"{chemistry}_irregular_{full_or_partial}"
    if full_or_partial == "full":
        return f"{chemistry}_regular"
    return f"{chemistry}_regular_{full_or_partial}"


def build_cell_level_domain_label(
    row: Dict[str, Any],
    rules: Dict[str, Any] | None = None,
) -> str:
    return build_domain_label(row, rules)
