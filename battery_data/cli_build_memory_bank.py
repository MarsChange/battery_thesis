from __future__ import annotations

import argparse
import json

import yaml

from .build_memory_bank import build_battery_memory_bank


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_encoder_from_config(cfg: dict):
    from retrieval.statistical_encoder import StatisticalWindowEncoder

    enc_cfg = cfg.get("encoder", {})
    return StatisticalWindowEncoder(eps=float(enc_cfg.get("eps", 1e-8)))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build battery SOH memory bank")
    parser.add_argument("--config", type=str, required=True, help="Path to battery YAML config")
    parser.add_argument("--skip-search-validation", action="store_true", help="Skip target-query search demo")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    encoder = build_encoder_from_config(cfg)

    result = build_battery_memory_bank(
        cfg=cfg,
        encoder=encoder,
        run_search_validation=not args.skip_search_validation,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
