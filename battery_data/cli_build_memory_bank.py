from __future__ import annotations

import argparse
import json

import yaml

from .build_memory_bank import build_battery_memory_bank


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_encoder_from_config(cfg: dict):
    from retrieval.encoder_chronos2 import Chronos2RetrieverEncoder

    enc_cfg = cfg["encoder"]
    return Chronos2RetrieverEncoder(
        model_path=enc_cfg.get("model_path", "autogluon/chronos-2"),
        pooling=enc_cfg.get("pooling", "mean"),
        device=enc_cfg.get("device", "auto"),
        context_length=enc_cfg.get("context_length"),
        batch_size=enc_cfg.get("batch_size", 256),
        torch_dtype=enc_cfg.get("torch_dtype", "float32"),
    )


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
