from __future__ import annotations

import argparse
import json

from battery_data.build_case_bank import load_config
from experiments.validate_preprocessing_features import validate_preprocessing_features
from experiments.validate_retrieval_quality import validate_retrieval_quality


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run smoke preprocessing/retrieval experiments")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = {
        "feature_validation": validate_preprocessing_features(cfg),
        "retrieval_validation": validate_retrieval_quality(cfg, num_queries=10),
    }
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
