"""
battery_data — Battery SOH memory-bank preprocessing on top of retrieval-only TS-RAG.
"""

__all__ = [
    "build_battery_memory_bank",
    "run_validation_search",
    "run_retrieval_visual_evaluation",
]


def __getattr__(name):
    if name in {"build_battery_memory_bank", "run_validation_search"}:
        from .build_memory_bank import build_battery_memory_bank, run_validation_search

        return {
            "build_battery_memory_bank": build_battery_memory_bank,
            "run_validation_search": run_validation_search,
        }[name]
    if name == "run_retrieval_visual_evaluation":
        from .retrieval_eval import run_retrieval_visual_evaluation

        return run_retrieval_visual_evaluation
    raise AttributeError(name)
