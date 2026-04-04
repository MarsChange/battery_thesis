from .hust import load_hust_cells
from .mit import load_mit_cells
from .tju import load_tju_cells
from .xjtu import load_xjtu_cells

ADAPTER_REGISTRY = {
    "xjtu": load_xjtu_cells,
    "tju": load_tju_cells,
    "hust": load_hust_cells,
    "mit": load_mit_cells,
}

__all__ = [
    "ADAPTER_REGISTRY",
    "load_xjtu_cells",
    "load_tju_cells",
    "load_hust_cells",
    "load_mit_cells",
]
