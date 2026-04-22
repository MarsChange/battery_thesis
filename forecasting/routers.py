from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class GroupedAdditiveRouter(nn.Module):
    def __init__(
        self,
        group_dims: Dict[str, int],
        num_outputs: int,
        top_k: int | None = None,
    ):
        super().__init__()
        self.group_dims = {name: int(dim) for name, dim in group_dims.items()}
        self.num_outputs = int(num_outputs)
        self.top_k = top_k
        self.group_layers = nn.ModuleDict(
            {
                name: nn.Linear(dim, self.num_outputs) if dim > 0 else nn.Identity()
                for name, dim in self.group_dims.items()
            }
        )
        self.bias = nn.Parameter(torch.zeros(self.num_outputs))

    def forward(self, group_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        contributions = {}
        for name, dim in self.group_dims.items():
            x = group_inputs[name]
            if dim <= 0:
                contributions[name] = torch.zeros(x.shape[0], self.num_outputs, device=x.device, dtype=x.dtype)
            else:
                contributions[name] = self.group_layers[name](x)
        logits = self.bias.unsqueeze(0)
        for value in contributions.values():
            logits = logits + value

        if self.top_k is not None and self.top_k < self.num_outputs:
            topk_values, topk_indices = torch.topk(logits, k=self.top_k, dim=-1)
            topk_mask = torch.zeros_like(logits)
            topk_mask.scatter_(1, topk_indices, 1.0)
            masked_logits = logits.masked_fill(topk_mask == 0, float("-inf"))
            weights = torch.softmax(masked_logits, dim=-1)
        else:
            topk_mask = torch.ones_like(logits)
            weights = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "weights": weights,
            "topk_mask": topk_mask,
            "contributions": contributions,
        }


class PhysicalDegradationRouter(GroupedAdditiveRouter):
    REQUIRED_GROUPS = [
        "state",
        "curve",
        "partial_charge",
        "relaxation",
        "operation",
        "chemistry",
        "retrieval",
        "availability",
    ]

    def __init__(self, group_dims: Dict[str, int], num_experts: int, top_k_experts: int = 2):
        super().__init__(group_dims=group_dims, num_outputs=num_experts, top_k=top_k_experts)


class BranchFusionRouter(GroupedAdditiveRouter):
    def __init__(self, group_dims: Dict[str, int], num_branches: int = 4):
        super().__init__(group_dims=group_dims, num_outputs=num_branches, top_k=None)
