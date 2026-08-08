from __future__ import annotations

from typing import List, Sequence

import torch
from torch import nn

from config import RunConfig


class Adapter(nn.Module):
    # Initialize a feature adapter.
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropouts: Sequence[float]) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropouts[0])),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropouts[1])),
        )

    # Run the adapter forward pass.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class FusionHead(nn.Module):
    # Initialize the fusion head.
    def __init__(self, input_dim: int, cfg: RunConfig) -> None:
        super().__init__()
        dims = (int(input_dim),) + tuple(int(value) for value in cfg.fusion_hidden_dims)
        modules: List[nn.Module] = []
        for index, dropout in enumerate(cfg.fusion_dropouts):
            modules.extend((
                nn.Linear(dims[index], dims[index + 1]),
                nn.LayerNorm(dims[index + 1]),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            ))
        modules.append(nn.Linear(dims[-1], 1))
        self.network = nn.Sequential(*modules)

    # Run the fusion forward pass.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


class ReservoirRegressor(nn.Module):
    # Initialize the reservoir expert.
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.branch_dims = tuple(int(value) for value in cfg.reservoir_branch_dims)
        adapters: List[nn.Module] = []
        output_dims: List[int] = []
        for index, input_dim in enumerate(self.branch_dims):
            is_extra_random = index == len(self.branch_dims) - 1
            is_ligand = (index % 2 == 1) and not is_extra_random
            if is_ligand:
                adapters.append(Adapter(
                    input_dim,
                    cfg.ligand_adapter_hidden,
                    cfg.ligand_adapter_output,
                    cfg.adapter_dropouts,
                ))
                output_dims.append(cfg.ligand_adapter_output)
            elif is_extra_random:
                adapters.append(Adapter(
                    input_dim,
                    cfg.extra_random_adapter_hidden,
                    cfg.extra_random_adapter_output,
                    cfg.adapter_dropouts,
                ))
                output_dims.append(cfg.extra_random_adapter_output)
            else:
                adapters.append(Adapter(
                    input_dim,
                    cfg.structure_adapter_hidden,
                    cfg.structure_adapter_output,
                    cfg.adapter_dropouts,
                ))
                output_dims.append(cfg.structure_adapter_output)
        self.adapters = nn.ModuleList(adapters)
        self.output_dims = tuple(int(value) for value in output_dims)
        self.branch_logits = nn.Parameter(torch.zeros(len(self.branch_dims)))
        self.fusion = FusionHead(sum(self.output_dims), cfg)

    # Compute branch softmax weights.
    def branch_weights(self) -> torch.Tensor:
        return torch.softmax(self.branch_logits, dim=0)

    # Adapt and weight reservoir branches.
    def adapted_features(self, value: torch.Tensor) -> torch.Tensor:
        weights = self.branch_weights() * float(len(self.branch_dims))
        parts: List[torch.Tensor] = []
        offset = 0
        for index, (input_dim, adapter) in enumerate(zip(self.branch_dims, self.adapters)):
            branch = value[:, offset : offset + input_dim]
            offset += input_dim
            branch = adapter(branch) * weights[index]
            if self.training and self.cfg.branch_dropout_probability > 0.0:
                keep = (
                    torch.rand((branch.shape[0], 1), device=branch.device)
                    >= self.cfg.branch_dropout_probability
                ).to(branch.dtype)
                branch = branch * keep / (1.0 - self.cfg.branch_dropout_probability)
            parts.append(branch)
        return torch.cat(parts, dim=-1)

    # Run the reservoir forward pass.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fusion(self.adapted_features(value))


class JointRegressor(ReservoirRegressor):
    # Initialize the joint expert.
    def __init__(self, cfg: RunConfig, physics_dim: int) -> None:
        super().__init__(cfg)
        self.physics_dim = int(physics_dim)
        self.physics_adapter = Adapter(
            self.physics_dim,
            cfg.physics_adapter_hidden,
            cfg.physics_adapter_output,
            cfg.adapter_dropouts,
        )
        self.fusion = FusionHead(sum(self.output_dims) + cfg.physics_adapter_output, cfg)

    # Fuse reservoir and physics features.
    def forward(self, reservoir_value: torch.Tensor, physics_value: torch.Tensor) -> torch.Tensor:
        reservoir = self.adapted_features(reservoir_value)
        physics = self.physics_adapter(physics_value)
        return self.fusion(torch.cat((reservoir, physics), dim=-1))
