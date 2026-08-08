from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from config import RunConfig
from data_pipeline import TensorBatch


# Compute masked graph means.
def scatter_masked_mean(
    values: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = values.new_zeros((int(size), values.shape[-1]))
    counts = values.new_zeros((int(size), 1))
    if values.numel():
        scalar = weight.float().view(-1, 1).to(values.dtype)
        output.index_add_(0, index.long(), values * scalar)
        counts.index_add_(0, index.long(), scalar)
    return output / counts.clamp_min(1.0)


# Compute scaled masked sums.
def scatter_masked_sum_sqrt(
    values: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    size: int,
) -> torch.Tensor:
    output = values.new_zeros((int(size), values.shape[-1]))
    counts = values.new_zeros((int(size), 1))
    if values.numel():
        scalar = weight.float().view(-1, 1).to(values.dtype)
        output.index_add_(0, index.long(), values * scalar)
        counts.index_add_(0, index.long(), scalar)
    return output / torch.sqrt(counts.clamp_min(1.0))


# Compute masked graph maxima.
def scatter_masked_max(
    values: torch.Tensor,
    index: torch.Tensor,
    weight: torch.Tensor,
    size: int,
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((int(size), values.shape[-1]))
    keep = weight.bool()
    masked = values.masked_fill(~keep.unsqueeze(-1), -torch.inf)
    output = values.new_full((int(size), values.shape[-1]), -torch.inf)
    expanded_index = index.long().view(-1, 1).expand_as(values)
    output.scatter_reduce_(0, expanded_index, masked, reduce="amax", include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


class MLP(nn.Module):
    # Initialize the MLP.
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> None:
        super().__init__()
        modules = []
        current = int(input_dim)
        for _ in range(int(layers) - 1):
            modules.extend((
                nn.Linear(current, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ))
            current = int(hidden_dim)
        modules.append(nn.Linear(current, output_dim))
        self.network = nn.Sequential(*modules)

    # Run the MLP forward pass.
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class RadialBasis(nn.Module):
    # Initialize radial basis features.
    def __init__(self, bins: int, maximum: float = 8.0) -> None:
        super().__init__()
        centers = torch.linspace(0.0, float(maximum), int(bins))
        spacing = float(maximum) / max(int(bins) - 1, 1)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / max(spacing * spacing, 1.0e-6)

    # Expand distances with radial bases.
    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        delta = distance.float().unsqueeze(-1) - self.centers.float()
        return torch.exp(-self.gamma * delta.square())


class HeteroMessageLayer(nn.Module):
    # Initialize one message layer.
    def __init__(self, hidden_dim: int, edge_hidden_dim: int) -> None:
        super().__init__()
        message_input = hidden_dim * 2 + edge_hidden_dim
        self.ligand_intra = MLP(message_input, hidden_dim, hidden_dim)
        self.protein_intra = MLP(message_input, hidden_dim, hidden_dim)
        self.ligand_to_protein = MLP(message_input, hidden_dim, hidden_dim)
        self.protein_to_ligand = MLP(message_input, hidden_dim, hidden_dim)
        self.ligand_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.protein_update = nn.GRUCell(hidden_dim, hidden_dim)
        self.ligand_norm = nn.LayerNorm(hidden_dim)
        self.protein_norm = nn.LayerNorm(hidden_dim)

    # Run heterogeneous message passing.
    def forward(
        self,
        ligand: torch.Tensor,
        protein: torch.Tensor,
        ligand_edge: torch.Tensor,
        protein_edge: torch.Tensor,
        pair_edge: torch.Tensor,
        batch: TensorBatch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ligand_message = self.ligand_intra(torch.cat((
            ligand[batch.ligand_edge_src],
            ligand[batch.ligand_edge_dst],
            ligand_edge,
        ), dim=-1))
        ligand_intra = scatter_masked_sum_sqrt(
            ligand_message,
            batch.ligand_edge_dst,
            batch.ligand_edge_keep,
            ligand.shape[0],
        )

        protein_message = self.protein_intra(torch.cat((
            protein[batch.protein_edge_src],
            protein[batch.protein_edge_dst],
            protein_edge,
        ), dim=-1))
        protein_intra = scatter_masked_sum_sqrt(
            protein_message,
            batch.protein_edge_dst,
            batch.protein_edge_keep,
            protein.shape[0],
        )

        to_protein = self.ligand_to_protein(torch.cat((
            ligand[batch.pair_ligand],
            protein[batch.pair_protein],
            pair_edge,
        ), dim=-1))
        protein_cross = scatter_masked_sum_sqrt(
            to_protein,
            batch.pair_protein,
            batch.pair_edge_keep,
            protein.shape[0],
        )

        to_ligand = self.protein_to_ligand(torch.cat((
            protein[batch.pair_protein],
            ligand[batch.pair_ligand],
            pair_edge,
        ), dim=-1))
        ligand_cross = scatter_masked_sum_sqrt(
            to_ligand,
            batch.pair_ligand,
            batch.pair_edge_keep,
            ligand.shape[0],
        )

        ligand_new = self.ligand_norm(
            self.ligand_update(ligand_intra + ligand_cross, ligand)
        )
        protein_new = self.protein_norm(
            self.protein_update(protein_intra + protein_cross, protein)
        )
        ligand_new = ligand_new * batch.ligand_node_keep.float().unsqueeze(-1)
        protein_new = protein_new * batch.protein_node_keep.float().unsqueeze(-1)
        return ligand_new, protein_new


class RandomComplexEncoder(nn.Module):
    # Initialize the frozen random encoder.
    def __init__(self, cfg: RunConfig) -> None:
        super().__init__()
        hidden = int(cfg.hidden_dim)
        edge_hidden = int(cfg.edge_hidden_dim)
        self.ligand_node_encoder = MLP(cfg.expected_node_dim, hidden, hidden)
        self.protein_node_encoder = MLP(cfg.expected_node_dim, hidden, hidden)
        self.ligand_edge_encoder = MLP(10, edge_hidden, edge_hidden)
        self.protein_edge_encoder = MLP(10, edge_hidden, edge_hidden)
        self.rbf = RadialBasis(cfg.rbf_bins, maximum=8.0)
        pair_input = cfg.rbf_bins + 3 + 10 + 10 + 1 + 1
        self.pair_edge_encoder = MLP(pair_input, edge_hidden, edge_hidden)
        self.layers = nn.ModuleList(
            HeteroMessageLayer(hidden, edge_hidden)
            for _ in range(cfg.message_layers)
        )
        self.pair_representation = MLP(hidden * 2 + edge_hidden, hidden, hidden)
        self.representation_head = MLP(
            hidden * 6,
            cfg.representation_dim,
            cfg.representation_dim,
            layers=2,
        )

    # Build cross-pair inputs.
    def pair_input(self, batch: TensorBatch) -> torch.Tensor:
        return torch.cat((
            self.rbf(batch.pair_dist),
            batch.pair_unit.float(),
            batch.phys_gates.float(),
            batch.phys_quality.float(),
            batch.pair_in_cutoff.float().unsqueeze(-1),
            batch.pair_is_fallback.float().unsqueeze(-1),
        ), dim=-1)

    # Encode a complex batch.
    def forward(self, batch: TensorBatch) -> Dict[str, torch.Tensor]:
        ligand = self.ligand_node_encoder(batch.ligand_x.float())
        protein = self.protein_node_encoder(batch.protein_x.float())
        ligand_edge = self.ligand_edge_encoder(batch.ligand_edge_attr.float())
        protein_edge = self.protein_edge_encoder(batch.protein_edge_attr.float())
        pair_edge = self.pair_edge_encoder(self.pair_input(batch))

        for layer in self.layers:
            ligand, protein = layer(
                ligand,
                protein,
                ligand_edge,
                protein_edge,
                pair_edge,
                batch,
            )

        pair_repr = self.pair_representation(torch.cat((
            ligand[batch.pair_ligand],
            protein[batch.pair_protein],
            pair_edge,
        ), dim=-1))

        ligand_mean = scatter_masked_mean(
            ligand, batch.ligand_graph, batch.ligand_node_keep, batch.num_graphs
        )
        ligand_max = scatter_masked_max(
            ligand, batch.ligand_graph, batch.ligand_node_keep, batch.num_graphs
        )
        protein_mean = scatter_masked_mean(
            protein, batch.protein_graph, batch.protein_node_keep, batch.num_graphs
        )
        protein_max = scatter_masked_max(
            protein, batch.protein_graph, batch.protein_node_keep, batch.num_graphs
        )
        interaction_mean = scatter_masked_mean(
            pair_repr, batch.pair_graph, batch.pair_edge_keep, batch.num_graphs
        )
        interaction_max = scatter_masked_max(
            pair_repr, batch.pair_graph, batch.pair_edge_keep, batch.num_graphs
        )

        protein_pool = torch.cat((protein_mean, protein_max), dim=-1)
        ligand_pool = torch.cat((ligand_mean, ligand_max), dim=-1)
        interaction_pool = torch.cat((interaction_mean, interaction_max), dim=-1)
        representation = self.representation_head(torch.cat((
            protein_pool,
            ligand_pool,
            interaction_pool,
        ), dim=-1))

        return {
            "protein_pool": protein_pool,
            "ligand_pool": ligand_pool,
            "interaction_pool": interaction_pool,
            "representation": representation,
        }
