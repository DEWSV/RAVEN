from __future__ import annotations

from typing import List

import torch

from data_pipeline import TensorBatch


# Compute grouped sums.
def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    output = values.new_zeros((int(size), values.shape[-1]))
    if values.numel():
        output.index_add_(0, index.long(), values)
    return output


# Compute grouped means.
def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    output = scatter_sum(values, index, size)
    counts = scatter_sum(values.new_ones((values.shape[0], 1)), index, size)
    return output / counts.clamp_min(1.0)


# Compute grouped standard deviations.
def scatter_std(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    mean = scatter_mean(values, index, size)
    centered = values - mean.index_select(0, index.long())
    variance = scatter_mean(centered.square(), index, size)
    return torch.sqrt(variance + 1.0e-8)


# Compute grouped extrema.
def scatter_extreme(
    values: torch.Tensor,
    index: torch.Tensor,
    size: int,
    mode: str,
) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    if values.numel() == 0:
        return values.new_zeros((int(size), values.shape[-1]))
    initial = -torch.inf if mode == "amax" else torch.inf
    output = values.new_full((int(size), values.shape[-1]), initial)
    expanded = index.long().view(-1, 1).expand_as(values)
    output.scatter_reduce_(0, expanded, values, reduce=mode, include_self=True)
    return torch.where(torch.isfinite(output), output, torch.zeros_like(output))


# Compute scaled grouped sums.
def scatter_sqrt_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if values.ndim == 1:
        values = values.unsqueeze(-1)
    output = scatter_sum(values, index, size)
    counts = scatter_sum(values.new_ones((values.shape[0], 1)), index, size)
    return output / torch.sqrt(counts.clamp_min(1.0))


# Compute basic grouped statistics.
def basic_stats(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    values = values.float()
    return torch.cat((
        scatter_mean(values, index, size),
        scatter_std(values, index, size),
        scatter_extreme(values, index, size, "amin"),
        scatter_extreme(values, index, size, "amax"),
        scatter_sqrt_sum(values, index, size),
    ), dim=-1)


# Compute weighted distance statistics.
def weighted_stats(
    value: torch.Tensor,
    weight: torch.Tensor,
    index: torch.Tensor,
    size: int,
) -> torch.Tensor:
    value = value.float().view(-1, 1)
    weight = weight.float().view(-1, 1)
    weighted_count = scatter_sum(weight, index, size)
    weighted_sum = scatter_sum(value * weight, index, size)
    mean = weighted_sum / weighted_count.clamp_min(1.0)
    centered = (value - mean.index_select(0, index.long())).square() * weight
    std = torch.sqrt(scatter_sum(centered, index, size) / weighted_count.clamp_min(1.0) + 1.0e-8)

    masked_min = value.masked_fill(weight <= 0.0, torch.inf)
    masked_max = value.masked_fill(weight <= 0.0, -torch.inf)
    minimum = scatter_extreme(masked_min, index, size, "amin")
    maximum = scatter_extreme(masked_max, index, size, "amax")
    return torch.cat((weighted_count, mean, std, minimum, maximum), dim=-1)


# Build distance histograms.
def distance_histogram(
    distance: torch.Tensor,
    graph_index: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    edges = distance.new_tensor((0.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.01))
    bucket = torch.bucketize(distance.float(), edges[1:-1], right=False)
    channels = edges.numel() - 1
    one_hot = torch.nn.functional.one_hot(bucket.long(), num_classes=int(channels)).float()
    counts = scatter_sum(one_hot, graph_index, graph_count)
    total = counts.sum(dim=-1, keepdim=True)
    ratio = counts / total.clamp_min(1.0)
    return torch.cat((torch.log1p(counts), ratio), dim=-1)


# Build the physics fingerprint.
def build_physics_fingerprint(batch: TensorBatch) -> torch.Tensor:
    graph_count = batch.num_graphs
    features: List[torch.Tensor] = []

    ligand_counts = batch.ligand_counts.float().view(-1, 1)
    protein_counts = batch.protein_counts.float().view(-1, 1)
    ligand_edge_counts = batch.ligand_edge_counts.float().view(-1, 1)
    protein_edge_counts = batch.protein_edge_counts.float().view(-1, 1)
    pair_counts = batch.pair_counts.float().view(-1, 1)

    cutoff_count = scatter_sum(batch.pair_in_cutoff.float(), batch.pair_graph, graph_count)
    fallback_count = scatter_sum(batch.pair_is_fallback.float(), batch.pair_graph, graph_count)
    global_counts = torch.cat((
        torch.log1p(ligand_counts),
        torch.log1p(protein_counts),
        torch.log1p(ligand_edge_counts),
        torch.log1p(protein_edge_counts),
        torch.log1p(pair_counts),
        ligand_edge_counts / ligand_counts.clamp_min(1.0),
        protein_edge_counts / protein_counts.clamp_min(1.0),
        pair_counts / ligand_counts.clamp_min(1.0),
        pair_counts / protein_counts.clamp_min(1.0),
        pair_counts / (ligand_counts * protein_counts).clamp_min(1.0),
        cutoff_count / pair_counts.clamp_min(1.0),
        fallback_count / pair_counts.clamp_min(1.0),
    ), dim=-1)
    features.append(global_counts)

    features.append(basic_stats(batch.ligand_x, batch.ligand_graph, graph_count))
    features.append(basic_stats(batch.protein_x, batch.protein_graph, graph_count))
    features.append(basic_stats(batch.ligand_edge_attr, batch.ligand_edge_graph, graph_count))
    features.append(basic_stats(batch.protein_edge_attr, batch.protein_edge_graph, graph_count))

    distance = batch.pair_dist.float().clamp_min(1.0e-3)
    features.append(basic_stats(distance, batch.pair_graph, graph_count))
    features.append(distance_histogram(distance, batch.pair_graph, graph_count))

    pair_unit = batch.pair_unit.float()
    features.append(torch.cat((
        scatter_mean(pair_unit, batch.pair_graph, graph_count),
        scatter_std(pair_unit, batch.pair_graph, graph_count),
        scatter_mean(pair_unit.abs(), batch.pair_graph, graph_count),
    ), dim=-1))

    rbf_centers = distance.new_tensor((2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0))
    rbf = torch.exp(-((distance.unsqueeze(-1) - rbf_centers) / 0.45).square())
    features.append(torch.cat((
        scatter_mean(rbf, batch.pair_graph, graph_count),
        scatter_sqrt_sum(rbf, batch.pair_graph, graph_count),
    ), dim=-1))

    gates = batch.phys_gates.float()
    quality = batch.phys_quality.float()
    gate_count = scatter_sum(gates, batch.pair_graph, graph_count)
    gate_ratio = gate_count / pair_counts.clamp_min(1.0)
    gate_sqrt = gate_count / torch.sqrt(pair_counts.clamp_min(1.0))
    features.append(torch.cat((gate_count, gate_ratio, gate_sqrt), dim=-1))

    distance_weights = torch.stack((
        torch.exp(-distance),
        torch.exp(-distance / 2.0),
        1.0 / distance,
        1.0 / distance.square(),
    ), dim=-1)
    weighted_gate = gates.unsqueeze(-1) * distance_weights.unsqueeze(1)
    weighted_gate = weighted_gate.flatten(1)
    features.append(scatter_sum(weighted_gate, batch.pair_graph, graph_count))

    channel_distance_parts: List[torch.Tensor] = []
    for channel in range(gates.shape[1]):
        channel_distance_parts.append(
            weighted_stats(distance, gates[:, channel], batch.pair_graph, graph_count)
        )
    features.append(torch.cat(channel_distance_parts, dim=-1))

    features.append(basic_stats(quality, batch.pair_graph, graph_count))
    active_quality = quality * gates
    features.append(torch.cat((
        scatter_mean(active_quality, batch.pair_graph, graph_count),
        scatter_sqrt_sum(active_quality, batch.pair_graph, graph_count),
        scatter_extreme(active_quality, batch.pair_graph, graph_count, "amax"),
    ), dim=-1))

    active_channels = (gate_count > 0.0).float().sum(dim=-1, keepdim=True)
    total_gate_events = gate_count.sum(dim=-1, keepdim=True)
    features.append(torch.cat((active_channels, torch.log1p(total_gate_events)), dim=-1))

    return torch.cat(features, dim=-1).float()
