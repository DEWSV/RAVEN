from __future__ import annotations

import csv
import dataclasses
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Sequence, Tuple

import dgl
import torch


ETYPE_LIG = ("ligand", "lig_bond", "ligand")
ETYPE_PRO = ("protein", "pro_contact", "protein")
ETYPE_LP = ("ligand", "lp_contact", "protein")


@dataclass(frozen=True)
class ChunkRecord:
    split: str
    chunk_id: int
    bin_path: Path
    csv_path: Path
    row_count: int


@dataclass
class TensorBatch:
    ligand_x: torch.Tensor
    protein_x: torch.Tensor
    ligand_graph: torch.Tensor
    protein_graph: torch.Tensor
    ligand_counts: torch.Tensor
    protein_counts: torch.Tensor
    ligand_node_keep: torch.Tensor
    protein_node_keep: torch.Tensor

    ligand_edge_src: torch.Tensor
    ligand_edge_dst: torch.Tensor
    ligand_edge_attr: torch.Tensor
    ligand_edge_graph: torch.Tensor
    ligand_edge_counts: torch.Tensor
    ligand_edge_keep: torch.Tensor

    protein_edge_src: torch.Tensor
    protein_edge_dst: torch.Tensor
    protein_edge_attr: torch.Tensor
    protein_edge_graph: torch.Tensor
    protein_edge_counts: torch.Tensor
    protein_edge_keep: torch.Tensor

    pair_ligand: torch.Tensor
    pair_protein: torch.Tensor
    pair_dist: torch.Tensor
    pair_unit: torch.Tensor
    pair_graph: torch.Tensor
    pair_counts: torch.Tensor
    pair_in_cutoff: torch.Tensor
    pair_is_fallback: torch.Tensor
    phys_gates: torch.Tensor
    phys_quality: torch.Tensor
    pair_edge_keep: torch.Tensor

    labels: torch.Tensor
    complex_ids: Tuple[str, ...]
    source: str

    # Count graphs in the batch.
    @property
    def num_graphs(self) -> int:
        return len(self.complex_ids)

    # Pin tensor memory in place.
    def pin_memory_(self) -> "TensorBatch":
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                setattr(self, field.name, value.pin_memory())
        return self

    # Move batch tensors to a device.
    def to(self, device: torch.device, non_blocking: bool = True) -> "TensorBatch":
        values: Dict[str, Any] = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(device, non_blocking=non_blocking)
                if isinstance(value, torch.Tensor)
                else value
            )
        return TensorBatch(**values)


# Read manifest rows.
def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# Discover encoded split chunks.
def discover_pdbbind_chunks(root: str, split: str) -> List[ChunkRecord]:
    directory = Path(root) / f"{split}_chunks"
    pattern = re.compile(rf"pdbbind_{split.lower()}_chunk_(\d{{6}})\.bin")
    records: List[ChunkRecord] = []
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            csv_path = path.with_suffix(".csv")
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)
            records.append(
                ChunkRecord(
                    split=split,
                    chunk_id=int(match.group(1)),
                    bin_path=path,
                    csv_path=csv_path,
                    row_count=row_count,
                )
            )
    records.sort(key=lambda item: item.chunk_id)
    return records


# Resolve complex identifier.
def complex_id_from_row(row: Mapping[str, str], fallback: str) -> str:
    for key in ("complex_id", "pdb_id", "id", "idx"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return fallback


# Resolve regression target.
def target_from_row(row: Mapping[str, str]) -> float:
    for key in ("y", "deltaG_kcal_mol", "label", "target", "affinity"):
        value = str(row.get(key, "")).strip()
        if value:
            return float(value)
    return float("")


# Resolve graph offset.
def graph_offset_from_row(row: Mapping[str, str], fallback: int) -> int:
    value = str(row.get("hetero_graph_offset", "")).strip()
    return int(value) if value else int(fallback)


# Expand graph indices.
def repeat_graph_ids(counts: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(
        torch.arange(counts.numel(), dtype=torch.long), counts.long()
    )


# Collate heterographs into tensors.
def collate_existing_pairs(
    graphs: Sequence[Any],
    source: str,
    labels: Sequence[float],
    complex_ids: Sequence[str],
) -> TensorBatch:
    graph = dgl.batch(list(graphs))
    ligand_counts = graph.batch_num_nodes("ligand").long().cpu()
    protein_counts = graph.batch_num_nodes("protein").long().cpu()
    ligand_edge_counts = graph.batch_num_edges(ETYPE_LIG).long().cpu()
    protein_edge_counts = graph.batch_num_edges(ETYPE_PRO).long().cpu()
    pair_counts = graph.batch_num_edges(ETYPE_LP).long().cpu()

    ligand_graph = repeat_graph_ids(ligand_counts)
    protein_graph = repeat_graph_ids(protein_counts)
    ligand_edge_graph = repeat_graph_ids(ligand_edge_counts)
    protein_edge_graph = repeat_graph_ids(protein_edge_counts)
    pair_graph = repeat_graph_ids(pair_counts)

    ligand_edge_src, ligand_edge_dst = graph.edges(etype=ETYPE_LIG)
    protein_edge_src, protein_edge_dst = graph.edges(etype=ETYPE_PRO)
    pair_ligand, pair_protein = graph.edges(etype=ETYPE_LP)

    ligand_pos = torch.nan_to_num(
        graph.nodes["ligand"].data["pos"].float(),
        nan=0.0,
        posinf=1.0e4,
        neginf=-1.0e4,
    ).clamp(-1.0e4, 1.0e4)
    protein_pos = torch.nan_to_num(
        graph.nodes["protein"].data["pos"].float(),
        nan=0.0,
        posinf=1.0e4,
        neginf=-1.0e4,
    ).clamp(-1.0e4, 1.0e4)

    lp = graph.edges[ETYPE_LP].data
    pair_dist = lp["dist"].float().view(-1)
    pair_vector = protein_pos[pair_protein.long()] - ligand_pos[pair_ligand.long()]
    pair_norm = torch.sqrt(pair_vector.square().sum(dim=-1) + 1.0e-12)
    pair_unit = pair_vector / pair_norm.clamp_min(1.0e-6).unsqueeze(-1)

    return TensorBatch(
        ligand_x=graph.nodes["ligand"].data["x"].float(),
        protein_x=graph.nodes["protein"].data["x"].float(),
        ligand_graph=ligand_graph,
        protein_graph=protein_graph,
        ligand_counts=ligand_counts,
        protein_counts=protein_counts,
        ligand_node_keep=torch.ones(ligand_graph.numel(), dtype=torch.uint8),
        protein_node_keep=torch.ones(protein_graph.numel(), dtype=torch.uint8),

        ligand_edge_src=ligand_edge_src.long(),
        ligand_edge_dst=ligand_edge_dst.long(),
        ligand_edge_attr=graph.edges[ETYPE_LIG].data["edge_attr"].float(),
        ligand_edge_graph=ligand_edge_graph,
        ligand_edge_counts=ligand_edge_counts,
        ligand_edge_keep=torch.ones(ligand_edge_src.numel(), dtype=torch.uint8),

        protein_edge_src=protein_edge_src.long(),
        protein_edge_dst=protein_edge_dst.long(),
        protein_edge_attr=graph.edges[ETYPE_PRO].data["edge_attr"].float(),
        protein_edge_graph=protein_edge_graph,
        protein_edge_counts=protein_edge_counts,
        protein_edge_keep=torch.ones(protein_edge_src.numel(), dtype=torch.uint8),

        pair_ligand=pair_ligand.long(),
        pair_protein=pair_protein.long(),
        pair_dist=pair_dist,
        pair_unit=pair_unit.float(),
        pair_graph=pair_graph,
        pair_counts=pair_counts,
        pair_in_cutoff=lp["in_cutoff"].float().view(-1),
        pair_is_fallback=lp["is_fallback"].float().view(-1),
        phys_gates=lp["phys_gates"].to(torch.uint8),
        phys_quality=lp["phys_quality"].to(torch.float16),
        pair_edge_keep=torch.ones(pair_ligand.numel(), dtype=torch.uint8),

        labels=torch.tensor(labels, dtype=torch.float32),
        complex_ids=tuple(str(value) for value in complex_ids),
        source=source,
    )


# Iterate split batches.
def iter_split_batches(
    records: Sequence[ChunkRecord],
    batch_size: int,
    split: str,
) -> Iterator[TensorBatch]:
    for record in records:
        graphs, _ = dgl.load_graphs(str(record.bin_path))
        rows = read_csv_rows(record.csv_path)
        offsets = [graph_offset_from_row(row, index) for index, row in enumerate(rows)]
        for start in range(0, len(rows), int(batch_size)):
            indices = list(range(start, min(start + int(batch_size), len(rows))))
            labels = [target_from_row(rows[index]) for index in indices]
            ids = [
                complex_id_from_row(rows[index], f"{split}:{record.chunk_id}:{index}")
                for index in indices
            ]
            yield collate_existing_pairs(
                [graphs[offsets[index]] for index in indices],
                source=f"PDBbind:{split}:{record.chunk_id}",
                labels=labels,
                complex_ids=ids,
            )
        del graphs
