from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Tuple
import warnings
import torch
from torch import Tensor, nn

from Scripts.prerequisite_dgcn import (
    PrerequisiteDGCN,
    PrerequisiteGraph,
    build_proximities,
    load_prerequisite_graph,
)


@dataclass(frozen=True)
class SimilarityGraph:
    """In-memory representation of an undirected similarity graph."""

    concepts: List[str]
    adjacency: Tensor

    @property
    def device(self) -> torch.device:
        return self.adjacency.device

    @property
    def size(self) -> int:
        return self.adjacency.size(0)

def _expand_prerequisite_graph(
    graph: PrerequisiteGraph, target_size: int
) -> PrerequisiteGraph:
    """Pad a prerequisite graph so it matches ``target_size`` nodes."""

    if graph.size > target_size:
        raise ValueError(
            "Prerequisite graph has more nodes than skill_num: "
            f"{graph.size} > {target_size}"
        )
    if graph.size == target_size:
        return graph

    padding = target_size - graph.size
    warnings.warn(
        "Prerequisite graph size does not match skill_num; "
        f"padding with {padding} isolated nodes.",
        RuntimeWarning,
    )
    padded_adjacency = graph.adjacency.new_zeros((target_size, target_size))
    padded_adjacency[: graph.size, : graph.size] = graph.adjacency
    padded_concepts = graph.concepts + [
        f"__missing_prereq_{idx}" for idx in range(padding)
    ]
    return PrerequisiteGraph(padded_concepts, padded_adjacency)


def _expand_similarity_graph(
    graph: SimilarityGraph, target_size: int
) -> SimilarityGraph:
    """Pad a similarity graph so it matches ``target_size`` nodes."""

    if graph.size > target_size:
        raise ValueError(
            "Similarity graph has more nodes than skill_num: "
            f"{graph.size} > {target_size}"
        )
    if graph.size == target_size:
        return graph

    padding = target_size - graph.size
    warnings.warn(
        "Similarity graph size does not match skill_num; "
        f"padding with {padding} isolated nodes.",
        RuntimeWarning,
    )
    padded_adjacency = graph.adjacency.new_zeros((target_size, target_size))
    padded_adjacency[: graph.size, : graph.size] = graph.adjacency
    padded_concepts = graph.concepts + [
        f"__missing_similarity_{idx}" for idx in range(padding)
    ]
    return SimilarityGraph(padded_concepts, padded_adjacency)


def load_similarity_graph(
    json_path: Path | str,
    *,
    confidence_threshold: float = 0.0,
    normalise_confidence: bool = True,
) -> SimilarityGraph:
    """Load a similarity graph from ``json_path``.

    The JSON file is expected to contain a ``"similarity_edges"`` array
    where each entry provides ``u`` (first concept), ``v`` (second
    concept), ``relation`` (kept for completeness), ``confidence`` and an
    optional ``weight``.  Edges are treated as undirected and stored in a
    dense adjacency matrix.
    """

    graph_path = Path(json_path)
    with graph_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    edges: Iterable[Mapping[str, object]] = data.get("similarity_edges", [])

    concepts: MutableMapping[str, int] = {}
    filtered_edges: List[Tuple[str, str, float]] = []
    max_confidence = 0.0

    for edge in edges:
        head = str(edge["u"])
        tail = str(edge["v"])
        confidence = float(edge.get("confidence", 1.0))
        if confidence < confidence_threshold:
            continue
        weight = float(edge.get("weight", confidence))
        filtered_edges.append((head, tail, weight))
        max_confidence = max(max_confidence, weight)
        for concept in (head, tail):
            if concept not in concepts:
                concepts[concept] = len(concepts)

    if not concepts:
        raise ValueError(f"No similarity edges found in {graph_path!s}.")

    scale = 1.0 / max_confidence if normalise_confidence and max_confidence else 1.0
    num_nodes = len(concepts)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for head, tail, weight in filtered_edges:
        i = concepts[head]
        j = concepts[tail]
        scaled_weight = weight * scale
        adjacency[i, j] += scaled_weight
        adjacency[j, i] += scaled_weight

    ordered_concepts: List[str] = [""] * num_nodes
    for name, index in concepts.items():
        ordered_concepts[index] = name

    return SimilarityGraph(ordered_concepts, adjacency)


def _symmetric_normalise(matrix: Tensor) -> Tensor:
    degree = matrix.sum(dim=1)
    inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    inv_sqrt[mask] = degree[mask].pow(-0.5)
    return inv_sqrt.unsqueeze(1) * matrix * inv_sqrt.unsqueeze(0)


class LightGCNEncoder(nn.Module):
    """LightGCN propagation on an undirected similarity graph."""

    def __init__(
        self,
        adjacency: Tensor,
        num_layers: int,
        *,
        add_self_loops: bool = True,
    ) -> None:
        super().__init__()
        if adjacency.dim() != 2 or adjacency.size(0) != adjacency.size(1):
            raise ValueError("adjacency must be a square matrix")
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")

        if add_self_loops:
            identity = torch.eye(adjacency.size(0), dtype=adjacency.dtype, device=adjacency.device)
            adjacency = adjacency + identity

        self.num_layers = num_layers
        self.register_buffer("normalised_adjacency", _symmetric_normalise(adjacency))

    def forward(self, features: Tensor) -> Tensor:
        if features.dim() != 2:
            raise ValueError("features must have shape (num_nodes, feature_dim)")
        if features.size(0) != self.normalised_adjacency.size(0):
            raise ValueError("Mismatch between feature nodes and adjacency size")

        propagation = features
        accumulated = features
        normalised = self.normalised_adjacency.to(features.device, dtype=features.dtype)

        for _ in range(self.num_layers):
            propagation = normalised @ propagation
            accumulated = accumulated + propagation

        return accumulated / (self.num_layers + 1)


class GraphFusionEncoder(nn.Module):
    """Fuse DGCN and LightGCN representations into skill embeddings."""

    def __init__(
        self,
        skill_num: int,
        embedding_dim: int,
        *,
        prerequisite_graph: PrerequisiteGraph,
        similarity_graph: SimilarityGraph,
        dgcn_layers: int = 2,
        lightgcn_layers: int = 2,
        fusion_weight: float = 0,
        dgcn_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        prerequisite_graph = _expand_prerequisite_graph(prerequisite_graph, skill_num)
        similarity_graph = _expand_similarity_graph(similarity_graph, skill_num)
        if not (0.0 <= fusion_weight <= 1.0):
            raise ValueError("fusion_weight must be between 0 and 1")

        self.embedding = nn.Embedding(skill_num, embedding_dim)

        proximities = build_proximities(prerequisite_graph)
        for idx, matrix in enumerate(proximities):
            self.register_buffer(f"_dgcn_proximity_{idx}", matrix)

        self.dgcn = PrerequisiteDGCN(
            num_features=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=dgcn_layers,
            dropout=dgcn_dropout,
        )
        self.lightgcn = LightGCNEncoder(similarity_graph.adjacency, lightgcn_layers)
        self.fusion_weight = fusion_weight

    def _dgcn_proximities(self) -> List[Tensor]:
        return [
            getattr(self, f"_dgcn_proximity_{idx}") for idx in range(3)
        ]

    def forward(self) -> Tensor:
        base_features = self.embedding.weight
        proximities = [
            matrix.to(base_features.device, dtype=base_features.dtype)
            for matrix in self._dgcn_proximities()
        ]
        dgcn_embeddings = self.dgcn(base_features, proximities)
        lightgcn_embeddings = self.lightgcn(base_features)
        return (
            self.fusion_weight * dgcn_embeddings
            + (1.0 - self.fusion_weight) * lightgcn_embeddings
        )


__all__ = [
    "GraphFusionEncoder",
    "LightGCNEncoder",
    "SimilarityGraph",
    "load_prerequisite_graph",
    "load_similarity_graph",
]
