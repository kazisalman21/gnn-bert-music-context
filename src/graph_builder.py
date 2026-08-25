"""
Graph construction module for music segment graphs.

Builds PyTorch Geometric Data objects from audio segment features:
- Nodes = fixed-duration audio time segments
- Node features = MFCC/chroma-derived segment vectors
- Edges = temporal adjacency + acoustic similarity above threshold tau

This is NOT a chord-transition graph. No chord detection is performed.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from typing import Optional
import json


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        features: (num_nodes, feat_dim) array
    
    Returns:
        (num_nodes, num_nodes) similarity matrix
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = features / norms
    sim = normalized @ normalized.T
    return sim


def build_segment_graph(segment_features: np.ndarray,
                        tau: float = 0.5,
                        temporal_edges: bool = True,
                        self_loops: bool = True,
                        directed: bool = False) -> Data:
    """Build a segment graph from audio segment features.
    
    Args:
        segment_features: (num_segments, feat_dim) array
        tau: cosine similarity threshold for similarity edges
        temporal_edges: add edges between consecutive segments
        self_loops: add self-loop edges
        directed: if False, add edges in both directions
    
    Returns:
        PyTorch Geometric Data object with:
        - x: (num_nodes, feat_dim) node features
        - edge_index: (2, num_edges) edge indices
        - edge_attr: (num_edges, 1) edge weights (similarity or 1.0 for temporal)
    """
    num_nodes = segment_features.shape[0]
    
    if num_nodes == 0:
        raise ValueError("Cannot build graph with 0 nodes")
    
    edges = []
    edge_weights = []
    
    # Temporal adjacency edges (consecutive segments)
    if temporal_edges:
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
            edge_weights.append(1.0)
            if not directed:
                edges.append((i + 1, i))
                edge_weights.append(1.0)
    
    # Similarity edges (cosine similarity > tau)
    if num_nodes > 1:
        sim_matrix = cosine_similarity_matrix(segment_features)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if sim_matrix[i, j] > tau:
                    # Skip if already a temporal edge
                    if temporal_edges and abs(i - j) == 1:
                        continue
                    edges.append((i, j))
                    edge_weights.append(float(sim_matrix[i, j]))
                    if not directed:
                        edges.append((j, i))
                        edge_weights.append(float(sim_matrix[i, j]))
    
    # Self-loops
    if self_loops:
        for i in range(num_nodes):
            edges.append((i, i))
            edge_weights.append(1.0)
    
    # Convert to tensors
    x = torch.tensor(segment_features, dtype=torch.float32)
    
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
    else:
        # Isolated nodes (only self-loops if enabled)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


def validate_graph(data: Data) -> dict:
    """Validate a PyTorch Geometric graph object.
    
    Returns dict with validation results.
    """
    results = {
        'valid': True,
        'num_nodes': data.x.shape[0],
        'num_edges': data.edge_index.shape[1],
        'feature_dim': data.x.shape[1],
        'issues': []
    }
    
    # Check non-empty
    if data.x.shape[0] == 0:
        results['valid'] = False
        results['issues'].append("Zero nodes")
    
    # Check finite values
    if not torch.isfinite(data.x).all():
        results['valid'] = False
        results['issues'].append("Non-finite node features")
    
    if data.edge_attr is not None and not torch.isfinite(data.edge_attr).all():
        results['valid'] = False
        results['issues'].append("Non-finite edge attributes")
    
    # Check edge index bounds
    if data.edge_index.shape[1] > 0:
        max_idx = data.edge_index.max().item()
        if max_idx >= data.x.shape[0]:
            results['valid'] = False
            results['issues'].append(f"Edge index {max_idx} >= num_nodes {data.x.shape[0]}")
        if data.edge_index.min().item() < 0:
            results['valid'] = False
            results['issues'].append("Negative edge index")
    
    return results


def save_graph(data: Data, path: str, metadata: Optional[dict] = None):
    """Save graph as .pt file with optional metadata."""
    save_dict = {
        'x': data.x,
        'edge_index': data.edge_index,
        'edge_attr': data.edge_attr,
    }
    # Include any extra attributes (labels, etc.)
    if hasattr(data, 'y') and data.y is not None:
        save_dict['y'] = data.y
    if metadata:
        save_dict['metadata'] = metadata
    torch.save(save_dict, path)


def load_graph(path: str) -> Data:
    """Load graph from .pt file."""
    save_dict = torch.load(path, weights_only=False)
    data = Data(
        x=save_dict['x'],
        edge_index=save_dict['edge_index'],
        edge_attr=save_dict.get('edge_attr', None),
    )
    if 'y' in save_dict:
        data.y = save_dict['y']
    return data


def graph_statistics(data: Data) -> dict:
    """Compute summary statistics for a graph."""
    num_nodes = data.x.shape[0]
    num_edges = data.edge_index.shape[1]
    
    stats = {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'feature_dim': data.x.shape[1],
        'avg_degree': num_edges / max(num_nodes, 1),
        'feature_mean': float(data.x.mean()),
        'feature_std': float(data.x.std()),
    }
    return stats
