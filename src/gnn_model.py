"""
GNN model module.

GraphSAGE encoder using PyTorch Geometric SAGEConv layers:
- 2-3 layer message passing
- Mean pooling graph readout
- Classification head
- Supports both cross-entropy (single-label) and BCE (multi-label)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for music segment graphs.
    
    Performs message passing on graph, then mean pooling for graph-level readout.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.dropout = dropout
        self.hidden_dim = hidden_dim
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass through GraphSAGE layers + mean pool readout.
        
        Args:
            x: (num_nodes_total, input_dim) node features
            edge_index: (2, num_edges_total) edge indices
            batch: (num_nodes_total,) batch assignment vector
        
        Returns:
            graph_embedding: (batch_size, hidden_dim) graph-level representation
            node_embeddings: (num_nodes_total, hidden_dim) final node embeddings
        """
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        node_embeddings = x
        
        # Graph-level readout (mean pooling)
        if batch is None:
            # Single graph
            graph_embedding = x.mean(dim=0, keepdim=True)
        else:
            graph_embedding = global_mean_pool(x, batch)
        
        return graph_embedding, node_embeddings


class GNNClassifier(nn.Module):
    """Full GNN classifier: GraphSAGE encoder + classification head."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_classes: int = 10,
                 dropout: float = 0.3):
        super().__init__()
        self.encoder = GraphSAGEEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x, edge_index, batch=None):
        """Returns logits (pre-softmax for CE or pre-sigmoid for BCE)."""
        graph_emb, _ = self.encoder(x, edge_index, batch)
        logits = self.classifier(graph_emb)
        return logits
