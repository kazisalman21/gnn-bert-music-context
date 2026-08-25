"""
Task 4: Contrastive dual-encoder for cross-modal alignment (MusicCaps).

Implements:
- Dual encoder: GNN -> projection -> L2-normalize; BERT -> projection -> L2-normalize
- InfoNCE contrastive loss with temperature tau
- Bidirectional retrieval: Caption->Audio and Audio->Caption
- R@1, R@5, R@10 evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn_model import GraphSAGEEncoder
from src.bert_encoder import BertEncoder


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss.
    
    L = -log(exp(sim(g_i, t_i) / tau) / sum_j exp(sim(g_i, t_j) / tau))
    
    Symmetric version averages graph->text and text->graph directions.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, graph_emb, text_emb):
        """
        Args:
            graph_emb: (B, dim) L2-normalized graph embeddings
            text_emb: (B, dim) L2-normalized text embeddings
        
        Returns:
            loss: scalar InfoNCE loss (symmetric)
        """
        # Similarity matrix: (B, B)
        sim = torch.matmul(graph_emb, text_emb.t()) / self.temperature
        
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)
        
        # Graph -> Text direction
        loss_g2t = F.cross_entropy(sim, labels)
        # Text -> Graph direction
        loss_t2g = F.cross_entropy(sim.t(), labels)
        
        return (loss_g2t + loss_t2g) / 2


class ContrastiveDualEncoder(nn.Module):
    """Dual encoder for graph-text alignment.
    
    Audio graph -> GNN -> projection -> L2-normalize
    Caption -> BERT -> projection -> L2-normalize
    """
    
    def __init__(self, gnn_input_dim: int, gnn_hidden_dim: int = 128,
                 gnn_num_layers: int = 2,
                 bert_model_name: str = "distilbert-base-uncased",
                 bert_hidden_dim: int = 768, freeze_bert: bool = False,
                 projection_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.gnn_encoder = GraphSAGEEncoder(
            gnn_input_dim, gnn_hidden_dim, gnn_num_layers, dropout
        )
        
        # BERT encoder (we only use the encode method, not the classifier)
        self.bert = BertEncoder(
            num_labels=1,
            model_name=bert_model_name,
            freeze_bert=freeze_bert,
            hidden_dim=bert_hidden_dim
        )
        self.bert.classifier = nn.Identity()
        
        # Projection heads
        self.graph_proj = nn.Sequential(
            nn.Linear(gnn_hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def encode_graph(self, x, edge_index, batch=None):
        """Encode graph to normalized embedding."""
        graph_emb, _ = self.gnn_encoder(x, edge_index, batch)
        projected = self.graph_proj(graph_emb)
        return F.normalize(projected, p=2, dim=-1)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text to normalized embedding."""
        cls_emb, _ = self.bert.encode(input_ids, attention_mask)
        projected = self.text_proj(cls_emb)
        return F.normalize(projected, p=2, dim=-1)
    
    def forward(self, graph_x, graph_edge_index, graph_batch,
                input_ids, attention_mask):
        """Return normalized graph and text embeddings."""
        graph_emb = self.encode_graph(graph_x, graph_edge_index, graph_batch)
        text_emb = self.encode_text(input_ids, attention_mask)
        return graph_emb, text_emb


def compute_retrieval_metrics(graph_embs, text_embs, ks=(1, 5, 10)):
    """Compute retrieval R@K metrics in both directions.
    
    Args:
        graph_embs: (N, dim) normalized graph embeddings
        text_embs: (N, dim) normalized text embeddings
        ks: tuple of K values for R@K
    
    Returns:
        dict with 'caption_to_audio' and 'audio_to_caption' R@K values
    """
    # Similarity matrix: (N, N)
    sim = torch.matmul(text_embs, graph_embs.t())
    N = sim.shape[0]
    
    results = {'caption_to_audio': {}, 'audio_to_caption': {}}
    
    max_k = min(max(ks), N)
    for direction, sims in [('caption_to_audio', sim), ('audio_to_caption', sim.t())]:
        # For each query, rank all candidates
        _, indices = sims.topk(max_k, dim=1)
        gt = torch.arange(N, device=sim.device).unsqueeze(1)
        
        for k in ks:
            effective_k = min(k, N)
            hits = (indices[:, :effective_k] == gt).any(dim=1).float()
            results[direction][f'R@{k}'] = hits.mean().item()
    
    return results
