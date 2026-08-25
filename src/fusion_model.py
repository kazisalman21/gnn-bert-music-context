"""
Fusion model module for GNN-BERT fusion (Task 3).

Implements all four required ablation conditions:
1. BERT-only: CLS -> classifier
2. GNN-only: graph readout -> classifier
3. Early concatenation: z = concat(g, t) -> MLP -> classifier
4. Cross-attention: Q from graph, K/V from BERT tokens -> fused classifier

Cross-attention follows faculty formulation:
    Q = g @ W_Q
    K = H_text @ W_K
    V = H_text @ W_V
    A = softmax(QK^T / sqrt(d))
    z = concat(g, A @ V)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.gnn_model import GraphSAGEEncoder
from src.bert_encoder import BertEncoder


class CrossAttentionFusion(nn.Module):
    """Cross-attention module: graph queries attend to BERT token embeddings."""
    
    def __init__(self, graph_dim: int, text_dim: int, projection_dim: int,
                 num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.head_dim = projection_dim // num_heads
        assert projection_dim % num_heads == 0, "projection_dim must be divisible by num_heads"
        
        self.W_Q = nn.Linear(graph_dim, projection_dim)
        self.W_K = nn.Linear(text_dim, projection_dim)
        self.W_V = nn.Linear(text_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(projection_dim, projection_dim)
    
    def forward(self, graph_emb, text_embeddings, attention_mask=None):
        """
        Args:
            graph_emb: (B, graph_dim) graph-level representation
            text_embeddings: (B, seq_len, text_dim) BERT token embeddings
            attention_mask: (B, seq_len) padding mask (1 = valid, 0 = pad)
        
        Returns:
            attended: (B, projection_dim) attention output
            attention_weights: (B, num_heads, 1, seq_len) for visualization
        """
        B, seq_len, _ = text_embeddings.shape
        
        # Q from graph: (B, 1, projection_dim)
        Q = self.W_Q(graph_emb).unsqueeze(1)
        # K, V from text: (B, seq_len, projection_dim)
        K = self.W_K(text_embeddings)
        V = self.W_V(text_embeddings)
        
        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, d)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, 1, S)
        
        # Apply padding mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, 1, S)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of values
        attended = torch.matmul(attention_weights, V)  # (B, H, 1, d)
        attended = attended.transpose(1, 2).contiguous().view(B, self.projection_dim)  # (B, proj_dim)
        attended = self.out_proj(attended)
        
        return attended, attention_weights


class FusionModel(nn.Module):
    """GNN-BERT fusion model supporting all four ablation modes.
    
    Modes:
        'bert_only': uses only BERT CLS -> classifier
        'gnn_only': uses only GNN readout -> classifier
        'concat': z = concat(g, t) -> MLP -> classifier
        'cross_attention': cross-attention fusion -> classifier
    """
    
    def __init__(self, gnn_input_dim: int, num_labels: int,
                 gnn_hidden_dim: int = 128, gnn_num_layers: int = 2,
                 bert_model_name: str = "distilbert-base-uncased",
                 bert_hidden_dim: int = 768, freeze_bert: bool = False,
                 fusion_mode: str = "cross_attention",
                 projection_dim: int = 128, num_attention_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.num_labels = num_labels
        self.gnn_hidden_dim = gnn_hidden_dim
        self.bert_hidden_dim = bert_hidden_dim
        
        # Instantiate only the encoders used by a mode. This keeps ablation
        # checkpoints unambiguous and avoids saving hundreds of MB of unused
        # parameters in the GNN-only condition.
        if fusion_mode != 'bert_only':
            self.gnn_encoder = GraphSAGEEncoder(
                gnn_input_dim, gnn_hidden_dim, gnn_num_layers, dropout
            )

        if fusion_mode != 'gnn_only':
            self.bert_encoder = BertEncoder(
                num_labels=num_labels,
                model_name=bert_model_name,
                freeze_bert=freeze_bert,
                hidden_dim=bert_hidden_dim
            )
            if fusion_mode not in {'bert_only'}:
                # Concat/cross-attention consume encode() outputs and never use
                # BertEncoder's standalone classifier. Removing it keeps the
                # ablation parameter counts and checkpoints unambiguous.
                self.bert_encoder.classifier = nn.Identity()
        
        # Fusion-specific components
        if fusion_mode == 'bert_only':
            # Use BERT's own classifier
            pass
        elif fusion_mode == 'gnn_only':
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(gnn_hidden_dim, gnn_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(gnn_hidden_dim // 2, num_labels)
            )
        elif fusion_mode == 'concat':
            concat_dim = gnn_hidden_dim + bert_hidden_dim
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(concat_dim, concat_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(concat_dim // 2, num_labels)
            )
        elif fusion_mode == 'cross_attention':
            self.cross_attention = CrossAttentionFusion(
                gnn_hidden_dim, bert_hidden_dim, projection_dim,
                num_attention_heads, dropout
            )
            fused_dim = gnn_hidden_dim + projection_dim
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(fused_dim, fused_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(fused_dim // 2, num_labels)
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    def forward(self, graph_x, graph_edge_index, graph_batch,
                input_ids, attention_mask):
        """
        Args:
            graph_x: node features (num_nodes_total, input_dim)
            graph_edge_index: edge indices (2, num_edges_total)
            graph_batch: batch assignment (num_nodes_total,)
            input_ids: (B, seq_len) BERT input token IDs
            attention_mask: (B, seq_len) BERT attention mask
        
        Returns:
            logits: (B, num_labels) pre-sigmoid logits
            extras: dict with embeddings for t-SNE, attention weights, etc.
        """
        extras = {}
        
        if self.fusion_mode == 'bert_only':
            cls_emb, _ = self.bert_encoder.encode(input_ids, attention_mask)
            logits = self.bert_encoder.classifier(cls_emb)
            extras['z'] = cls_emb.detach()
            return logits, extras
        
        if self.fusion_mode == 'gnn_only':
            graph_emb, _ = self.gnn_encoder(graph_x, graph_edge_index, graph_batch)
            logits = self.classifier(graph_emb)
            extras['z'] = graph_emb.detach()
            return logits, extras
        
        # Both GNN and BERT needed for concat and cross_attention
        graph_emb, node_embs = self.gnn_encoder(graph_x, graph_edge_index, graph_batch)
        cls_emb, token_embs = self.bert_encoder.encode(input_ids, attention_mask)
        
        if self.fusion_mode == 'concat':
            z = torch.cat([graph_emb, cls_emb], dim=-1)
            logits = self.classifier(z)
            extras['z'] = z.detach()
            return logits, extras
        
        if self.fusion_mode == 'cross_attention':
            attended, attn_weights = self.cross_attention(
                graph_emb, token_embs, attention_mask
            )
            z = torch.cat([graph_emb, attended], dim=-1)
            logits = self.classifier(z)
            extras['z'] = z.detach()
            extras['attention_weights'] = attn_weights.detach()
            return logits, extras
        
        raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
    
    def count_parameters(self) -> dict:
        """Count trainable and total parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}
