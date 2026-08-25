"""
BERT/DistilBERT encoder module.

Wraps HuggingFace distilbert-base-uncased for text encoding.
- Uses last_hidden_state[:, 0, :] for CLS pooling (DistilBERT has no pooler_output)
- Tokenization with max_length=128, truncation, padding, attention masks
- Classification head with BCE-with-logits loss for multi-label tasks
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer


class BertEncoder(nn.Module):
    """DistilBERT-based text encoder with classification head.
    
    For Task 1 (MusicCaps caption -> tags) and Task 3 (MTAT context -> targets).
    """
    
    def __init__(self, num_labels: int, model_name: str = "distilbert-base-uncased",
                 freeze_bert: bool = False, hidden_dim: int = 768):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_labels)
        )
    
    def encode(self, input_ids, attention_mask):
        """Get CLS representation from DistilBERT.
        
        Returns:
            cls_output: (batch_size, hidden_dim)
            token_embeddings: (batch_size, seq_len, hidden_dim) for cross-attention
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        cls_output = token_embeddings[:, 0, :]  # (B, hidden_dim) - CLS token
        return cls_output, token_embeddings
    
    def forward(self, input_ids, attention_mask):
        """Full forward: BERT encoding + classification.
        
        Returns logits (pre-sigmoid for BCEWithLogitsLoss).
        """
        cls_output, _ = self.encode(input_ids, attention_mask)
        logits = self.classifier(cls_output)
        return logits


class TextTokenizer:
    """Wrapper around DistilBERT tokenizer for consistent preprocessing."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 128):
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def tokenize(self, text: str) -> dict:
        """Tokenize a single text string.
        
        Returns dict with input_ids and attention_mask tensors.
        """
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }
    
    def tokenize_batch(self, texts: list) -> dict:
        """Tokenize a batch of text strings."""
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
