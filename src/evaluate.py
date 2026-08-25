"""
Evaluation module.

Computes all faculty-required metrics:
- Precision, Recall, F1 (per-class, Macro, Micro)
- AUC-PR (per-tag, mean)
- Confusion matrix (for single-label)
- MAE, R² (for DEAM emotion regression, optional)
- Threshold tuning on validation data only

All metrics are computed from actual model outputs and saved to results/metrics.json.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
    mean_absolute_error, r2_score, accuracy_score
)
from typing import Optional


def compute_multilabel_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray,
                                threshold: float = 0.5,
                                tag_names: Optional[list] = None) -> dict:
    """Compute multi-label classification metrics.
    
    Args:
        y_true: (N, K) binary ground truth
        y_pred_probs: (N, K) predicted probabilities (post-sigmoid)
        threshold: classification threshold
        tag_names: optional list of tag names for per-tag reporting
    
    Returns:
        dict with macro_f1, micro_f1, per_tag metrics, mean_auc_pr, etc.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    # Macro and Micro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    # Per-tag AUC-PR
    K = y_true.shape[1]
    per_tag_auc_pr = []
    per_tag_metrics = []
    
    for k in range(K):
        tag_name = tag_names[k] if tag_names else f"tag_{k}"
        
        # AUC-PR (skip if tag has no positive or no negative examples)
        if y_true[:, k].sum() > 0 and y_true[:, k].sum() < len(y_true[:, k]):
            auc_pr = average_precision_score(y_true[:, k], y_pred_probs[:, k])
        else:
            auc_pr = float('nan')
        
        per_tag_auc_pr.append(auc_pr)
        
        tag_f1 = f1_score(y_true[:, k], y_pred[:, k], zero_division=0)
        per_tag_metrics.append({
            'tag': tag_name,
            'f1': float(tag_f1),
            # JSON has no NaN value. None correctly records an AUC that is
            # undefined because this evaluation split has only one class.
            'auc_pr': None if np.isnan(auc_pr) else float(auc_pr),
            'support': int(y_true[:, k].sum())
        })
    
    # Mean AUC-PR (excluding NaN tags)
    valid_aucs = [a for a in per_tag_auc_pr if not np.isnan(a)]
    mean_auc_pr = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    
    return {
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'mean_auc_pr': float(mean_auc_pr),
        'threshold': float(threshold),
        'num_samples': int(y_true.shape[0]),
        'num_tags': int(K),
        'per_tag': per_tag_metrics
    }


def compute_singlelabel_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 class_names: Optional[list] = None) -> dict:
    """Compute single-label classification metrics.
    
    Args:
        y_true: (N,) integer class labels
        y_pred: (N,) integer predicted labels
        class_names: optional list of class names
    """
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    
    result = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'num_samples': int(len(y_true)),
        'confusion_matrix': cm.tolist()
    }
    
    if class_names:
        result['class_names'] = class_names
    
    return result


def tune_threshold(y_true: np.ndarray, y_pred_probs: np.ndarray,
                   metric: str = 'macro_f1',
                   thresholds: Optional[list] = None) -> float:
    """Find optimal classification threshold on validation data.
    
    Args:
        y_true: (N, K) binary ground truth (VALIDATION set only!)
        y_pred_probs: (N, K) predicted probabilities
        metric: metric to optimize ('macro_f1' or 'micro_f1')
        thresholds: list of thresholds to try
    
    Returns:
        best threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05).tolist()
    
    best_score = -1
    best_threshold = 0.5
    
    for t in thresholds:
        y_pred = (y_pred_probs >= t).astype(int)
        if metric == 'macro_f1':
            score = f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif metric == 'micro_f1':
            score = f1_score(y_true, y_pred, average='micro', zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = t
    
    return best_threshold


def save_metrics(metrics: dict, path: str):
    """Save metrics dict to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)


def load_metrics(path: str) -> dict:
    """Load metrics from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
