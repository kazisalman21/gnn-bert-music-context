"""
Training module.

Unified training loop for all models with:
- Seed setting and device selection
- Train/validation/test discipline
- Loss and metric logging per epoch
- Early stopping based on validation metric
- Best checkpoint saving
- NaN/Inf detection
- Gradient clipping
- Mixed precision support (optional)
- Total training time recording
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, Callable
import copy


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Get compute device."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TrainingLogger:
    """Simple training logger that records metrics per epoch."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'lr': [],
            'epoch': [],
        }
        self.best_val_score = -float('inf')
        self.best_epoch = 0
    
    def log(self, epoch: int, train_loss: float, val_loss: float,
            train_metrics: dict, val_metrics: dict, lr: float):
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_metrics'].append(train_metrics)
        self.history['val_metrics'].append(val_metrics)
        self.history['lr'].append(lr)
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)


def train_epoch(model, dataloader, criterion, optimizer, device,
                is_graph_model: bool = False, gradient_clip: float = 1.0):
    """Train for one epoch.
    
    Returns:
        epoch_loss: average loss over batches
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if is_graph_model:
            # PyG batched graph data
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y
        else:
            # Standard tensor batch
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            targets = batch['labels'].to(device)
            logits = model(**inputs)
        
        loss = criterion(logits, targets)
        
        # NaN check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected. Skipping batch.")
            continue
        
        loss.backward()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device,
                   is_graph_model: bool = False):
    """Evaluate model on a dataset.
    
    Returns:
        epoch_loss: average loss
        all_logits: numpy array of model outputs
        all_targets: numpy array of targets
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_logits = []
    all_targets = []
    
    for batch in dataloader:
        if is_graph_model:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y
        else:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            targets = batch['labels'].to(device)
            logits = model(**inputs)
        
        loss = criterion(logits, targets)
        total_loss += loss.item()
        num_batches += 1
        
        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return total_loss / max(num_batches, 1), all_logits, all_targets


def train_model(model, train_loader, val_loader, criterion, optimizer,
                scheduler, device, num_epochs: int, patience: int,
                checkpoint_path: str, is_graph_model: bool = False,
                gradient_clip: float = 1.0,
                val_metric_fn: Optional[Callable] = None,
                val_metric_name: str = 'macro_f1'):
    """Full training loop with early stopping and checkpointing.
    
    Args:
        val_metric_fn: function(logits, targets) -> float metric (higher is better)
        val_metric_name: name of the metric for logging
    
    Returns:
        logger: TrainingLogger with full history
        best_model_state: state dict of best model
    """
    logger = TrainingLogger()
    best_val_score = -float('inf')
    best_model_state = None
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device,
            is_graph_model, gradient_clip
        )
        
        # Validate
        val_loss, val_logits, val_targets = evaluate_epoch(
            model, val_loader, criterion, device, is_graph_model
        )
        
        # Compute validation metric
        train_metrics = {'loss': train_loss}
        val_metrics = {'loss': val_loss}
        
        if val_metric_fn is not None:
            val_score = val_metric_fn(val_logits, val_targets)
            val_metrics[val_metric_name] = val_score
        else:
            val_score = -val_loss  # lower loss = better
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()
        
        logger.log(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)
        
        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val {val_metric_name}: {val_score:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Checkpoint best model
        if val_score > best_val_score:
            best_val_score = val_score
            best_model_state = copy.deepcopy(model.state_dict())
            logger.best_val_score = best_val_score
            logger.best_epoch = epoch
            
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': best_val_score,
                'val_metric_name': val_metric_name,
            }, checkpoint_path)
            
            patience_counter = 0
            print(f"  ✓ Best model saved (epoch {epoch}, {val_metric_name}={val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break
    
    total_time = time.time() - start_time
    logger.history['total_training_time_sec'] = total_time
    logger.history['best_epoch'] = logger.best_epoch
    logger.history['best_val_score'] = logger.best_val_score
    logger.history['device'] = str(device)
    
    print(f"\nTraining complete. Best epoch: {logger.best_epoch}, "
          f"Best {val_metric_name}: {logger.best_val_score:.4f}, "
          f"Total time: {total_time:.1f}s")
    
    return logger, best_model_state
