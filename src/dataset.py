"""
Dataset utilities module.

Handles:
- GTZAN genre dataset loading (Task 2)
- MagnaTagATune loading with disjoint context/target tag split (Task 3)
- MusicCaps loading with aspect_list tags (Task 1, 4)
- Context text sentence construction for MTAT
- Split management
- PyTorch Dataset/DataLoader wrappers
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from pathlib import Path
from typing import Optional, Tuple
import yaml


# --- Tag partitions for Task 3 (MTAT disjoint context-to-target) ---

CONTEXT_TAGS = [
    "guitar", "strings", "drums", "piano", "violin", "vocal", "synth",
    "female", "male", "singing", "vocals", "no vocals", "harpsichord",
    "flute", "woman", "male vocal", "no vocal", "sitar", "solo", "man",
    "choir", "voice", "male voice", "female vocal", "beats", "harp",
    "cello", "no voice", "female voice", "choral", "beat"
]

TARGET_GENRE_TAGS = [
    "classical", "techno", "electronic", "rock", "pop", "country",
    "metal", "indian", "opera", "new age", "dance", "classic"
]

TARGET_MOOD_TAGS = [
    "slow", "fast", "ambient", "loud", "quiet", "soft", "weird"
]

TARGET_TAGS = TARGET_GENRE_TAGS + TARGET_MOOD_TAGS  # 19 total


def build_context_sentence(active_context_tags: list) -> str:
    """Build a natural-language template sentence from active context tags.
    
    Example: ["guitar", "vocals", "drums"] ->
    "This music features guitar, vocals, and drums."
    """
    if not active_context_tags:
        return "This music features no described instruments or vocals."
    
    if len(active_context_tags) == 1:
        return f"This music features {active_context_tags[0]}."
    elif len(active_context_tags) == 2:
        return f"This music features {active_context_tags[0]} and {active_context_tags[1]}."
    else:
        items = ", ".join(active_context_tags[:-1])
        return f"This music features {items}, and {active_context_tags[-1]}."


# --- GTZAN Dataset ---

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]


class GTZANDataset(Dataset):
    """GTZAN genre dataset for Task 2.
    
    Expects structure: data/raw/gtzan/genres_original/{genre}/{genre}.xxxxx.wav
    """
    
    def __init__(self, data_dir: str, split_ids: list,
                 processed_dir: Optional[str] = None):
        """
        Args:
            data_dir: path to GTZAN genres_original directory
            split_ids: list of track IDs to include (e.g., ["blues.00000", ...])
            processed_dir: path to preprocessed features (graphs, mel-specs)
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.split_ids = split_ids
        self.genre_to_idx = {g: i for i, g in enumerate(GTZAN_GENRES)}
    
    def __len__(self):
        return len(self.split_ids)
    
    def __getitem__(self, idx):
        track_id = self.split_ids[idx]
        genre = track_id.split(".")[0]
        label = self.genre_to_idx[genre]
        
        if self.processed_dir:
            # Load preprocessed graph
            graph_path = self.processed_dir / f"{track_id}.pt"
            if graph_path.exists():
                graph_data = torch.load(graph_path, weights_only=False)
                data = Data(
                    x=graph_data['x'],
                    edge_index=graph_data['edge_index'],
                    edge_attr=graph_data.get('edge_attr'),
                )
                data.y = torch.tensor(label, dtype=torch.long)
                data.track_id = track_id
                return data
        
        # Return minimal info if no preprocessed data
        return {'track_id': track_id, 'label': label, 'genre': genre}


# --- MagnaTagATune Dataset ---

class MTATDataset(Dataset):
    """MagnaTagATune dataset for Task 3.
    
    Uses disjoint context-to-target formulation:
    - Context tags (31 instrument/vocal) -> BERT text input
    - Target tags (19 genre/mood) -> prediction targets
    """
    
    def __init__(self, annotations: pd.DataFrame, split_ids: list,
                 all_tags: list, processed_dir: Optional[str] = None,
                 tokenizer=None):
        """
        Args:
            annotations: DataFrame with clip_id as index, tag columns
            split_ids: list of clip IDs for this split
            all_tags: list of all 50 tag names (column names in annotations)
            processed_dir: path to preprocessed graph .pt files
            tokenizer: TextTokenizer instance for BERT encoding
        """
        self.annotations = annotations
        self.split_ids = [sid for sid in split_ids if sid in annotations.index]
        self.all_tags = all_tags
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.tokenizer = tokenizer
        
        # Identify context and target tag column indices
        self.context_tag_indices = [i for i, t in enumerate(all_tags) if t in CONTEXT_TAGS]
        self.target_tag_indices = [i for i, t in enumerate(all_tags) if t in TARGET_TAGS]
        self.context_tag_names = [all_tags[i] for i in self.context_tag_indices]
        self.target_tag_names = [all_tags[i] for i in self.target_tag_indices]
    
    def __len__(self):
        return len(self.split_ids)
    
    def __getitem__(self, idx):
        clip_id = self.split_ids[idx]
        row = self.annotations.loc[clip_id]
        
        # Get all tag values
        tag_values = row[self.all_tags].values.astype(float)
        
        # Context tags -> text sentence
        active_context = [self.context_tag_names[i]
                          for i, ci in enumerate(self.context_tag_indices)
                          if tag_values[ci] > 0]
        context_text = build_context_sentence(active_context)
        
        # Target tags -> binary vector
        target_vector = np.array([tag_values[i] for i in self.target_tag_indices],
                                  dtype=np.float32)
        
        result = {
            'clip_id': clip_id,
            'text': context_text,
            'targets': torch.tensor(target_vector, dtype=torch.float32),
        }
        
        # Tokenize if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(context_text)
            result['input_ids'] = tokens['input_ids']
            result['attention_mask'] = tokens['attention_mask']
        
        # Load preprocessed graph if available
        if self.processed_dir:
            # Sanitize clip_id for filename
            safe_id = clip_id.replace("/", "_").replace("\\", "_")
            graph_path = self.processed_dir / f"{safe_id}.pt"
            if graph_path.exists():
                graph_data = torch.load(graph_path, weights_only=False)
                result['graph_x'] = graph_data['x']
                result['graph_edge_index'] = graph_data['edge_index']
                result['graph_edge_attr'] = graph_data.get('edge_attr')
        
        return result


# --- MusicCaps Dataset ---

class MusicCapsDataset(Dataset):
    """MusicCaps dataset for Task 1 (caption -> tags) and Task 4 (contrastive).
    
    Uses expert captions as text input and aspect_list as multi-label targets.
    """
    
    def __init__(self, metadata: pd.DataFrame, split_ids: list,
                 tag_vocabulary: list,
                 processed_dir: Optional[str] = None,
                 tokenizer=None):
        """
        Args:
            metadata: DataFrame with ytid, caption, aspect_list columns
            split_ids: list of ytid values for this split
            tag_vocabulary: list of tag strings to use as target labels
            processed_dir: path to preprocessed audio/graph files
            tokenizer: TextTokenizer instance
        """
        self.metadata = metadata
        self.split_ids = [sid for sid in split_ids if sid in metadata['ytid'].values]
        self.tag_vocabulary = tag_vocabulary
        self.tag_to_idx = {t: i for i, t in enumerate(tag_vocabulary)}
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.split_ids)
    
    def __getitem__(self, idx):
        ytid = self.split_ids[idx]
        row = self.metadata[self.metadata['ytid'] == ytid].iloc[0]
        
        caption = str(row['caption'])
        
        # Parse aspect_list to multi-label vector
        aspect_list = row.get('aspect_list', '[]')
        if isinstance(aspect_list, str):
            # Handle string representation of list
            try:
                tags = json.loads(aspect_list.replace("'", '"'))
            except (json.JSONDecodeError, ValueError):
                tags = [t.strip() for t in aspect_list.strip("[]").split(",")]
        elif isinstance(aspect_list, list):
            tags = aspect_list
        else:
            tags = []
        
        # Build target vector
        target = np.zeros(len(self.tag_vocabulary), dtype=np.float32)
        for tag in tags:
            tag = tag.strip().lower()
            if tag in self.tag_to_idx:
                target[self.tag_to_idx[tag]] = 1.0
        
        result = {
            'ytid': ytid,
            'caption': caption,
            'targets': torch.tensor(target, dtype=torch.float32),
        }
        
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(caption)
            result['input_ids'] = tokens['input_ids']
            result['attention_mask'] = tokens['attention_mask']
        
        return result


# --- Split utilities ---

def save_split(split_ids: list, path: str):
    """Save split IDs to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(split_ids, f, indent=2)


def load_split(path: str) -> list:
    """Load split IDs from JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
