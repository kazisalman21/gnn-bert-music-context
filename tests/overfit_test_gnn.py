"""
Overfit test: GNNClassifier on ~50 GTZAN graphs.

Verifies the model can memorize a tiny training subset.
Expected: training loss drops significantly, training accuracy approaches ~100%.
Does NOT require literal zero loss — just strong memorization.
"""

import json
import sys
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.data import Data, Batch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.gnn_model import GNNClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load 50 GTZAN graphs (5 per genre)
PROCESSED_DIR = ROOT / "data/processed/gtzan_graphs"
with (ROOT / "data/splits/gtzan_splits.json").open(encoding="utf-8") as handle:
    splits = json.load(handle)
GENRES = splits['genres']

# Take first 5 from each genre in train set
subset_ids = []
genre_count = {g: 0 for g in GENRES}
for tid in splits['train']:
    genre = tid.split('.')[0]
    if genre_count[genre] < 5:
        subset_ids.append(tid)
        genre_count[genre] += 1
    if len(subset_ids) >= 50:
        break

print(f"Overfit subset: {len(subset_ids)} graphs")
print(f"Per genre: {genre_count}")

# Load graphs
graphs = []
for tid in subset_ids:
    path = PROCESSED_DIR / f"{tid}.pt"
    data = torch.load(path, weights_only=False)
    # Reconstruct PyG Data
    g = Data(x=data['x'], edge_index=data['edge_index'])
    g.y = data['y'] if isinstance(data['y'], torch.Tensor) else torch.tensor(data['y'], dtype=torch.long)
    graphs.append(g)

print(f"Loaded {len(graphs)} graphs, feature dim={graphs[0].x.shape[1]}")

# Model
model = GNNClassifier(
    input_dim=graphs[0].x.shape[1], hidden_dim=128,
    num_layers=2, num_classes=10, dropout=0.0  # No dropout for overfit test
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train for 200 epochs
print("\nTraining (overfit test, 200 epochs):")
print("-" * 50)

batch = Batch.from_data_list(graphs).to(device)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    
    logits = model(batch.x, batch.edge_index, batch.batch)
    loss = criterion(logits, batch.y)
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        acc = (preds == batch.y).float().mean().item()
    
    if epoch <= 10 or epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc:.1%}")

print("-" * 50)
final_loss = loss.item()
final_acc = acc

if final_acc >= 0.95:
    print(f"OVERFIT TEST PASSED: acc={final_acc:.1%}, loss={final_loss:.4f}")
    print("Model can strongly memorize a tiny training subset.")
elif final_acc >= 0.80:
    raise SystemExit(
        f"OVERFIT TEST MARGINAL: acc={final_acc:.1%}; expected at least 95%."
    )
else:
    raise SystemExit(
        f"OVERFIT TEST FAILED: acc={final_acc:.1%}; architecture or training bug likely."
    )
