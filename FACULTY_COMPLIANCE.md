# Faculty instruction compliance checklist

This checklist maps the requirements in `CSE425_Project_GNN_BERT_Music_Context.pdf` to the
implementation and generated evidence. `Course_Content.pdf` is the official CSE425 Neural
Networks course outline and is used as a second alignment reference. A requirement is marked
complete only after a real run has produced its artifact. Missing experiments are not replaced
with estimated or synthetic scores.

## Course-outline alignment

| CSE425 topic | Project evidence |
|---|---|
| CNNs and feature extraction | Mel-spectrogram CNN baseline and 128-bin log-mel features in Task 2 |
| Transformers, attention, and BERT | DistilBERT baseline and graph-to-token cross-attention in Tasks 1 and 3 |
| Optimization and regularization | AdamW, gradient clipping, dropout, learning-rate scheduling, and early stopping |
| Transfer learning | Pretrained DistilBERT weights with controlled freezing and fine-tuning |
| Explainability and trustworthy comparison | Ablations, confusion matrices, per-tag scores, attention views, split audits, and stated limitations |
| End-to-end neural-network solution | Real-data preprocessing, training, checkpointing, held-out evaluation, plots, and report generation |

## Project motivation and problem definition

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Explain why music context needs both audio structure and text | Complete in source | `report/final_report.tex`, Introduction |
| Formulate nodes, edges, text input, targets, and GNN--BERT fusion | Complete in source | `report/final_report.tex`, Method; `src/graph_builder.py`; `src/fusion_model.py` |
| Use graph-level prediction with clearly stated labels | Complete | GTZAN genre labels for Task 2; 19 disjoint genre/mood targets for Task 3 |

## Dataset and preprocessing requirements

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Use MusicCaps metadata/captions | Complete | 5,521 real metadata rows; frozen splits in `data/splits/musiccaps_*.json` |
| Use a graph-ready music dataset such as GTZAN/FMA | Complete | 999 usable GTZAN tracks in `data/processed/gtzan_graphs/` |
| Use MagnaTagATune where appropriate | Complete | 25,860 real five-second graphs; official 0--b/c/d--f split |
| Resample audio to 22,050 Hz | Complete | `config.yaml`; both preprocessing scripts |
| Extract 128-bin log-mel and 12-bin chroma information | Complete | cached GTZAN log-mels; chroma mean/std is included in each 77-D graph node |
| Normalize features | Complete | per-track, per-feature standardization in `src/audio_features.py` |
| Use fixed or beat-synchronous segmentation | Complete | fixed non-overlapping five-second GTZAN/MTAT segments |
| Build temporal and acoustic-similarity graph edges | Complete | `src/graph_builder.py`, threshold $\tau=0.5$ plus adjacent-segment edges |
| Supply at least 20 graph samples | Complete | 999 GTZAN graphs and 25,860 MTAT graphs |
| Use fixed train/validation/test splits and prevent leakage | Complete for current splits | MusicCaps YouTube-ID overlap 0; MTAT clip/exact-song overlap 0; GTZAN exact-audio cross-split duplicate groups 0 |

## Task 1: BERT text baseline

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Caption/text to multi-label tags | Complete | MusicCaps caption to 191 training-vocabulary aspect tags |
| Fine-tune BERT/DistilBERT | Complete | frozen stage followed by full-backbone fine-tuning; `scripts/run_task1.py` |
| Report Macro-F1 and Micro-F1 | Complete | real held-out scores in `results/task1_results.json` |
| Plot learning curves | Complete | `plots/task1_f1_curves.png` |
| Show prediction examples | Complete | five held-out examples in `results/task1_results.json` |

## Task 2: audio graph model and CNN comparison

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Train GraphSAGE or GAT on music graphs | Complete | real duplicate-safe GTZAN GraphSAGE checkpoint and result JSON |
| Compare with a CNN on log-mel features | Complete | same 797/99/103 split and labels; real MelCNN checkpoint and result JSON |
| Report Accuracy and Macro-F1 | Complete | `results/task2_results.json` |
| Include confusion matrices and model comparison | Complete | five Task 2 PNG files under `plots/` |

## Task 3: GNN--BERT fusion

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Combine graph and text context | Complete | trained MTAT graph plus annotation-derived instrument/vocal context sentence |
| Keep textual context separate from prediction target | Complete by design | 31 context tags and 19 lexically disjoint genre/mood targets |
| Implement BERT-only, GNN-only, concatenation, and cross-attention | Complete in code | `src/fusion_model.py`; `scripts/run_task3.py` |
| Report Macro-F1 and AUC-PR for all ablations | Complete | four real held-out runs in `results/task3_results.json` |
| Plot genre and mood t-SNE | Complete | held-out fused embeddings; labels use ground truth only |
| Show three graph-path/text-alignment examples | Complete | real MTAT low/median/high cases plus three held-out MusicCaps transfer cases |

## Task 4: advanced contrastive retrieval

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Train an audio--text contrastive model | Complete | real recovered MusicCaps intervals, symmetric InfoNCE, and validation-selected checkpoint in `scripts/run_task4.py` |
| Report Recall@1, Recall@5, and Recall@10 | Complete | both caption-to-audio and audio-to-caption metrics in `results/task4_results.json` |
| Show 10 retrieval examples and zero-shot use | Complete | `retrieval_examples/task4_top3_retrievals.*`; caption-to-tag-prompt and Task 3 direct-transfer comparison |

## Baselines, figures, report, and submission

| Faculty requirement | Status | Evidence |
|---|---:|---|
| Include at least two baselines | Complete in code/results | training-prevalence baselines plus Task 2 CNN and Task 3 single-modality ablations |
| Provide result tables, curves, heatmaps, and important plots | Complete | generated from saved Task 1--4 result files under `plots/` and `results/` |
| Provide `notebooks/demo_context.ipynb` | Complete and packaged | real held-out GTZAN graph inference; the graph and small trained checkpoint are included in the ZIP |
| Provide an EDA notebook | Complete in source | `notebooks/eda.ipynb`, reading real split/data/result files |
| Submit a 6--10 page IEEE/NeurIPS/ICML-style report | Complete | visually inspected `report/final_report.pdf`; IEEEtran source and generated macros included |
| Include reproducible code and requirements | Complete in source | portable `requirements.txt` plus the exact tested Windows/CUDA lock file |
| Organize repository/ZIP submission | Complete | verified ZIP under `submission/`; exact size, file count, exclusions, and SHA-256 are in `submission/package_summary.json` |

## Integrity notes

- Task 1 scores are from an actual DistilBERT optimization run on the frozen MusicCaps split.
- The first Task 2 attempt was rejected after detecting five exact-audio duplicate groups crossing
  the split. Its checkpoints were moved to `checkpoints/legacy_invalid_gtzan_split_2026-08-21/`.
  The current split keeps all 14 exact duplicate groups in one partition.
- The earlier Task 3 work was rejected because a random split caused extensive artist and exact-song
  overlap. The replacement uses official MTAT shards and retains all usable samples, including clips
  with no positive target tags.
- An initial Task 4 feasibility run used the first recovered test block. Before the final model was
  trained, a different official-test range (records 200--299) was frozen for the final evaluation.
  No unavailable clip was replaced or synthesized.
- Generated report tables use only result JSON files created by completed scripts. Missing values
  remain unavailable rather than being filled with plausible-looking numbers.
