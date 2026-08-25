# GNN-Based BERT for Understanding Context from Music

This repository contains our CSE425 Neural Networks project. The project studies whether a
Graph Neural Network (GNN) representation of audio structure can be combined with a BERT
text representation to improve music-context prediction.

## Project group

| Member | Student ID | Section |
|---|---:|---:|
| Kazi Salman Salim | 23101209 | 3 |
| Miskatul Afrin Anika | 23101409 | 3 |

**Course:** CSE425 — Neural Networks  
**Course Instructor:** Moin Mostakim

The project brief is available in `CSE425_Project_GNN_BERT_Music_Context.pdf`. The official
course outline is preserved as `Course_Content.pdf`; the report maps the implementation to its
CNN, transformer, transfer-learning, optimization, and explainability topics.

The implementation follows three main tasks and one optional advanced task:

1. **Task 1:** DistilBERT caption-to-tag classification on MusicCaps.
2. **Task 2:** GraphSAGE genre classification on GTZAN, compared with a mel-spectrogram CNN.
3. **Task 3:** GNN-BERT fusion on MagnaTagATune with BERT-only, GNN-only, concatenation,
   and cross-attention ablations.
4. **Task 4:** Contrastive MusicCaps audio-text retrieval with bidirectional Recall@K,
   ten top-three examples, and zero-shot tag transfer.

## Project structure

```text
CSE425/
  README.md
  DATA_SETUP.md
  PROJECT_INFO.md
  requirements.txt
  config.yaml
  implementation_plan.md
  data/
    raw/                 downloaded datasets (not committed)
    processed/           graphs and cached mel spectrograms (not committed)
    splits/              frozen train/validation/test assignments
  checkpoints/           trained model checkpoints (not committed)
  notebooks/
    eda.ipynb
    demo_context.ipynb
  plots/                 generated figures used in the report
  report/
    final_report.tex
    final_report.pdf
  results/               metrics, tables, audits, and training logs
  retrieval_examples/    Task 4 and qualitative case-study outputs
  scripts/               preprocessing, training, and evaluation entry points
  src/                   reusable datasets, models, metrics, and plotting code
  tests/                 preprocessing and model smoke tests
```

Old pre-audit artifacts are kept in folders named `legacy_pre_audit_2026-08-21`. They are
not final results and are excluded by `.gitignore`.

## Environment

The tested environment uses Python 3.12, PyTorch with CUDA 12.6, PyTorch Geometric,
Transformers, and librosa. On the development machine, training uses an NVIDIA GeForce
RTX 3060. All scripts fall back to CPU where practical, although BERT training is much
slower on CPU.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducing the experiments

Run commands from the repository root in this order:

```powershell
python scripts/prepare_splits.py
python scripts/preprocess_gtzan.py
python scripts/preprocess_mtat.py
python scripts/download_musiccaps_audio.py --split train --offset 0 --limit 400
python scripts/download_musiccaps_audio.py --split val --offset 0 --limit 100
python scripts/download_musiccaps_audio.py --split test --offset 200 --limit 100
python scripts/preprocess_musiccaps.py
python scripts/run_baselines.py
python scripts/run_task1.py
python scripts/run_task2.py
python scripts/run_task3.py
python scripts/run_task4.py
python scripts/generate_task3_case_studies.py
python scripts/generate_musiccaps_case_studies.py
python scripts/generate_architecture_figure.py
python scripts/generate_report_assets.py
python -m pytest tests/test_core.py -q
python scripts/validate_project.py
python scripts/create_submission.py
```

The split script must run before preprocessing or training. The training scripts select
checkpoints using validation metrics and evaluate the test split only after model selection.
Classification thresholds for multi-label tasks are chosen using validation data.

## Split and leakage policy

- **MusicCaps:** rows marked `is_audioset_eval=True` form the test set. The remaining rows
  are split 90/10 into training and validation sets using seed 42. The Task 1 vocabulary is
  built only from training tags with at least 20 occurrences.
- **GTZAN:** a seeded per-genre group split is used after excluding the unreadable track.
  SHA-256-identical audio files are kept in the same partition; all 14 exact duplicate groups
  have zero cross-split overlap.
- **MagnaTagATune:** archive shards `0-b` are training, `c` is validation, and `d-f` are
  test. Clip, artist, and exact `(artist, title, album)` overlap are reported in
  `results/data_split_audit.json`.

For Task 3, 31 instrument/vocal tags are used only to form the BERT input sentence. The
19 genre/mood target tags are lexically disjoint from those context tags. Both groups still
come from the same annotation corpus, so indirect statistical relationships remain and are
reported as a limitation rather than described as fully leakage-free.

## Generated evidence

The scripts produce the following faculty-requested evidence:

- per-epoch Macro-F1 and Micro-F1 histories;
- test Macro-F1, Micro-F1, and mean AUC-PR;
- Task 2 confusion-matrix heatmaps and GNN-vs-CNN comparison;
- Task 3 four-way ablation tables and per-tag metrics;
- genre and mood t-SNE plots using ground-truth labels only;
- three graph/text alignment case studies when MusicCaps audio is available;
- bidirectional retrieval R@1, R@5, and R@10, ten top-three examples, and zero-shot
  tag transfer for Task 4.

All reported numbers are written by the evaluation scripts from saved checkpoints. No result
is manually edited into the final metrics files.

## Dataset note

Raw audio and generated tensors are large and are intentionally excluded from Git. Anyone
reproducing the project must obtain the datasets from their official sources and preserve
their licenses and terms of use. MusicCaps contains YouTube identifiers rather than bundled
audio, so some source clips may be unavailable.

## Faculty submission

The final 6--10 page report is `report/final_report.pdf`. The verified source-and-results package is
`submission/CSE425_GNN_BERT_Kazi_Salman_Salim_Miskatul_Afrin_Anika.zip`. It contains source code,
notebooks, split files, real result JSON files, all report plots, the final report, small
checkpoints, the self-contained demo asset, and representative real graph tensors.

Large Transformer checkpoints and raw/processed datasets are kept in the working project but
excluded from the ZIP. Exact file count, archive size, excluded checkpoints, and the archive
SHA-256 are recorded beside it in `submission/package_summary.json`. Inside the ZIP,
`SUBMISSION_NOTES.txt` explains the scope and `SHA256SUMS.txt` verifies every project file.
