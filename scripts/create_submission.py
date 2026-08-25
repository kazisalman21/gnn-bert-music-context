"""Create a clean faculty-submission ZIP from verified project artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "submission"
ARCHIVE_NAME = "CSE425_GNN_BERT_Kazi_Salman_Salim_Miskatul_Afrin_Anika.zip"
ARCHIVE_ROOT = "CSE425_GNN_BERT_Project"

ROOT_FILES = [
    ".gitignore",
    "README.md",
    "PROJECT_INFO.md",
    "DATA_SETUP.md",
    "FACULTY_COMPLIANCE.md",
    "implementation_plan.md",
    "config.yaml",
    "requirements.txt",
    "requirements-windows-cu126-lock.txt",
    "CSE425_Project_GNN_BERT_Music_Context.pdf",
    "Course_Content.pdf",
]
SOURCE_DIRECTORIES = ["src", "tests"]
ARTIFACT_DIRECTORIES = [
    "notebooks", "plots", "data/splits", "retrieval_examples",
]
SCRIPT_FILES = [
    "scripts/create_submission.py",
    "scripts/download_musiccaps_audio.py",
    "scripts/generate_architecture_figure.py",
    "scripts/generate_musiccaps_case_studies.py",
    "scripts/generate_report_assets.py",
    "scripts/generate_task3_case_studies.py",
    "scripts/preprocess_gtzan.py",
    "scripts/preprocess_mtat.py",
    "scripts/preprocess_musiccaps.py",
    "scripts/prepare_splits.py",
    "scripts/regenerate_plots.py",
    "scripts/run_baselines.py",
    "scripts/run_task1.py",
    "scripts/run_task2.py",
    "scripts/run_task3.py",
    "scripts/run_task4.py",
    "scripts/validate_project.py",
]
REPORT_FILES = [
    "report/final_report.pdf",
    "report/final_report.tex",
    "report/generated_results.tex",
]
SMALL_CHECKPOINTS = [
    "checkpoints/task2_gnn_best.pt",
    "checkpoints/task2_cnn_best.pt",
    "checkpoints/task3_gnn_only_best.pt",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def add_path(files: dict[str, Path], path: Path, archive_name: str | None = None) -> None:
    if path.is_file():
        relative = archive_name or path.relative_to(ROOT).as_posix()
        files[f"{ARCHIVE_ROOT}/{relative}"] = path


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for relative in ROOT_FILES + REPORT_FILES + SMALL_CHECKPOINTS + SCRIPT_FILES:
        add_path(files, ROOT / relative)

    for directory in SOURCE_DIRECTORIES + ARTIFACT_DIRECTORIES:
        base = ROOT / directory
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                add_path(files, path)

    results_dir = ROOT / "results"
    for path in sorted(results_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".json", ".csv", ".md", ".npz"}:
            add_path(files, path)

    sample_sets = {
        "gtzan": sorted((ROOT / "data/processed/gtzan_graphs").glob("*.pt"))[:20],
        "mtat": sorted((ROOT / "data/processed/mtat_graphs").glob("*.pt"))[:20],
        "musiccaps": sorted((ROOT / "data/processed/musiccaps_graphs").glob("*.pt")),
    }
    for dataset, paths in sample_sets.items():
        for path in paths:
            add_path(files, path, f"graph_samples/{dataset}/{path.name}")

    # The demo notebook uses one real graph and the small Task 2 checkpoint.
    gtzan_splits = json.loads(
        (ROOT / "data/splits/gtzan_splits.json").read_text(encoding="utf-8")
    )
    demo_graph = ROOT / "data/processed/gtzan_graphs" / f"{gtzan_splits['test'][0]}.pt"
    add_path(files, demo_graph, "demo_assets/gtzan_demo_graph.pt")

    forbidden = [
        name for name in files
        if "legacy_pre_audit" in name.lower() or "/$latextemp/" in name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"Forbidden legacy/temp artifacts selected: {forbidden}")
    missing_scripts = [name for name in SCRIPT_FILES if not (ROOT / name).is_file()]
    if missing_scripts:
        raise FileNotFoundError(f"Missing active scripts: {missing_scripts}")
    return files


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = OUTPUT_DIR / ARCHIVE_NAME
    files = collect_files()
    if len([name for name in files if "/graph_samples/" in name]) < 20:
        raise RuntimeError("At least 20 real graph samples are required in the package")

    hashes = []
    total_bytes = 0
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for archive_name, source_path in sorted(files.items()):
            data = source_path.read_bytes()
            archive.writestr(archive_name, data)
            hashes.append(f"{sha256_bytes(data)}  {archive_name}")
            total_bytes += len(data)

        notes = [
            "CSE425 Neural Networks project submission",
            "Members: Kazi Salman Salim (23101209) and Miskatul Afrin Anika (23101409)",
            "Section: 3",
            "Instructor: Moin Mostakim",
            "",
            "Included: source code, notebooks, split files, real results, plots, final report,",
            "small audio-model checkpoints, Task 4 retrieval examples, and representative",
            "real graph tensors. The packaged demo uses its included graph and checkpoint.",
            "",
            "Excluded: raw licensed datasets, generated full processed datasets, .venv, legacy",
            "artifacts, logs, and BERT-based checkpoints larger than 250 MB each.",
            "The excluded checkpoints remain in the working project and are reproducible with",
            "the documented scripts and frozen splits.",
        ]
        archive.writestr(f"{ARCHIVE_ROOT}/SUBMISSION_NOTES.txt", "\n".join(notes) + "\n")
        archive.writestr(f"{ARCHIVE_ROOT}/SHA256SUMS.txt", "\n".join(hashes) + "\n")

    summary = {
        "archive": str(archive_path.relative_to(ROOT)).replace("\\", "/"),
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "included_project_files": len(files),
        "uncompressed_project_bytes": total_bytes,
        "archive_bytes": archive_path.stat().st_size,
        "graph_samples": {
            "gtzan": sum("/graph_samples/gtzan/" in name for name in files),
            "mtat": sum("/graph_samples/mtat/" in name for name in files),
            "musiccaps": sum("/graph_samples/musiccaps/" in name for name in files),
        },
        "large_checkpoints_excluded": [
            "task1_best.pt",
            "task3_bert_only_best.pt",
            "task3_concat_best.pt",
            "task3_cross_attention_best.pt",
            "task4_contrastive_best.pt",
        ],
    }
    (OUTPUT_DIR / "package_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
