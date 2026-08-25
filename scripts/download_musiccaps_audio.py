"""Download exact MusicCaps audio intervals with a resumable status log.

MusicCaps distributes metadata rather than audio. This script asks yt-dlp for the
public source video and lets FFmpeg extract the [start_s, end_s] interval. Missing,
private, removed, and blocked videos are recorded instead of being substituted.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import librosa
import pandas as pd
import psutil
import yt_dlp
from yt_dlp.utils import download_range_func


ROOT = Path(__file__).resolve().parents[1]
METADATA_PATH = ROOT / "data/raw/musiccaps/musiccaps-public.csv"
SPLIT_DIR = ROOT / "data/splits"
AUDIO_DIR = ROOT / "data/raw/musiccaps/audio"
FAILED_DIR = ROOT / "data/raw/musiccaps/failed_downloads"
STATUS_PATH = ROOT / "results/musiccaps_audio_download.json"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)


def usable_wav(path: Path, expected_seconds: float) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        duration = float(librosa.get_duration(path=str(path)))
        return duration >= max(1.0, expected_seconds - 1.0)
    except Exception:
        return False


def write_status(payload: dict) -> None:
    payload["summary"] = {
        name: sum(item["status"] == name for item in payload["clips"].values())
        for name in ("downloaded", "cached", "unavailable", "failed")
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def classify_error(message: str) -> str:
    lowered = message.lower()
    unavailable_markers = (
        "video unavailable", "private video", "has been removed", "not available",
        "copyright", "members-only", "sign in to confirm your age",
    )
    return "unavailable" if any(marker in lowered for marker in unavailable_markers) else "failed"


def download_one(ytid: str, start_s: float, end_s: float,
                 max_seconds: int) -> dict:
    expected_seconds = end_s - start_s
    wav_path = AUDIO_DIR / f"{ytid}.wav"
    if usable_wav(wav_path, expected_seconds):
        return {"status": "cached", "path": str(wav_path.relative_to(ROOT)).replace("\\", "/")}

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / f"{ytid}.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 2,
        "fragment_retries": 2,
        "socket_timeout": 20,
        "download_ranges": download_range_func(None, [(start_s, end_s)]),
        "force_keyframes_at_cuts": True,
        # Prevent one slow or dead media stream from blocking a long resumable run.
        "external_downloader_args": {
            "ffmpeg_i": ["-rw_timeout", "30000000"],
        },
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
    }
    started = time.time()
    timed_out = threading.Event()

    def stop_slow_download() -> None:
        timed_out.set()
        try:
            children = psutil.Process().children(recursive=True)
            for child in children:
                child.terminate()
            _, alive = psutil.wait_procs(children, timeout=3)
            for child in alive:
                child.kill()
        except psutil.Error:
            pass

    timer = threading.Timer(max_seconds, stop_slow_download)
    timer.daemon = True
    timer.start()
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            downloader.download([f"https://www.youtube.com/watch?v={ytid}"])
        if timed_out.is_set():
            raise TimeoutError(
                f"download exceeded the {max_seconds}-second per-clip limit"
            )
        if not usable_wav(wav_path, expected_seconds):
            raise RuntimeError("yt-dlp completed but the expected WAV interval is missing or too short")
        return {
            "status": "downloaded",
            "path": str(wav_path.relative_to(ROOT)).replace("\\", "/"),
            "elapsed_seconds": time.time() - started,
        }
    except Exception as exc:
        if timed_out.is_set():
            message = (
                f"TimeoutError: download exceeded the "
                f"{max_seconds}-second per-clip limit"
            )
        else:
            message = f"{type(exc).__name__}: {exc}"
        # yt-dlp can leave a tiny HTML/error response with an audio extension.
        # Preserve it for diagnosis, but keep it outside the usable audio folder.
        for leftover in AUDIO_DIR.glob(f"{ytid}.*"):
            if leftover != wav_path:
                destination = FAILED_DIR / leftover.name
                if destination.exists():
                    destination = FAILED_DIR / f"{leftover.stem}.{int(time.time())}{leftover.suffix}"
                shutil.move(str(leftover), str(destination))
        return {
            "status": classify_error(message),
            "error": message[:1000],
            "elapsed_seconds": time.time() - started,
        }
    finally:
        timer.cancel()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many split records first")
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to attempt in this invocation")
    parser.add_argument(
        "--max-seconds", type=int, default=90,
        help="Wall-clock limit for one non-cached source clip",
    )
    args = parser.parse_args()
    if args.max_seconds < 1:
        parser.error("--max-seconds must be positive")

    metadata = pd.read_csv(METADATA_PATH).set_index("ytid", drop=False)
    if args.split == "all":
        ids = metadata["ytid"].astype(str).tolist()
    else:
        ids = json.loads((SPLIT_DIR / f"musiccaps_{args.split}.json").read_text(encoding="utf-8"))
    ids = ids[max(0, args.offset):]
    if args.limit is not None:
        ids = ids[:max(0, args.limit)]

    if STATUS_PATH.exists():
        payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    else:
        payload = {
            "source": "MusicCaps public YouTube IDs and start/end timestamps",
            "audio_redistribution": False,
            "clips": {},
        }

    for position, ytid in enumerate(ids, start=1):
        row = metadata.loc[str(ytid)]
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        record = download_one(
            str(ytid), start_s, end_s, max_seconds=args.max_seconds
        )
        record.update({"start_s": start_s, "end_s": end_s, "requested_split": args.split})
        payload["clips"][str(ytid)] = record
        write_status(payload)
        print(
            f"[{position}/{len(ids)}] {ytid}: {record['status']} "
            f"({record.get('elapsed_seconds', 0):.1f}s)",
            flush=True,
        )
    print(json.dumps(payload["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
