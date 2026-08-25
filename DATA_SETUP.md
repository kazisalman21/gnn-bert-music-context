# Dataset setup

Raw audio is not redistributed with the source submission. The datasets must be obtained under
their original research licenses/terms and placed in the layouts below. The preprocessing scripts
check these paths and record unreadable or empty files.

## GTZAN

Expected layout:

```text
data/raw/gtzan/Data/genres_original/
  blues/*.wav
  classical/*.wav
  country/*.wav
  disco/*.wav
  hiphop/*.wav
  jazz/*.wav
  metal/*.wav
  pop/*.wav
  reggae/*.wav
  rock/*.wav
```

The local copy contains 1,000 source WAV files. `jazz.00054.wav` is unreadable, so the canonical
processed set contains 999 tracks. Run `python scripts/preprocess_gtzan.py`; it also hashes every
source WAV and creates the duplicate-safe split.

## MagnaTagATune

Expected layout:

```text
data/raw/magnatagatune/
  annotations_final.csv
  clip_info_final.csv
  0/*.mp3
  1/*.mp3
  ...
  f/*.mp3
```

The `mp3_path` column in `annotations_final.csv` must resolve relative to
`data/raw/magnatagatune/`. Run `python scripts/prepare_splits.py` before
`python scripts/preprocess_mtat.py`. The split follows official archive shards 0--b/c/d--f.

## MusicCaps

Expected metadata:

```text
data/raw/musiccaps/musiccaps-public.csv
```

MusicCaps provides YouTube IDs and ten-second start/end timestamps rather than bundled audio.
The final Task 4 protocol requests training records 0--399, validation records 0--99, and
official-test records 200--299:

```powershell
python scripts/download_musiccaps_audio.py --split train --offset 0 --limit 400
python scripts/download_musiccaps_audio.py --split val --offset 0 --limit 100
python scripts/download_musiccaps_audio.py --split test --offset 200 --limit 100
```

The `--offset` option can resume at another frozen split position, and `--max-seconds` bounds
one throttled source without stopping the whole run. yt-dlp and FFmpeg extract only the
published interval and write a per-ID status report. Successfully recovered WAV files are stored under
`data/raw/musiccaps/audio/`; invalid responses are isolated under
`data/raw/musiccaps/failed_downloads/`. Run `python scripts/preprocess_musiccaps.py`
after downloads finish.

Availability changes over time. Unavailable clips must remain recorded as unavailable; they must
not be replaced with unrelated audio.
