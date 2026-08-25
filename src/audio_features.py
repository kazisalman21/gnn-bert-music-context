"""
Audio feature extraction module.

Handles:
- Resampling to 22,050 Hz
- 128-bin log-mel spectrogram extraction
- 12-bin chroma extraction
- MFCC extraction (20 coefficients)
- Per-track normalization
- Fixed-window and beat-synchronous segmentation
- Feature caching to disk
"""

import numpy as np
import librosa
import json
from pathlib import Path
from typing import Optional


def load_and_resample(audio_path: str, sr: int = 22050) -> np.ndarray:
    """Load audio file and resample to target sample rate."""
    y, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
    return y


def extract_log_mel(y: np.ndarray, sr: int = 22050, n_mels: int = 128,
                    n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Extract 128-bin log-mel spectrogram. Returns (n_mels, T) array."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel


def extract_chroma(y: np.ndarray, sr: int = 22050, n_chroma: int = 12,
                   hop_length: int = 512) -> np.ndarray:
    """Extract 12-bin chroma features. Returns (12, T) array."""
    chroma = librosa.feature.chroma_stft(
        y=y, sr=sr, n_chroma=n_chroma, hop_length=hop_length
    )
    return chroma


def extract_mfcc(y: np.ndarray, sr: int = 22050, n_mfcc: int = 20,
                 hop_length: int = 512) -> np.ndarray:
    """Extract MFCC features. Returns (n_mfcc, T) array."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return mfcc


def normalize_features(features: np.ndarray) -> np.ndarray:
    """Per-track zero-mean unit-variance normalization along time axis."""
    mean = features.mean(axis=-1, keepdims=True)
    std = features.std(axis=-1, keepdims=True)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero
    return (features - mean) / std


def segment_fixed_window(y: np.ndarray, sr: int, window_sec: float) -> list:
    """Split audio into fixed-duration non-overlapping segments.
    
    Returns list of (start_sample, end_sample) tuples.
    """
    window_samples = int(window_sec * sr)
    n_samples = len(y)
    segments = []
    start = 0
    while start < n_samples:
        end = min(start + window_samples, n_samples)
        if end - start >= window_samples // 2:  # keep if at least half a window
            segments.append((start, end))
        start = end
    return segments


def segment_beat_sync(y: np.ndarray, sr: int) -> list:
    """Beat-synchronous segmentation using librosa beat tracker.
    
    Returns list of (start_sample, end_sample) tuples.
    Falls back to fixed 2s windows if beat tracking fails.
    """
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        if len(beat_frames) < 2:
            return segment_fixed_window(y, sr, 2.0)
        beat_samples = librosa.frames_to_samples(beat_frames)
        segments = []
        # Preserve audio before the first detected beat.
        if beat_samples[0] > 0:
            segments.append((0, int(beat_samples[0])))
        for i in range(len(beat_samples) - 1):
            segments.append((int(beat_samples[i]), int(beat_samples[i + 1])))
        # Add final segment from last beat to end
        if beat_samples[-1] < len(y):
            segments.append((beat_samples[-1], len(y)))
        return segments
    except Exception:
        return segment_fixed_window(y, sr, 2.0)


def compute_segment_features(y: np.ndarray, sr: int, segments: list,
                             feature_type: str = "mfcc",
                             n_mfcc: int = 20, n_chroma: int = 12,
                             hop_length: int = 512) -> np.ndarray:
    """Compute mean feature vector for each segment.
    
    Args:
        feature_type: "mfcc", "chroma", or "both"
    
    Returns:
        (num_segments, feature_dim) array
    """
    features_list = []
    for start, end in segments:
        segment_audio = y[start:end]
        if len(segment_audio) < hop_length:
            # Segment too short, pad
            segment_audio = np.pad(segment_audio, (0, hop_length - len(segment_audio)))
        
        if feature_type == "mfcc":
            feat = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=n_mfcc,
                                         hop_length=hop_length)
            feat_mean = feat.mean(axis=-1)  # (n_mfcc,)
        elif feature_type == "chroma":
            feat = librosa.feature.chroma_stft(y=segment_audio, sr=sr,
                                                n_chroma=n_chroma,
                                                hop_length=hop_length)
            feat_mean = feat.mean(axis=-1)  # (12,)
        elif feature_type == "both":
            mfcc = librosa.feature.mfcc(y=segment_audio, sr=sr, n_mfcc=n_mfcc,
                                         hop_length=hop_length)
            chroma = librosa.feature.chroma_stft(y=segment_audio, sr=sr,
                                                  n_chroma=n_chroma,
                                                  hop_length=hop_length)
            feat_mean = np.concatenate([mfcc.mean(axis=-1), chroma.mean(axis=-1)])
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
        features_list.append(feat_mean)
    
    return np.stack(features_list, axis=0)


def compute_rich_segment_features(y: np.ndarray, sr: int = 22050,
                                  window_sec: float = 5.0,
                                  hop_length: int = 512,
                                  n_fft: int = 2048,
                                  normalize: bool = True) -> np.ndarray:
    """Extract the documented 77-dimensional feature vector per segment.

    The vector contains MFCC mean/std (40), chroma mean/std (24), spectral
    contrast mean (7), spectral centroid mean/std (2), zero-crossing-rate
    mean/std (2), and RMS mean/std (2). A final partial window is retained
    when it contains at least half of the requested window, matching
    :func:`segment_fixed_window`.
    """
    segments = segment_fixed_window(y, sr, window_sec)
    feature_rows = []

    for start, end in segments:
        segment_audio = y[start:end]
        if len(segment_audio) < n_fft:
            segment_audio = np.pad(segment_audio, (0, n_fft - len(segment_audio)))

        mfcc = librosa.feature.mfcc(
            y=segment_audio, sr=sr, n_mfcc=20,
            n_fft=n_fft, hop_length=hop_length,
        )
        chroma = librosa.feature.chroma_stft(
            y=segment_audio, sr=sr, n_chroma=12,
            n_fft=n_fft, hop_length=hop_length,
        )
        contrast = librosa.feature.spectral_contrast(
            y=segment_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        )
        centroid = librosa.feature.spectral_centroid(
            y=segment_audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        )[0]
        zcr = librosa.feature.zero_crossing_rate(
            segment_audio, frame_length=n_fft, hop_length=hop_length,
        )[0]
        rms = librosa.feature.rms(
            y=segment_audio, frame_length=n_fft, hop_length=hop_length,
        )[0]

        row = np.concatenate([
            mfcc.mean(axis=1), mfcc.std(axis=1),
            chroma.mean(axis=1), chroma.std(axis=1),
            contrast.mean(axis=1),
            np.array([centroid.mean(), centroid.std()]),
            np.array([zcr.mean(), zcr.std()]),
            np.array([rms.mean(), rms.std()]),
        ]).astype(np.float32)
        feature_rows.append(row)

    if not feature_rows:
        raise ValueError("Audio produced no valid segments")

    features = np.stack(feature_rows, axis=0)
    if features.shape[1] != 77:
        raise RuntimeError(f"Expected 77 rich features, got {features.shape[1]}")

    if normalize:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        features = (features - mean) / std

    return features.astype(np.float32)


def process_track(audio_path: str, sr: int = 22050,
                  window_sec: float = 5.0, use_beat_sync: bool = False,
                  feature_type: str = "mfcc", n_mfcc: int = 20,
                  n_chroma: int = 12, hop_length: int = 512,
                  normalize: bool = True) -> dict:
    """Full preprocessing pipeline for a single track.
    
    Returns dict with:
        - 'log_mel': (128, T) log-mel spectrogram
        - 'segment_features': (num_segments, feat_dim) segment features
        - 'segments': list of (start_sample, end_sample)
        - 'duration_sec': float
        - 'num_segments': int
        - 'sr': int
        - 'feature_type': str
        - 'window_sec': float
    """
    y = load_and_resample(audio_path, sr)
    
    # Full-track log-mel for CNN baseline
    log_mel = extract_log_mel(y, sr)
    if normalize:
        log_mel = normalize_features(log_mel)
    
    # Segmentation
    if use_beat_sync:
        segments = segment_beat_sync(y, sr)
    else:
        segments = segment_fixed_window(y, sr, window_sec)
    
    # Segment features for graph nodes
    seg_features = compute_segment_features(
        y, sr, segments, feature_type=feature_type,
        n_mfcc=n_mfcc, n_chroma=n_chroma, hop_length=hop_length
    )
    if normalize:
        mean = seg_features.mean(axis=0, keepdims=True)
        std = seg_features.std(axis=0, keepdims=True)
        std = np.where(std == 0, 1.0, std)
        seg_features = (seg_features - mean) / std
    
    return {
        'log_mel': log_mel,
        'segment_features': seg_features,
        'segments': segments,
        'duration_sec': len(y) / sr,
        'num_segments': len(segments),
        'sr': sr,
        'feature_type': feature_type,
        'window_sec': window_sec,
    }
