"""Audio loading and framing helpers."""

from __future__ import annotations

import audioop
import struct
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype in (np.float32, np.float64):
        return audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        if max_value <= 0:
            raise ValueError(f"Unsupported integer dtype: {audio.dtype}")
        return audio.astype(np.float32) / float(max_value)
    raise ValueError(f"Unsupported audio dtype: {audio.dtype}")


def load_audio_mono(
    audio_path: str, *, target_sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 data in [-1, 1]."""
    path = Path(audio_path)
    try:
        sample_rate, audio = wavfile.read(path)
    except ValueError as exc:
        # scipy.io.wavfile does not support some companded WAV formats (ALAW/ULAW).
        try:
            sample_rate, audio = _load_companded_wav(path)
        except ValueError as fallback_exc:
            raise exc from fallback_exc
    audio = _to_float32(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        factor = gcd(sample_rate, target_sample_rate)
        up = target_sample_rate // factor
        down = sample_rate // factor
        audio = resample_poly(audio, up, down).astype(np.float32)
        sample_rate = target_sample_rate
    return audio, sample_rate


def _load_companded_wav(audio_path: Path) -> tuple[int, np.ndarray]:
    blob = audio_path.read_bytes()
    if len(blob) < 12 or blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise ValueError("Not a RIFF/WAVE file.")

    offset = 12
    format_tag: int | None = None
    channels: int | None = None
    sample_rate: int | None = None
    bits_per_sample: int | None = None
    companded_payload: bytes | None = None

    while offset + 8 <= len(blob):
        chunk_id = blob[offset : offset + 4]
        chunk_size = struct.unpack("<I", blob[offset + 4 : offset + 8])[0]
        data_start = offset + 8
        data_end = data_start + chunk_size
        if data_end > len(blob):
            break
        chunk = blob[data_start:data_end]
        offset = data_end + (chunk_size % 2)

        if chunk_id == b"fmt " and len(chunk) >= 16:
            format_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack(
                "<HHIIHH", chunk[:16]
            )
        elif chunk_id == b"data":
            companded_payload = chunk

    if format_tag not in (6, 7):
        raise ValueError(f"Unsupported companded WAV format tag: {format_tag}.")
    if channels is None or sample_rate is None or bits_per_sample is None:
        raise ValueError("Missing required WAV fmt metadata.")
    if companded_payload is None:
        raise ValueError("Missing WAV data chunk.")
    if bits_per_sample != 8:
        raise ValueError(f"Unsupported companded sample bit depth {bits_per_sample}. Expected 8.")

    if format_tag == 6:
        pcm_bytes = audioop.alaw2lin(companded_payload, 2)
    else:
        pcm_bytes = audioop.ulaw2lin(companded_payload, 2)

    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return sample_rate, audio


def frame_audio(audio: np.ndarray, *, frame_size: int, hop_size: int) -> np.ndarray:
    """Split a 1D signal into overlapping frames with zero-padding on tail."""
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be > 0")
    if audio.size == 0:
        return np.zeros((0, frame_size), dtype=np.float32)

    frame_count = 1 + max(0, int(np.ceil((audio.size - frame_size) / hop_size)))
    total_size = (frame_count - 1) * hop_size + frame_size
    if total_size > audio.size:
        audio = np.pad(audio, (0, total_size - audio.size))

    frames = np.empty((frame_count, frame_size), dtype=np.float32)
    for idx in range(frame_count):
        start = idx * hop_size
        frames[idx] = audio[start : start + frame_size]
    return frames
