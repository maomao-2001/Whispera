from __future__ import annotations

import math

import librosa
import numpy as np
import torch


_WINDOWS = {
    "hamming": "hamming",
    "hann": "hann",
    "povey": "hann",
    "rectangular": "boxcar",
}


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << math.ceil(math.log2(value))


def _normalize_waveform(waveform: torch.Tensor, dither: float) -> np.ndarray:
    audio = waveform.detach().cpu().float().reshape(-1)
    if dither > 0:
        audio = audio + torch.randn_like(audio) * float(dither)
    if torch.max(torch.abs(audio)).item() > 2.0:
        audio = audio / 32768.0
    return np.ascontiguousarray(audio.numpy(), dtype=np.float32)


def fbank(
    waveform: torch.Tensor,
    num_mel_bins: int = 80,
    frame_length: float = 25.0,
    frame_shift: float = 10.0,
    dither: float = 0.0,
    energy_floor: float = 0.0,
    window_type: str = "hamming",
    sample_frequency: float = 16000.0,
    snip_edges: bool = True,
    **_: object,
) -> torch.Tensor:
    sample_rate = int(sample_frequency)
    win_length = max(int(round(sample_rate * float(frame_length) / 1000.0)), 1)
    hop_length = max(int(round(sample_rate * float(frame_shift) / 1000.0)), 1)
    n_fft = _next_power_of_two(win_length)
    center = not bool(snip_edges)

    audio = _normalize_waveform(waveform, dither=float(dither))
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=_WINDOWS.get(str(window_type).lower(), "hamming"),
        center=center,
        power=2.0,
        n_mels=int(num_mel_bins),
        htk=True,
        norm=None,
    )
    mel = np.maximum(mel, float(energy_floor) if energy_floor > 0 else 1e-10)
    return torch.from_numpy(np.log(mel + 1e-10).T.astype(np.float32))
