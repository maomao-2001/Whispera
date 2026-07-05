from __future__ import annotations

import numpy as np
import librosa
import torch


class Resample:
    def __init__(self, orig_freq: int, new_freq: int):
        self.orig_freq = int(orig_freq)
        self.new_freq = int(new_freq)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.new_freq:
            return waveform

        source = waveform.detach().cpu().numpy()
        if source.ndim == 1:
            source = source[None, :]

        resampled = [
            librosa.resample(channel.astype(np.float32), orig_sr=self.orig_freq, target_sr=self.new_freq)
            for channel in source
        ]
        return torch.from_numpy(np.stack(resampled).astype(np.float32))
