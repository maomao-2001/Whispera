from __future__ import annotations

from io import BufferedIOBase
from pathlib import Path
from typing import BinaryIO

import numpy as np
import soundfile as sf
import torch


def _coerce_input(src: str | Path | BinaryIO):
    if isinstance(src, (str, Path)):
        return str(src)
    if isinstance(src, BufferedIOBase) or hasattr(src, "read"):
        if hasattr(src, "seek"):
            src.seek(0)
        return src
    raise TypeError(f"unsupported audio input type: {type(src)!r}")


def load(src: str | Path | BinaryIO):
    data, sample_rate = sf.read(_coerce_input(src), always_2d=True, dtype="float32")
    audio = torch.from_numpy(np.ascontiguousarray(data.T, dtype=np.float32))
    return audio, int(sample_rate)
