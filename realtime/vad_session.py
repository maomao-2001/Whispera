from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


class SileroVAD:
    """Small ONNXRuntime wrapper around the bundled Silero VAD model."""

    def __init__(self, path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self.h = np.zeros((2, 1, 64), dtype=np.float32)
        self.c = np.zeros((2, 1, 64), dtype=np.float32)

    def reset(self) -> None:
        self.h[:] = 0
        self.c[:] = 0

    def __call__(self, chunk: np.ndarray, sample_rate: int = 16000) -> float:
        out, self.h, self.c = self.session.run(
            None,
            {
                "input": chunk.reshape(1, -1).astype(np.float32),
                "h": self.h,
                "c": self.c,
                "sr": np.array(sample_rate, dtype="int64"),
            },
        )
        return float(out[0][0])


@dataclass
class VADConfig:
    sample_rate: int = 16000
    threshold: float = 0.8
    min_speech_ms: int = 128
    min_silence_ms: int = 800
    chunk_window: int = 1024
    preroll_ms: int = 1000


class RealtimeSession:
    """Voice turn state machine used by the realtime assistant backend."""

    def __init__(self, vad_path: str, config: VADConfig | None = None):
        self.config = config or VADConfig()
        self.vad = SileroVAD(vad_path)
        self.min_speech = int(self.config.sample_rate * self.config.min_speech_ms / 1000)
        self.min_silence = int(self.config.sample_rate * self.config.min_silence_ms / 1000)
        self.preroll_frames = max(
            1,
            math.ceil((self.config.sample_rate * self.config.preroll_ms / 1000) / self.config.chunk_window),
        )
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        self.buffer: list[np.ndarray] = []
        self.ring: deque[np.ndarray] = deque(maxlen=self.preroll_frames)
        self.speaking = False
        self.generating = False
        self.interrupt = False
        self.speech_samples = 0
        self.silence_samples = 0
        self.tail_silence = 0

    def push_chunk(self, chunk: np.ndarray) -> str:
        window = self.config.chunk_window
        for i in range(0, max(len(chunk), 1), window):
            frame = chunk[i : i + window]
            if len(frame) < window:
                frame = np.pad(frame, (0, window - len(frame)))
            prob = self.vad(frame, self.config.sample_rate)
            if prob > self.config.threshold:
                self.silence_samples = 0
                self.tail_silence = 0
                self.speech_samples += len(frame)
                self.buffer.append(frame)
                if self.speech_samples >= self.min_speech and not self.speaking:
                    self.speaking = True
                    self.buffer = list(self.ring) + self.buffer
                    self.ring.clear()
                if self.generating and self.speaking:
                    self.interrupt = True
                    return "interrupt"
            elif self.speaking:
                self.silence_samples += len(frame)
                self.tail_silence += 1
                self.buffer.append(frame)
                if self.silence_samples >= self.min_silence:
                    if self.tail_silence > 1:
                        del self.buffer[-(self.tail_silence - 1) :]
                    self.speaking = False
                    self.speech_samples = 0
                    self.silence_samples = 0
                    self.tail_silence = 0
                    return "speech_end"
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self.ring.append(frame)
        return "listening"

    def get_audio(self) -> np.ndarray:
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio
