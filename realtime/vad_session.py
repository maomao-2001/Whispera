from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from time import perf_counter

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


@dataclass
class VADTurnTiming:
    speech_started_at: float | None = None
    last_speech_at: float | None = None
    speech_end_at: float | None = None
    speech_finished_at_est: float | None = None
    vad_tail_ms: float | None = None
    speech_duration_ms: float | None = None
    buffered_audio_ms: float | None = None


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
        self._speech_started_at: float | None = None
        self._last_speech_at: float | None = None
        self._completed_turn_timing: VADTurnTiming | None = None

    def push_chunk(self, chunk: np.ndarray) -> str:
        window = self.config.chunk_window
        for i in range(0, max(len(chunk), 1), window):
            frame = chunk[i : i + window]
            if len(frame) < window:
                frame = np.pad(frame, (0, window - len(frame)))
            prob = self.vad(frame, self.config.sample_rate)
            frame_processed_at = perf_counter()
            if prob > self.config.threshold:
                self.silence_samples = 0
                self.tail_silence = 0
                if self.speech_samples == 0:
                    self._speech_started_at = frame_processed_at
                self._last_speech_at = frame_processed_at
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
                    buffered_samples = sum(len(item) for item in self.buffer)
                    vad_tail_ms = (self.silence_samples / self.config.sample_rate) * 1000.0
                    speech_duration_ms = None
                    if self._speech_started_at is not None and self._last_speech_at is not None:
                        speech_duration_ms = (self._last_speech_at - self._speech_started_at) * 1000.0
                    self._completed_turn_timing = VADTurnTiming(
                        speech_started_at=self._speech_started_at,
                        last_speech_at=self._last_speech_at,
                        speech_end_at=frame_processed_at,
                        speech_finished_at_est=frame_processed_at - (vad_tail_ms / 1000.0),
                        vad_tail_ms=vad_tail_ms,
                        speech_duration_ms=speech_duration_ms,
                        buffered_audio_ms=(buffered_samples / self.config.sample_rate) * 1000.0,
                    )
                    self.speaking = False
                    self.speech_samples = 0
                    self.silence_samples = 0
                    self.tail_silence = 0
                    self._speech_started_at = None
                    self._last_speech_at = None
                    return "speech_end"
            else:
                if self.speech_samples > 0:
                    self.buffer.clear()
                self.speech_samples = 0
                self._speech_started_at = None
                self._last_speech_at = None
                self.ring.append(frame)
        return "listening"

    def get_audio(self) -> np.ndarray:
        audio = np.concatenate(self.buffer) if self.buffer else np.array([], dtype=np.float32)
        self.buffer.clear()
        return audio

    def consume_completed_timing(self) -> VADTurnTiming | None:
        timing = self._completed_turn_timing
        self._completed_turn_timing = None
        return timing
