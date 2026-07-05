from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import Any, Optional

import numpy as np


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "ASR audio preprocessing requires 'librosa'. Install project dependencies with `pip install -e .` "
            "or install `librosa` in the active environment."
        ) from exc
    return librosa


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "ASR waveform serialization requires 'soundfile'. Install project dependencies with `pip install -e .` "
            "or install `soundfile` in the active environment."
        ) from exc
    return sf


def _resolve_default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _require_automodel_class():
    try:
        from funasr import AutoModel
    except ImportError as exc:
        raise ImportError(
            "SenseVoice ASR requires 'funasr'. Install project dependencies with `pip install -e .` "
            "or install `funasr` in the active environment."
        ) from exc
    return AutoModel


def _normalize_audio(audio: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        array = array.reshape(-1)
    if array.size == 0:
        return np.zeros(0, dtype=np.float32)
    return np.ascontiguousarray(np.clip(array, -1.0, 1.0), dtype=np.float32)


def normalize_transcript_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if "|>" in cleaned:
        cleaned = cleaned.split("|>")[-1]
    return cleaned.strip()


def extract_raw_text(raw_result: Any) -> str:
    if isinstance(raw_result, list) and raw_result:
        first_item = raw_result[0]
        if isinstance(first_item, dict):
            return str(first_item.get("text", ""))
        return str(first_item)
    if isinstance(raw_result, dict):
        return str(raw_result.get("text", ""))
    if raw_result is None:
        return ""
    return str(raw_result)


def resample_audio(audio: np.ndarray, source_sample_rate: int, target_sample_rate: int) -> np.ndarray:
    audio = _normalize_audio(audio)
    if audio.size == 0:
        return audio
    if int(source_sample_rate) == int(target_sample_rate):
        return audio
    librosa = _require_librosa()
    resampled = librosa.resample(audio, orig_sr=int(source_sample_rate), target_sr=int(target_sample_rate))
    return np.ascontiguousarray(resampled, dtype=np.float32)


def load_audio_for_asr(audio_path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    librosa = _require_librosa()
    audio, _ = librosa.load(str(audio_path), sr=int(sample_rate), mono=True)
    return np.ascontiguousarray(audio, dtype=np.float32)


@dataclass
class SenseVoiceASRConfig:
    model: str = "iic/SenseVoiceSmall"
    target_sample_rate: int = 16000
    device: Optional[str] = None
    disable_update: bool = True
    log_level: str = "ERROR"

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = _resolve_default_device()


@dataclass
class ASRResult:
    text: str
    raw_text: str
    language: str
    sample_rate: int
    target_sample_rate: int
    num_samples: int
    duration_ms: float
    raw_result: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "raw_text": self.raw_text,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "target_sample_rate": self.target_sample_rate,
            "num_samples": self.num_samples,
            "duration_ms": self.duration_ms,
            "raw_result": self.raw_result,
        }


class SenseVoiceASR:
    def __init__(self, config: Optional[SenseVoiceASRConfig] = None):
        self.config = config or SenseVoiceASRConfig()
        self._model = None
        self._model_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self):
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            auto_model_cls = _require_automodel_class()
            self._model = auto_model_cls(
                model=self.config.model,
                disable_update=self.config.disable_update,
                log_level=self.config.log_level,
                device=self.config.device,
            )
            return self._model

    def transcribe_path(
        self,
        audio_path: str | Path,
        language: str = "auto",
        use_itn: bool = True,
        sample_rate: Optional[int] = None,
        target_sample_rate: Optional[int] = None,
    ) -> ASRResult:
        effective_sample_rate = int(sample_rate or self.config.target_sample_rate)
        audio = load_audio_for_asr(audio_path, sample_rate=effective_sample_rate)
        return self.transcribe_audio(
            audio,
            sample_rate=effective_sample_rate,
            language=language,
            use_itn=use_itn,
            target_sample_rate=target_sample_rate,
        )

    def transcribe_audio(
        self,
        audio: np.ndarray | list[float],
        sample_rate: int,
        language: str = "auto",
        use_itn: bool = True,
        target_sample_rate: Optional[int] = None,
    ) -> ASRResult:
        normalized_audio = _normalize_audio(audio)
        if normalized_audio.size == 0:
            raise ValueError("audio must be non-empty")

        effective_target_sample_rate = int(target_sample_rate or self.config.target_sample_rate)

        target_audio = resample_audio(
            normalized_audio,
            source_sample_rate=int(sample_rate),
            target_sample_rate=effective_target_sample_rate,
        )

        temp_path: Optional[Path] = None
        try:
            with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = Path(temp_file.name)

            sf = _require_soundfile()
            sf.write(str(temp_path), target_audio, effective_target_sample_rate)
            raw_result = self.get_model().generate(
                input=str(temp_path),
                language=language,
                use_itn=use_itn,
            )
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

        raw_text = extract_raw_text(raw_result)
        text = normalize_transcript_text(raw_text)
        duration_ms = round((int(normalized_audio.size) / int(sample_rate)) * 1000.0, 2)

        return ASRResult(
            text=text,
            raw_text=raw_text,
            language=language,
            sample_rate=int(sample_rate),
            target_sample_rate=effective_target_sample_rate,
            num_samples=int(normalized_audio.size),
            duration_ms=duration_ms,
            raw_result=raw_result,
        )