from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .local_modules import REPO_ROOT, ensure_repo_root_on_path


@dataclass
class ASRRuntimeConfig:
    model_path: str = str(REPO_ROOT / "model" / "SenseVoiceSmall")
    device: str = "cuda"
    language: str = "auto"
    use_itn: bool = True
    strip_emotion: bool = True


_EMOTION_TAG_RE = re.compile(r"<\|[^|]+?\|>")
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emotion_markup(text: str) -> str:
    cleaned = _EMOTION_TAG_RE.sub("", text or "")
    cleaned = _EMOJI_RE.sub("", cleaned)
    return cleaned.strip()


class FunAsrService:
    def __init__(self, config: ASRRuntimeConfig | None = None):
        self.config = config or ASRRuntimeConfig()
        self._model = None
        self._postprocess = None
        self._warmup_result: dict[str, float | bool] | None = None

    @property
    def is_warmed(self) -> bool:
        return self._warmup_result is not None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        ensure_repo_root_on_path()
        from funasr import AutoModel
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        self._model = AutoModel(
            model=self.config.model_path,
            trust_remote_code=True,
            device=self.config.device,
            disable_update=True,
        )
        self._postprocess = rich_transcription_postprocess

    def transcribe(self, samples: np.ndarray) -> str:
        self._ensure_loaded()
        result = self._model.generate(
            input=samples,
            cache={},
            language=self.config.language,
            use_itn=self.config.use_itn,
        )
        if not result:
            return ""
        text = str(self._postprocess(result[0]["text"])).strip()
        if self.config.strip_emotion:
            text = _strip_emotion_markup(text)
        return text

    def warmup(self, force: bool = False) -> dict[str, float | bool]:
        if self._warmup_result is not None and not force:
            return {
                **self._warmup_result,
                "cached": True,
            }

        total_started_at = perf_counter()
        model_loaded_before = self._model is not None

        load_started_at = perf_counter()
        self._ensure_loaded()
        asr_model_load_ms = (perf_counter() - load_started_at) * 1000.0
        if model_loaded_before:
            asr_model_load_ms = 0.0

        probe_audio = np.zeros(int(0.32 * 16000), dtype=np.float32)
        inference_started_at = perf_counter()
        self.transcribe(probe_audio)
        asr_inference_ms = (perf_counter() - inference_started_at) * 1000.0

        result = {
            "cached": False,
            "asr_model_load_ms": round(asr_model_load_ms, 2),
            "asr_inference_ms": round(asr_inference_ms, 2),
            "total_ms": round((perf_counter() - total_started_at) * 1000.0, 2),
        }
        self._warmup_result = result
        return result
