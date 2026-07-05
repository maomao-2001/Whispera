from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_name(value: str, fallback: str) -> str:
    text = str(value or "").strip()
    normalized = _SAFE_NAME_RE.sub("_", text).strip("._")
    return normalized or fallback


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = np.where(clipped < 0, clipped * 32768.0, clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())


@dataclass
class TurnCaptureRecorder:
    root_dir: str | None
    sample_rate: int = 16000
    enabled: bool = False
    session_id: str = "session"

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def set_session_id(self, session_id: str) -> None:
        self.session_id = _safe_name(session_id, "session")

    def capture_user_turn(
        self,
        turn_id: str,
        samples: np.ndarray,
        transcript: str | None,
        *,
        interrupted: bool = False,
        error_message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, str] | None:
        if not self.enabled or not self.root_dir:
            return None

        root = Path(self.root_dir)
        session_dir = root / self.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"{_timestamp_slug()}-{_safe_name(turn_id, 'turn')}"
        wav_path = session_dir / f"{base_name}-user.wav"
        text_path = session_dir / f"{base_name}-user.txt"
        meta_path = session_dir / f"{base_name}-meta.json"

        _write_wav(wav_path, samples, self.sample_rate)
        text_path.write_text(str(transcript or ""), encoding="utf-8")

        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        metadata: dict[str, Any] = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "turn_id": turn_id,
            "sample_rate": self.sample_rate,
            "sample_count": int(audio.size),
            "duration_ms": round((audio.size / float(self.sample_rate)) * 1000.0, 2) if self.sample_rate > 0 else 0.0,
            "interrupted": bool(interrupted),
            "transcript": str(transcript or ""),
            "error_message": error_message,
            "audio_file": wav_path.name,
            "text_file": text_path.name,
        }
        if extra:
            metadata.update(extra)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "audio_path": str(wav_path),
            "text_path": str(text_path),
            "meta_path": str(meta_path),
        }
