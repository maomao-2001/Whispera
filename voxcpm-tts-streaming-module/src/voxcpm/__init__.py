from typing import TYPE_CHECKING

from .session_protocol import RequestState, SessionState, SessionStore

if TYPE_CHECKING:
    from .asr import ASRResult, SenseVoiceASR, SenseVoiceASRConfig, load_audio_for_asr, normalize_transcript_text
    from .core import VoxCPM

__all__ = [
    "VoxCPM",
    "SessionStore",
    "SessionState",
    "RequestState",
    "SenseVoiceASR",
    "SenseVoiceASRConfig",
    "ASRResult",
    "load_audio_for_asr",
    "normalize_transcript_text",
]


def __getattr__(name: str):
    if name == "VoxCPM":
        from .core import VoxCPM

        return VoxCPM
    if name in {"SenseVoiceASR", "SenseVoiceASRConfig", "ASRResult", "load_audio_for_asr", "normalize_transcript_text"}:
        from .asr import ASRResult, SenseVoiceASR, SenseVoiceASRConfig, load_audio_for_asr, normalize_transcript_text

        return {
            "SenseVoiceASR": SenseVoiceASR,
            "SenseVoiceASRConfig": SenseVoiceASRConfig,
            "ASRResult": ASRResult,
            "load_audio_for_asr": load_audio_for_asr,
            "normalize_transcript_text": normalize_transcript_text,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
