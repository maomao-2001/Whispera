from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TTSRuntimeConfig:
    model_path: str | None = None
    lora_root: str | None = None
    load_denoiser: bool = False
    optimize: bool = True
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    min_len: int = 2
    max_len: int = 4096
    normalize: bool = False
    denoise: bool = False


@dataclass
class TTSRequestOptions:
    lora_selection: str | None = None
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    reference_wav_path: str | None = None
    cfg_value: float | None = None
    inference_timesteps: int | None = None
    seed: int | None = None
