import os
import sys
import re
import time
import tempfile
import json
import numpy as np
from typing import Any, Dict, Generator, Optional
from huggingface_hub import snapshot_download
from .model.voxcpm import VoxCPMModel, LoRAConfig
from .model.voxcpm2 import VoxCPM2Model

class VoxCPM:
    def __init__(self,
            voxcpm_model_path : str,
            zipenhancer_model_path : str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            enable_denoiser : bool = True,
            optimize: bool = True,
            device: Optional[str] = None,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
        ):
        """Initialize VoxCPM TTS pipeline.

        Args:
            voxcpm_model_path: Local filesystem path to the VoxCPM model assets
                (weights, configs, etc.). Typically the directory returned by
                a prior download step.
            zipenhancer_model_path: ModelScope acoustic noise suppression model
                id or local path. If None, denoiser will not be initialized.
            enable_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            device: Runtime device. If set to None or "auto", VoxCPM will choose automatically.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is 
                provided without lora_config, a default config will be created.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded.
        """
        print(f"voxcpm_model_path: {voxcpm_model_path}, zipenhancer_model_path: {zipenhancer_model_path}, enable_denoiser: {enable_denoiser}", file=sys.stderr)
        
        # If lora_weights_path is provided but no lora_config, create a default one
        if lora_weights_path is not None and lora_config is None:
            lora_config = LoRAConfig(
                enable_lm=True,
                enable_dit=True,
                enable_proj=False,
            )
            print(f"Auto-created default LoRAConfig for loading weights from: {lora_weights_path}", file=sys.stderr)
        
        config_path = os.path.join(voxcpm_model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
        arch = str(model_config.get("architecture", "voxcpm")).lower()

        if arch == "voxcpm2":
            self.tts_model = VoxCPM2Model.from_local(
                voxcpm_model_path,
                optimize=optimize,
                device=device,
                lora_config=lora_config,
            )
            print("Loaded VoxCPM2Model", file=sys.stderr)
        elif arch == "voxcpm":
            try:
                self.tts_model = VoxCPMModel.from_local(
                    voxcpm_model_path,
                    optimize=optimize,
                    device=device,
                    lora_config=lora_config,
                )
            except TypeError:
                self.tts_model = VoxCPMModel.from_local(
                    voxcpm_model_path,
                    optimize=optimize,
                    lora_config=lora_config,
                )
            print("Loaded VoxCPMModel", file=sys.stderr)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        # Load LoRA weights if path is provided
        if lora_weights_path is not None:
            print(f"Loading LoRA weights from: {lora_weights_path}", file=sys.stderr)
            loaded_keys, skipped_keys = self.tts_model.load_lora_weights(lora_weights_path)
            print(f"Loaded {len(loaded_keys)} LoRA parameters, skipped {len(skipped_keys)}", file=sys.stderr)
        
        self.text_normalizer = None
        if enable_denoiser and zipenhancer_model_path is not None:
            from .zipenhancer import ZipEnhancer
            self.denoiser = ZipEnhancer(zipenhancer_model_path)
        else:
            self.denoiser = None
        self.last_generation_metrics: Optional[Dict[str, Any]] = None
        if optimize and getattr(self.tts_model, "device", None) == "cuda":
            print("Warm up VoxCPMModel...", file=sys.stderr)
            self.tts_model.generate(
                target_text="Hello, this is the first test sentence.",
                max_len=10,
            )

    @classmethod
    def from_pretrained(cls,
            hf_model_id: str = "openbmb/VoxCPM2",
            load_denoiser: bool = True,
            zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base",
            cache_dir: str = None,
            local_files_only: bool = False,
            optimize: bool = True,
            device: Optional[str] = None,
            lora_config: Optional[LoRAConfig] = None,
            lora_weights_path: Optional[str] = None,
            **kwargs,
        ):
        """Instantiate ``VoxCPM`` from a Hugging Face Hub snapshot.

        Args:
            hf_model_id: Explicit Hugging Face repository id (e.g. "org/repo") or local path.
            load_denoiser: Whether to initialize the denoiser pipeline.
            optimize: Whether to optimize the model with torch.compile. True by default, but can be disabled for debugging.
            zipenhancer_model_id: Denoiser model id or path for ModelScope
                acoustic noise suppression.
            cache_dir: Custom cache directory for the snapshot.
            local_files_only: If True, only use local files and do not attempt
                to download.
            device: Runtime device. Use None/"auto" for automatic fallback, or an explicit device.
            lora_config: LoRA configuration for fine-tuning. If lora_weights_path is 
                provided without lora_config, a default config will be created with
                enable_lm=True and enable_dit=True.
            lora_weights_path: Path to pre-trained LoRA weights (.pth file or directory
                containing lora_weights.ckpt). If provided, LoRA weights will be loaded
                after model initialization.
        Kwargs:
            Additional keyword arguments passed to the ``VoxCPM`` constructor.

        Returns:
            VoxCPM: Initialized instance whose ``voxcpm_model_path`` points to
            the downloaded snapshot directory.

        Raises:
            ValueError: If neither a valid ``hf_model_id`` nor a resolvable
                ``hf_model_id`` is provided.
        """
        repo_id = hf_model_id
        if not repo_id:
            raise ValueError("You must provide hf_model_id")
        
        # Load from local path if provided
        if os.path.isdir(repo_id):
            local_path = repo_id
        else:
            # Otherwise, try from_pretrained (Hub); exit on failure
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )

        return cls(
            voxcpm_model_path=local_path,
            zipenhancer_model_path=zipenhancer_model_id if load_denoiser else None,
            enable_denoiser=load_denoiser,
            optimize=optimize,
            device=device,
            lora_config=lora_config,
            lora_weights_path=lora_weights_path,
            **kwargs,
        )

    def generate(self, *args, **kwargs) -> np.ndarray:
        result = None
        for result in self._generate(*args, streaming=False, **kwargs):
            pass

        if result is None:
            raise RuntimeError("VoxCPM.generate returned no audio")

        return result

    def generate_streaming(self, *args, **kwargs) -> Generator[np.ndarray, None, None]:
        return self._generate(*args, streaming=True, **kwargs)

    def _update_generation_metrics(self, total_samples: int, elapsed_sec: float, streaming: bool) -> None:
        sample_rate = int(self.tts_model.sample_rate)
        audio_duration_sec = float(total_samples) / sample_rate if sample_rate > 0 else 0.0
        rtf = elapsed_sec /  audio_duration_sec if elapsed_sec > 0 else None

        self.last_generation_metrics = {
            "audio_duration_sec": round(audio_duration_sec, 4),
            "elapsed_sec": round(elapsed_sec, 4),
            "rtf": round(rtf, 4) if rtf is not None else None,
            "total_samples": int(total_samples),
            "sample_rate": sample_rate,
            "streaming": bool(streaming),
        }

        rtf_text = f"{rtf:.4f}" if rtf is not None else "n/a"
        print(
            "[VoxCPM] generation completed "
            f"| streaming={streaming} "
            f"| audio_duration={audio_duration_sec:.2f}s "
            f"| elapsed={elapsed_sec:.2f}s "
            f"| RTF={rtf_text}",
            file=sys.stderr,
        )

    def _generate(self, 
            text : str,
            prompt_wav_path : str = None,
            prompt_text : str = None,
            reference_wav_path: str = None,
            cfg_value : float = 2.0,    
            inference_timesteps : int = 10,
            min_len : int = 2,
            max_len : int = 4096,
            normalize : bool = False,
            denoise : bool = False,
            retry_badcase : bool = True,
            retry_badcase_max_times : int = 3,
            retry_badcase_ratio_threshold : float = 6.0,
            streaming_prefix_len: int = 4,
            streaming_emit_interval: int = 1,
            streaming: bool = False,
            seed: Optional[int] = None,
        ) -> Generator[np.ndarray, None, None]:
        """Synthesize speech for the given text and return a single waveform.

        This method optionally uses a fixed prompt cache. If an external prompt
        (``prompt_wav_path`` + ``prompt_text``) is provided, it is built once
        and reused for the whole request. Without an external prompt, synthesis
        runs without prompt cache; cross-segment prompt-cache orchestration is
        expected to be handled by the embedding application if needed.

        Args:
            text: Input text. Can include newlines; each non-empty line is
                treated as a sub-sentence.
            prompt_wav_path: Path to a reference audio file for prompting.
            prompt_text: Text content corresponding to the prompt audio.
            reference_wav_path: Path to reference audio for VoxCPM2 voice cloning.
            cfg_value: Guidance scale for the generation model.
            inference_timesteps: Number of inference steps.
            max_len: Maximum token length during generation.
            normalize: Whether to run text normalization before generation.
            denoise: Whether to denoise the prompt audio if a denoiser is
                available.
            retry_badcase: Whether to retry badcase.
            retry_badcase_max_times: Maximum number of times to retry badcase.
            retry_badcase_ratio_threshold: Threshold for audio-to-text ratio.
            streaming_prefix_len: Number of prefix audio patches to use for streaming mode.
            streaming_emit_interval: V1 chunk emit interval. Ignored for VoxCPM2.
            streaming: Whether to return a generator of audio chunks.
            seed: Optional VoxCPM2 random seed for reproducibility.
        Returns:
            Generator of numpy.ndarray: 1D waveform array (float32) on CPU. 
            Yields audio chunks for each generations step if ``streaming=True``,
            otherwise yields a single array containing the final audio.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("target text must be a non-empty string")
        
        if prompt_wav_path is not None:
            if not os.path.exists(prompt_wav_path):
                raise FileNotFoundError(f"prompt_wav_path does not exist: {prompt_wav_path}")

        if reference_wav_path is not None:
            if not os.path.exists(reference_wav_path):
                raise FileNotFoundError(f"reference_wav_path does not exist: {reference_wav_path}")
        
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")

        is_v2 = isinstance(self.tts_model, VoxCPM2Model)
        if reference_wav_path is not None and not is_v2:
            raise ValueError("reference_wav_path is only supported with VoxCPM2 models")
        if seed is not None and int(seed) < 0:
            seed = None
        
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)
        temp_files = []
        total_samples = 0
        generation_start_time = time.perf_counter()
        self.last_generation_metrics = None
        
        try:
            actual_prompt_path = prompt_wav_path
            actual_ref_path = reference_wav_path

            if denoise and self.denoiser is not None:
                if prompt_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_files.append(tmp_file.name)
                    self.denoiser.enhance(prompt_wav_path, output_path=temp_files[-1])
                    actual_prompt_path = temp_files[-1]
                if reference_wav_path is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        temp_files.append(tmp_file.name)
                    self.denoiser.enhance(reference_wav_path, output_path=temp_files[-1])
                    actual_ref_path = temp_files[-1]

            if actual_prompt_path is not None or actual_ref_path is not None:
                if is_v2:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_wav_path=actual_prompt_path,
                        prompt_text=prompt_text,
                        reference_wav_path=actual_ref_path,
                    )
                else:
                    fixed_prompt_cache = self.tts_model.build_prompt_cache(
                        prompt_wav_path=actual_prompt_path,
                        prompt_text=prompt_text
                    )
            else:
                fixed_prompt_cache = None
            
            if normalize:
                if self.text_normalizer is None:
                    from .utils.text_normalize import TextNormalizer
                    self.text_normalizer = TextNormalizer()
                text = self.text_normalizer.normalize(text)
            
            generate_kwargs = dict(
                target_text=text,
                prompt_cache=fixed_prompt_cache,
                min_len=min_len,
                max_len=max_len,
                inference_timesteps=inference_timesteps,
                cfg_value=cfg_value,
                retry_badcase=retry_badcase,
                retry_badcase_max_times=retry_badcase_max_times,
                retry_badcase_ratio_threshold=retry_badcase_ratio_threshold,
                streaming=streaming,
                streaming_prefix_len=streaming_prefix_len,
            )
            if is_v2:
                generate_kwargs["seed"] = seed
            else:
                generate_kwargs["streaming_emit_interval"] = streaming_emit_interval

            generate_result = self.tts_model._generate_with_prompt_cache(**generate_kwargs)
        
            try:
                for wav, _, _ in generate_result:
                    chunk = wav.squeeze(0).cpu().numpy()
                    total_samples += int(chunk.size)
                    yield chunk
            finally:
                close = getattr(generate_result, "close", None)
                if callable(close):
                    close()

            elapsed_sec = time.perf_counter() - generation_start_time
            self._update_generation_metrics(
                total_samples=total_samples,
                elapsed_sec=elapsed_sec,
                streaming=streaming,
            )
        
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except OSError:
                        pass

    # ------------------------------------------------------------------ #
    # LoRA Interface (delegated to VoxCPMModel)
    # ------------------------------------------------------------------ #
    def load_lora(self, lora_weights_path: str) -> tuple:
        """Load LoRA weights from a checkpoint file.
        
        Args:
            lora_weights_path: Path to LoRA weights (.pth file or directory
                containing lora_weights.ckpt).
        
        Returns:
            tuple: (loaded_keys, skipped_keys) - lists of loaded and skipped parameter names.
        
        Raises:
            RuntimeError: If model was not initialized with LoRA config.
        """
        if self.tts_model.lora_config is None:
            raise RuntimeError(
                "Cannot load LoRA weights: model was not initialized with LoRA config. "
                "Please reinitialize with lora_config or lora_weights_path parameter."
            )
        return self.tts_model.load_lora_weights(lora_weights_path)

    def unload_lora(self):
        """Unload LoRA by resetting all LoRA weights to initial state (effectively disabling LoRA)."""
        self.tts_model.reset_lora_weights()
    
    def set_lora_enabled(self, enabled: bool):
        """Enable or disable LoRA layers without unloading weights.
        
        Args:
            enabled: If True, LoRA layers are active; if False, only base model is used.
        """
        self.tts_model.set_lora_enabled(enabled)
    
    def get_lora_state_dict(self) -> dict:
        """Get current LoRA parameters state dict.
        
        Returns:
            dict: State dict containing all LoRA parameters (lora_A, lora_B).
        """
        return self.tts_model.get_lora_state_dict()
    
    @property
    def lora_enabled(self) -> bool:
        """Check if LoRA is currently configured."""
        return self.tts_model.lora_config is not None
