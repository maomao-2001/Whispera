from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, Generator, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .session_protocol import (
    SessionStore,
    build_error_message,
    build_interrupt_ack_message,
    build_request_state_message,
    build_server_ready_message,
    build_session_ready_message,
    build_tts_chunk_message,
    build_tts_completed_message,
    build_tts_started_message,
    create_request_id,
)

if TYPE_CHECKING:
    from .core import VoxCPM


WEBSOCKET_PATH = "/ws/tts"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_default_model_path() -> Optional[str]:
    candidates = [
        _repo_root() / "models" / "openbmb__VoxCPM1.5",
        _repo_root() / "models" / "VoxCPM1.5",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)
    return None


def resolve_default_lora_root() -> Path:
    return _repo_root() / "lora"


def scan_lora_checkpoints(root_dir: str | Path | None = None, with_info: bool = False) -> list[Any]:
    checkpoints: list[Any] = []
    root_path = Path(root_dir) if root_dir is not None else resolve_default_lora_root()
    root_path.mkdir(parents=True, exist_ok=True)

    for directory, _, files in os.walk(root_path):
        if "lora_weights.safetensors" not in files and "lora_weights.ckpt" not in files:
            continue

        checkpoint_dir = Path(directory)
        rel_path = checkpoint_dir.relative_to(root_path).as_posix()
        if with_info:
            _, base_model = load_lora_config_from_checkpoint(checkpoint_dir)
            checkpoints.append((rel_path, base_model))
        else:
            checkpoints.append(rel_path)

    checkpoints.sort(reverse=True)
    return checkpoints


def load_lora_config_from_checkpoint(lora_path: str | Path):
    from .model.voxcpm import LoRAConfig

    checkpoint_path = Path(lora_path)
    config_file = checkpoint_path / "lora_config.json"
    if config_file.is_file():
        try:
            lora_info = json.loads(config_file.read_text(encoding="utf-8"))
            lora_cfg_dict = lora_info.get("lora_config", {})
            if lora_cfg_dict:
                return LoRAConfig(**lora_cfg_dict), lora_info.get("base_model")
            return None, lora_info.get("base_model")
        except Exception as exc:
            print(f"[streaming-service] failed to load {config_file}: {exc}", file=sys.stderr)
    return None, None


def get_default_lora_config():
    from .model.voxcpm import LoRAConfig

    return LoRAConfig(
        enable_lm=True,
        enable_dit=True,
        r=32,
        alpha=16,
        target_modules_lm=["q_proj", "v_proj", "k_proj", "o_proj"],
        target_modules_dit=["q_proj", "v_proj", "k_proj", "o_proj"],
    )


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _next_stream_item(stream: Any) -> tuple[bool, Any]:
    try:
        return False, next(stream)
    except StopIteration:
        return True, None


def _close_stream(stream: Any) -> None:
    close = getattr(stream, "close", None)
    if callable(close):
        close()


@dataclass
class StreamingServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    model_path: Optional[str] = None
    hf_model_id: str = "openbmb/VoxCPM1.5"
    cache_dir: Optional[str] = None
    local_files_only: bool = False
    load_denoiser: bool = False
    zipenhancer_model_id: str = "iic/speech_zipenhancer_ans_multiloss_16k_base"
    optimize: bool = True
    streaming_prefix_len: int = 3
    streaming_emit_interval: int = 4
    log_level: str = "info"


class StreamingTTSService:
    def __init__(self, config: StreamingServiceConfig):
        self.config = config
        self._model: Optional[VoxCPM] = None
        self._model_lock = Lock()
        self._lora_root = resolve_default_lora_root()
        self._active_lora_key: Optional[str] = None
        self._active_lora_path: Optional[str] = None
        self._loaded_model_source: Optional[str] = None
        self.session_store = SessionStore()

    @property
    def is_model_loaded(self) -> bool:
        return self._model is not None

    @property
    def active_lora_key(self) -> Optional[str]:
        return self._active_lora_key

    @property
    def active_lora_path(self) -> Optional[str]:
        return self._active_lora_path

    @property
    def loaded_model_source(self) -> Optional[str]:
        return self._loaded_model_source

    def _create_model(self, model_path: Optional[str] = None):
        from .core import VoxCPM

        lora_config = get_default_lora_config()
        if model_path:
            print(f"[streaming-service] loading local model from: {model_path}", file=sys.stderr)
            model = VoxCPM(
                voxcpm_model_path=model_path,
                zipenhancer_model_path=self.config.zipenhancer_model_id,
                enable_denoiser=self.config.load_denoiser,
                optimize=self.config.optimize,
                lora_config=lora_config,
            )
            self._loaded_model_source = str(Path(model_path).resolve())
            return model

        print(
            f"[streaming-service] loading model from hub: {self.config.hf_model_id}",
            file=sys.stderr,
        )
        model = VoxCPM.from_pretrained(
            hf_model_id=self.config.hf_model_id,
            load_denoiser=self.config.load_denoiser,
            zipenhancer_model_id=self.config.zipenhancer_model_id,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
            optimize=self.config.optimize,
            lora_config=lora_config,
        )
        self._loaded_model_source = f"hf:{self.config.hf_model_id}"
        return model

    def get_model(self) -> VoxCPM:
        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            model_path = self.config.model_path or resolve_default_model_path()
            self._model = self._create_model(model_path=model_path)
            return self._model

    def list_lora_checkpoints(self, with_info: bool = False) -> list[Any]:
        return scan_lora_checkpoints(self._lora_root, with_info=with_info)

    def resolve_lora_selection(self, selection: Any) -> tuple[Optional[str], Optional[Path]]:
        raw = str(selection or "").strip()
        if not raw or raw.lower() == "none":
            return None, None

        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (self._lora_root / raw).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {candidate}")

        try:
            key = candidate.relative_to(self._lora_root.resolve()).as_posix()
        except Exception:
            key = candidate.as_posix()
        return key, candidate

    def get_lora_base_model_path(self, lora_selection: Any) -> Optional[str]:
        _, checkpoint_path = self.resolve_lora_selection(lora_selection)
        if checkpoint_path is None:
            return None

        _, base_model = load_lora_config_from_checkpoint(checkpoint_path)
        if not base_model:
            return None

        base_model_path = Path(str(base_model)).expanduser()
        if not base_model_path.is_absolute():
            base_model_path = (_repo_root() / base_model_path).resolve()
        if base_model_path.exists():
            return str(base_model_path)
        return None

    def ensure_model_for_lora_selection(self, selection: Any) -> VoxCPM:
        required_model_path = self.get_lora_base_model_path(selection)
        required_source = str(Path(required_model_path).resolve()) if required_model_path else None

        with self._model_lock:
            if self._model is None:
                self._model = self._create_model(model_path=required_model_path or self.config.model_path or resolve_default_model_path())
                return self._model

            if required_source and self._loaded_model_source != required_source:
                print(
                    f"[streaming-service] reloading model for LoRA base model: {required_source}",
                    file=sys.stderr,
                )
                self._model = self._create_model(model_path=required_source)
                self._active_lora_key = None
                self._active_lora_path = None

            return self._model

    def apply_lora_selection(self, selection: Any) -> dict[str, Any]:
        model = self.ensure_model_for_lora_selection(selection)
        key, checkpoint_path = self.resolve_lora_selection(selection)

        if checkpoint_path is None:
            model.set_lora_enabled(False)
            self._active_lora_key = None
            self._active_lora_path = None
            return {"enabled": False, "selection": None, "path": None}

        checkpoint_str = str(checkpoint_path)
        if self._active_lora_path != checkpoint_str:
            model.load_lora(checkpoint_str)
            self._active_lora_path = checkpoint_str
        model.set_lora_enabled(True)
        self._active_lora_key = key
        return {"enabled": True, "selection": key, "path": checkpoint_str}

    def generate_stream(self, request: Dict[str, Any]) -> Generator[np.ndarray, None, None]:
        text = str(request.get("text", "")).strip()
        if not text:
            raise ValueError("'text' must be a non-empty string")

        prompt_audio_path = request.get("prompt_audio_path")
        prompt_text = request.get("prompt_text")

        if prompt_audio_path == "":
            prompt_audio_path = None
        if prompt_text == "":
            prompt_text = None

        self.apply_lora_selection(request.get("lora_path") or request.get("lora_selection"))
        model = self.get_model()

        return model.generate_streaming(
            text=text,
            prompt_wav_path=prompt_audio_path,
            prompt_text=prompt_text,
            cfg_value=_to_float(request.get("cfg_value"), 2.0),
            inference_timesteps=_to_int(request.get("inference_timesteps"), 10),
            min_len=_to_int(request.get("min_len"), 2),
            max_len=_to_int(request.get("max_len"), 4096),
            normalize=_to_bool(request.get("normalize"), False),
            denoise=_to_bool(request.get("denoise"), False),
            streaming_prefix_len=_to_int(request.get("streaming_prefix_len"), self.config.streaming_prefix_len),
            streaming_emit_interval=_to_int(request.get("streaming_emit_interval"), self.config.streaming_emit_interval),
            retry_badcase=False,
        )


async def _send_error(websocket: WebSocket, message: str, request_id: Optional[str] = None) -> None:
    await websocket.send_json(build_error_message(message=message, request_id=request_id))


def create_app(
    config: Optional[StreamingServiceConfig] = None,
    service: Optional[StreamingTTSService] = None,
) -> FastAPI:
    config = config or StreamingServiceConfig()
    service = service or StreamingTTSService(config)

    app = FastAPI(title="VoxCPM Streaming TTS Service", version="0.1.0")
    app.state.streaming_tts_service = service

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "voxcpm-streaming-tts",
            "websocket_path": WEBSOCKET_PATH,
            "model_loaded": service.is_model_loaded,
            "session_count": len(service.session_store.list_sessions()),
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        model = service._model
        return {
            "status": "ok",
            "model_loaded": service.is_model_loaded,
            "sample_rate": getattr(getattr(model, "tts_model", None), "sample_rate", None),
            "websocket_path": WEBSOCKET_PATH,
            "session_count": len(service.session_store.list_sessions()),
        }

    @app.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        sessions = service.session_store.list_sessions()
        return {
            "count": len(sessions),
            "sessions": sessions,
        }

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> Dict[str, Any]:
        session = service.session_store.get_session_snapshot(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
        return session

    @app.websocket(WEBSOCKET_PATH)
    async def websocket_tts(websocket: WebSocket) -> None:
        await websocket.accept()
        current_session_id: Optional[str] = None
        send_lock = asyncio.Lock()
        active_generation_task: Optional[asyncio.Task[None]] = None

        async def send_json(payload: Dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def run_generation(session_id: str, request_id: str, request_message: Dict[str, Any]) -> None:
            stream = None
            try:
                model = service.get_model()
                sample_rate = int(model.tts_model.sample_rate)
                stream = service.generate_stream(request_message)
                if service.session_store.is_request_interrupted(session_id, request_id):
                    return

                request_snapshot = service.session_store.mark_request_started(
                    session_id=session_id,
                    request_id=request_id,
                    sample_rate=sample_rate,
                )
                await send_json(build_request_state_message(request_snapshot))
                await send_json(build_tts_started_message(session_id, request_id, sample_rate))

                chunk_count = 0
                while True:
                    if service.session_store.is_request_interrupted(session_id, request_id):
                        break

                    stream_done, chunk = await asyncio.to_thread(_next_stream_item, stream)
                    if stream_done:
                        break
                    if chunk is None or service.session_store.is_request_interrupted(session_id, request_id):
                        continue

                    chunk = np.asarray(chunk, dtype=np.float32)
                    if chunk.ndim != 1:
                        chunk = chunk.reshape(-1)

                    payload = base64.b64encode(np.ascontiguousarray(chunk).tobytes()).decode("ascii")
                    request_snapshot = service.session_store.mark_request_streaming(
                        session_id=session_id,
                        request_id=request_id,
                        num_samples=int(chunk.size),
                    )
                    await send_json(build_request_state_message(request_snapshot))
                    await send_json(
                        build_tts_chunk_message(
                            session_id=session_id,
                            request_id=request_id,
                            index=chunk_count,
                            num_samples=int(chunk.size),
                            data=payload,
                        )
                    )
                    chunk_count += 1

                if service.session_store.is_request_interrupted(session_id, request_id):
                    return

                request_snapshot = service.session_store.mark_request_completed(
                    session_id=session_id,
                    request_id=request_id,
                )
                await send_json(build_request_state_message(request_snapshot))
                await send_json(build_tts_completed_message(request_snapshot))
            except Exception as exc:
                if service.session_store.is_request_interrupted(session_id, request_id):
                    return

                request_snapshot = service.session_store.get_request_snapshot(request_id)
                if request_snapshot is not None and request_snapshot["state"] not in {"completed", "interrupted", "failed"}:
                    request_snapshot = service.session_store.mark_request_failed(
                        session_id=session_id,
                        request_id=request_id,
                        error_message=str(exc),
                    )
                    await send_json(build_request_state_message(request_snapshot))
                await send_json(
                    build_error_message(
                        message=str(exc),
                        session_id=session_id,
                        request_id=request_id,
                    )
                )
            finally:
                if stream is not None:
                    await asyncio.to_thread(_close_stream, stream)

        def reset_active_generation_task(task: asyncio.Task[None]) -> None:
            nonlocal active_generation_task
            if active_generation_task is task:
                active_generation_task = None

        await websocket.send_json(
            build_server_ready_message(
                WEBSOCKET_PATH,
                supported_client_messages=["session.start", "tts.start", "tts.interrupt", "ping"],
                message="send 'session.start' to establish a session, then send 'tts.start' to begin streaming synthesis",
            )
        )

        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")

                if message_type == "ping":
                    await websocket.send_json({"type": "pong", "session_id": current_session_id})
                    continue

                if message_type == "session.start":
                    session_snapshot = service.session_store.start_session(message.get("session_id"))
                    current_session_id = session_snapshot["session_id"]
                    await websocket.send_json(build_session_ready_message(session_snapshot))
                    continue

                request_id = message.get("request_id") or create_request_id()
                message["request_id"] = request_id

                if message_type == "tts.interrupt":
                    if current_session_id is None:
                        session_snapshot = service.session_store.start_session(message.get("session_id"))
                        current_session_id = session_snapshot["session_id"]
                        await send_json(build_session_ready_message(session_snapshot))

                    interrupted_snapshot = service.session_store.interrupt_active_request(
                        session_id=current_session_id,
                        request_type="tts",
                        reason="client_interrupt",
                    )
                    if interrupted_snapshot is not None and interrupted_snapshot["state"] == "interrupted":
                        await send_json(build_request_state_message(interrupted_snapshot))

                    await send_json(
                        build_interrupt_ack_message(
                            session_id=current_session_id,
                            request_id=request_id,
                            interrupted_request_id=interrupted_snapshot["request_id"] if interrupted_snapshot else None,
                            request_type="tts",
                            accepted=bool(interrupted_snapshot and interrupted_snapshot["state"] == "interrupted"),
                            reason="client_interrupt",
                        )
                    )
                    continue

                if message_type != "tts.start":
                    request_id = message.get("request_id")
                    await _send_error(
                        websocket,
                        "unsupported message type, expected 'tts.start' or 'tts.interrupt'",
                        request_id=request_id,
                    )
                    continue

                if current_session_id is None:
                    session_snapshot = service.session_store.start_session(message.get("session_id"))
                    current_session_id = session_snapshot["session_id"]
                    await websocket.send_json(build_session_ready_message(session_snapshot))

                active_request_id = service.session_store.get_active_request_id(current_session_id)
                if active_request_id is not None:
                    await send_json(
                        build_error_message(
                            "active tts request is still running, send 'tts.interrupt' before starting a new one",
                            session_id=current_session_id,
                            request_id=request_id,
                        )
                    )
                    continue

                try:
                    request_snapshot = service.session_store.register_request(
                        session_id=current_session_id,
                        request_id=request_id,
                        request_type="tts",
                        text=message.get("text"),
                    )
                    await send_json(build_request_state_message(request_snapshot))
                    active_generation_task = asyncio.create_task(run_generation(current_session_id, request_id, dict(message)))
                    active_generation_task.add_done_callback(reset_active_generation_task)
                except Exception as exc:
                    request_snapshot = service.session_store.get_request_snapshot(request_id)
                    if request_snapshot is not None and request_snapshot["state"] not in {"completed", "interrupted", "failed"}:
                        request_snapshot = service.session_store.mark_request_failed(
                            session_id=current_session_id,
                            request_id=request_id,
                            error_message=str(exc),
                        )
                        await send_json(build_request_state_message(request_snapshot))
                    await send_json(
                        build_error_message(
                            message=str(exc),
                            session_id=current_session_id,
                            request_id=request_id,
                        )
                    )
        except WebSocketDisconnect:
            return
        finally:
            if current_session_id is not None:
                service.session_store.interrupt_active_request(
                    session_id=current_session_id,
                    request_type="tts",
                    reason="disconnect",
                )
            if active_generation_task is not None:
                with suppress(Exception):
                    await active_generation_task
            if current_session_id is not None:
                service.session_store.close_session(current_session_id)

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VoxCPM streaming TTS websocket service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--model-path", type=str, default=None, help="Local VoxCPM model path")
    parser.add_argument(
        "--hf-model-id",
        type=str,
        default="openbmb/VoxCPM1.5",
        help="Hugging Face model id used when local model path is not available",
    )
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Disable network downloads")
    parser.add_argument("--load-denoiser", action="store_true", help="Initialize ZipEnhancer denoiser")
    parser.add_argument(
        "--zipenhancer-model-id",
        type=str,
        default="iic/speech_zipenhancer_ans_multiloss_16k_base",
        help="ZipEnhancer model id or local path",
    )
    parser.add_argument("--disable-optimize", action="store_true", help="Disable torch.compile optimization")
    parser.add_argument(
        "--streaming-prefix-len",
        type=int,
        default=3,
        help="Streaming decode context window in patches",
    )
    parser.add_argument(
        "--streaming-emit-interval",
        type=int,
        default=4,
        help="Emit one TTS chunk after accumulating this many inference steps",
    )
    parser.add_argument("--log-level", type=str, default="info", help="uvicorn log level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = StreamingServiceConfig(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        hf_model_id=args.hf_model_id,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        load_denoiser=args.load_denoiser,
        zipenhancer_model_id=args.zipenhancer_model_id,
        optimize=not args.disable_optimize,
        streaming_prefix_len=args.streaming_prefix_len,
        streaming_emit_interval=args.streaming_emit_interval,
        log_level=args.log_level,
    )

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
