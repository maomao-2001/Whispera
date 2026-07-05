from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .asr import SenseVoiceASR, SenseVoiceASRConfig
from .session_protocol import (
    SessionStore,
    build_asr_completed_message,
    build_asr_final_message,
    build_asr_started_message,
    build_error_message,
    build_request_state_message,
    build_server_ready_message,
    build_session_ready_message,
    create_request_id,
)


WEBSOCKET_PATH = "/ws/asr"


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _decode_audio_message(message: Dict[str, Any]) -> np.ndarray:
    audio_format = str(message.get("audio_format") or ("pcm16" if message.get("pcm16") else "pcm_f32le")).lower()
    payload = message.get("data") or message.get("pcm16") or message.get("pcm_f32le")
    if not payload:
        raise ValueError("audio payload is required in 'data', 'pcm16', or 'pcm_f32le'")

    raw = base64.b64decode(payload)
    if audio_format in {"pcm16", "pcm_s16le", "pcm16le"}:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if audio_format in {"pcm_f32le", "float32", "f32"}:
        return np.frombuffer(raw, dtype=np.float32).astype(np.float32)
    raise ValueError(f"unsupported audio_format: {audio_format}")


@dataclass
class ASRRequestBuffer:
    sample_rate: int
    language: str
    use_itn: bool
    target_sample_rate: int
    chunks: list[np.ndarray] = field(default_factory=list)

    def append(self, audio: np.ndarray) -> None:
        if audio.size == 0:
            raise ValueError("received empty audio chunk")
        self.chunks.append(np.ascontiguousarray(audio, dtype=np.float32).reshape(-1))

    def get_audio(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(self.chunks), dtype=np.float32)


@dataclass
class ASRServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8003
    model: str = "iic/SenseVoiceSmall"
    target_sample_rate: int = 16000
    device: Optional[str] = None
    disable_update: bool = True
    language: str = "auto"
    use_itn: bool = True
    asr_log_level: str = "ERROR"
    log_level: str = "info"


class ASRService:
    def __init__(self, config: ASRServiceConfig):
        self.config = config
        self.session_store = SessionStore()
        self.asr = SenseVoiceASR(
            SenseVoiceASRConfig(
                model=config.model,
                target_sample_rate=config.target_sample_rate,
                device=config.device,
                disable_update=config.disable_update,
                log_level=config.asr_log_level,
            )
        )


def create_app(config: Optional[ASRServiceConfig] = None) -> FastAPI:
    config = config or ASRServiceConfig()
    service = ASRService(config)

    app = FastAPI(title="VoxCPM SenseVoice ASR Service", version="0.1.0")
    app.state.asr_service = service

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "voxcpm-sensevoice-asr",
            "websocket_path": WEBSOCKET_PATH,
            "model": config.model,
            "target_sample_rate": config.target_sample_rate,
            "model_loaded": service.asr.is_loaded,
            "session_count": len(service.session_store.list_sessions()),
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "backend": "sensevoice",
            "model": config.model,
            "target_sample_rate": config.target_sample_rate,
            "model_loaded": service.asr.is_loaded,
            "websocket_path": WEBSOCKET_PATH,
            "session_count": len(service.session_store.list_sessions()),
        }

    @app.get("/sessions")
    async def list_sessions() -> Dict[str, Any]:
        sessions = service.session_store.list_sessions()
        return {"count": len(sessions), "sessions": sessions}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str) -> Dict[str, Any]:
        session = service.session_store.get_session_snapshot(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
        return session

    @app.websocket(WEBSOCKET_PATH)
    async def websocket_asr(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            build_server_ready_message(
                WEBSOCKET_PATH,
                supported_client_messages=["session.start", "audio.append", "audio.commit", "ping"],
                message="send 'session.start' to establish a session, then stream audio with 'audio.append' and finish with 'audio.commit'",
            )
        )

        current_session_id: Optional[str] = None
        active_requests: dict[str, ASRRequestBuffer] = {}

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

                if current_session_id is None:
                    session_snapshot = service.session_store.start_session(message.get("session_id"))
                    current_session_id = session_snapshot["session_id"]
                    await websocket.send_json(build_session_ready_message(session_snapshot))

                request_id = message.get("request_id") or create_request_id()
                message["request_id"] = request_id

                if message_type == "audio.append":
                    try:
                        audio = _decode_audio_message(message)
                        sample_rate = int(message.get("sample_rate") or config.target_sample_rate)

                        if request_id not in active_requests:
                            request_snapshot = service.session_store.register_request(
                                session_id=current_session_id,
                                request_id=request_id,
                                request_type="asr",
                            )
                            await websocket.send_json(build_request_state_message(request_snapshot))

                            request_buffer = ASRRequestBuffer(
                                sample_rate=sample_rate,
                                language=str(message.get("language") or config.language),
                                use_itn=_to_bool(message.get("use_itn"), default=config.use_itn),
                                target_sample_rate=int(message.get("target_sample_rate") or config.target_sample_rate),
                            )
                            active_requests[request_id] = request_buffer

                            request_snapshot = service.session_store.mark_request_started(
                                session_id=current_session_id,
                                request_id=request_id,
                                sample_rate=sample_rate,
                            )
                            await websocket.send_json(build_request_state_message(request_snapshot))
                            await websocket.send_json(
                                build_asr_started_message(
                                    session_id=current_session_id,
                                    request_id=request_id,
                                    sample_rate=sample_rate,
                                    target_sample_rate=request_buffer.target_sample_rate,
                                    language=request_buffer.language,
                                    use_itn=request_buffer.use_itn,
                                )
                            )
                        else:
                            request_buffer = active_requests[request_id]
                            if sample_rate != request_buffer.sample_rate:
                                raise ValueError("sample_rate must stay consistent within one ASR request")

                        request_buffer.append(audio)
                        request_snapshot = service.session_store.mark_request_streaming(
                            session_id=current_session_id,
                            request_id=request_id,
                            num_samples=int(audio.size),
                        )
                        await websocket.send_json(build_request_state_message(request_snapshot))
                    except Exception as exc:
                        if request_id in active_requests:
                            request_snapshot = service.session_store.mark_request_failed(
                                session_id=current_session_id,
                                request_id=request_id,
                                error_message=str(exc),
                            )
                            await websocket.send_json(build_request_state_message(request_snapshot))
                            active_requests.pop(request_id, None)
                        await websocket.send_json(
                            build_error_message(str(exc), session_id=current_session_id, request_id=request_id)
                        )
                    continue

                if message_type == "audio.commit":
                    request_buffer = active_requests.get(request_id)
                    if request_buffer is None:
                        await websocket.send_json(
                            build_error_message(
                                "audio.commit received for unknown request_id",
                                session_id=current_session_id,
                                request_id=request_id,
                            )
                        )
                        continue

                    try:
                        result = service.asr.transcribe_audio(
                            request_buffer.get_audio(),
                            sample_rate=request_buffer.sample_rate,
                            language=request_buffer.language,
                            use_itn=request_buffer.use_itn,
                            target_sample_rate=request_buffer.target_sample_rate,
                        )
                        await websocket.send_json(
                            build_asr_final_message(
                                session_id=current_session_id,
                                request_id=request_id,
                                text=result.text,
                                language=result.language,
                            )
                        )
                        request_snapshot = service.session_store.mark_request_completed(
                            session_id=current_session_id,
                            request_id=request_id,
                            text=result.text,
                        )
                        await websocket.send_json(build_request_state_message(request_snapshot))
                        await websocket.send_json(
                            build_asr_completed_message(
                                request_snapshot,
                                target_sample_rate=request_buffer.target_sample_rate,
                            )
                        )
                    except Exception as exc:
                        request_snapshot = service.session_store.mark_request_failed(
                            session_id=current_session_id,
                            request_id=request_id,
                            error_message=str(exc),
                        )
                        await websocket.send_json(build_request_state_message(request_snapshot))
                        await websocket.send_json(
                            build_error_message(str(exc), session_id=current_session_id, request_id=request_id)
                        )
                    finally:
                        active_requests.pop(request_id, None)
                    continue

                await websocket.send_json(
                    build_error_message(
                        "unsupported message type, expected 'audio.append' or 'audio.commit'",
                        session_id=current_session_id,
                        request_id=request_id,
                    )
                )
        except WebSocketDisconnect:
            return
        finally:
            if current_session_id is not None:
                service.session_store.close_session(current_session_id)

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VoxCPM SenseVoice ASR websocket service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind (default: 8003)")
    parser.add_argument("--model", default="iic/SenseVoiceSmall", help="SenseVoice model id or local path")
    parser.add_argument("--target-sample-rate", type=int, default=16000, help="ASR preprocessing sample rate")
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--language", type=str, default="auto", help="SenseVoice language mode")
    parser.add_argument("--disable-itn", action="store_true", help="Disable inverse text normalization")
    parser.add_argument("--enable-update", action="store_true", help="Allow FunASR model metadata update checks")
    parser.add_argument("--asr-log-level", type=str, default="ERROR", help="FunASR model log level")
    parser.add_argument("--log-level", type=str, default="info", help="uvicorn log level")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ASRServiceConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        target_sample_rate=args.target_sample_rate,
        device=args.device,
        disable_update=not args.enable_update,
        language=args.language,
        use_itn=not args.disable_itn,
        asr_log_level=args.asr_log_level,
        log_level=args.log_level,
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()