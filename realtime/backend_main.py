from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import os
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect

from .asr_service import ASRRuntimeConfig, FunAsrService
from .llm_client import LLMRuntimeConfig, LocalLLMClient
from .local_modules import REPO_ROOT
from .memory_service import MemoryRuntimeConfig, RealtimeMemoryService

import logging
logger = logging.getLogger("uvicorn.error.realtime")
from .session_manager import ConversationState
from .text_segmenter import StreamingTextSegmenter
from .turn_capture import TurnCaptureRecorder
from .tts_types import TTSRequestOptions, TTSRuntimeConfig
from .vad_session import RealtimeSession, VADConfig, VADTurnTiming

if TYPE_CHECKING:
    from .tts_service import VoxCpmTtsService


def _next_stream_item(stream):
    try:
        return False, next(stream)
    except StopIteration:
        return True, None


def _encode_pcm_f32(audio: np.ndarray) -> tuple[str, int]:
    chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
    return base64.b64encode(chunk.tobytes()).decode("ascii"), int(chunk.size)


def _elapsed_ms(started_at: float | None, ended_at: float | None) -> float | None:
    if started_at is None or ended_at is None:
        return None
    return (ended_at - started_at) * 1000.0


def _normalize_log_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 2)
    return value


def _log_turn_timing(
    session_id: str,
    turn_id: str | None,
    stage: str,
    *,
    event_at: float | None = None,
    speech_finished_at: float | None = None,
    **fields: Any,
) -> None:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "stage": stage,
    }
    if turn_id:
        payload["turn_id"] = turn_id
    if event_at is not None and speech_finished_at is not None:
        # since_user_done_ms: elapsed time since the user actually finished
        # speaking (estimated as VAD speech-end detection time minus the VAD
        # trailing-silence window).
        payload["since_user_done_ms"] = round((event_at - speech_finished_at) * 1000.0, 2)
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = _normalize_log_value(value)
    logger.info("[TURN_TIMING] %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))


@dataclass
class RealtimeAppConfig:
    host: str = "127.0.0.1"
    port: int = 8011
    llm_base_url: str = "http://127.0.0.1:8080"
    llm_model: str | None = None
    max_history_turns: int = 6
    asr_model_path: str = str(REPO_ROOT / "model" / "SenseVoiceSmall")
    asr_device: str = "cuda"
    vad_path: str = str(REPO_ROOT / "model" / "vad" / "silero_vad.onnx")
    vad_threshold: float = 0.8
    vad_min_speech_ms: int = 128
    vad_min_silence_ms: int = 800
    tts_enabled: bool = False
    tts_model_path: str | None = None
    tts_lora_root: str | None = None
    debug_turns: bool = False
    debug_output_dir: str | None = None
    memory_enabled: bool = False


def _build_tts_service(config: RealtimeAppConfig) -> tuple["VoxCpmTtsService | None", str | None]:
    if not config.tts_enabled:
        return None, None

    try:
        from .tts_service import VoxCpmTtsService
    except Exception as exc:
        return None, f"optional TTS dependencies are unavailable: {exc}"

    try:
        service = VoxCpmTtsService(TTSRuntimeConfig(model_path=config.tts_model_path, lora_root=config.tts_lora_root))
    except Exception as exc:
        return None, f"failed to initialize optional TTS service: {exc}"

    return service, None


class RealtimeRuntime:
    def __init__(self, config: RealtimeAppConfig):
        self.config = config
        self.llm = LocalLLMClient(
            LLMRuntimeConfig(
                base_url=config.llm_base_url,
                model=config.llm_model,
            )
        )
        self.asr = FunAsrService(
            ASRRuntimeConfig(
                model_path=config.asr_model_path,
                device=config.asr_device,
            )
        )
        self.tts_requested = bool(config.tts_enabled)
        self.tts, self.tts_error = _build_tts_service(config)
        # torch.compile(mode="reduce-overhead") stores its CUDA-graph manager in
        # thread-local storage, so every call that touches the compiled TTS model
        # (warmup, sample_rate, each streaming next()) must run on the SAME thread.
        # asyncio.to_thread uses the default multi-threaded pool, which routes calls
        # to arbitrary threads and trips `assert _is_key_in_tls(...)` in
        # cudagraph_trees. Pinning all TTS work to one dedicated thread fixes it.
        self.tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self.memory = RealtimeMemoryService(
            MemoryRuntimeConfig.from_env(
                llm_base_url=config.llm_base_url,
                llm_model=config.llm_model,
                enabled=config.memory_enabled,
            )
        )
        logger.info("[MEMORY-BOOT] enabled=%s available=%s last_error=%s",
                    self.memory.config.enabled, self.memory.available,
                    self.memory._last_error)

    @property
    def tts_enabled(self) -> bool:
        return self.tts is not None


def _decode_pcm16(audio_bytes: bytes) -> np.ndarray:
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _number_or_none(value: Any, kind: type[int] | type[float]):
    if value is None or value == "":
        return None
    return kind(value)


def _build_tts_options(payload: dict[str, Any], base: TTSRequestOptions | None = None) -> TTSRequestOptions:
    nested = payload.get("tts_options")
    source = nested if isinstance(nested, dict) else payload
    current = base or TTSRequestOptions()
    return TTSRequestOptions(
        model_version=_clean_text(source.get("model_version", current.model_version)) or current.model_version,
        lora_selection=_clean_text(source.get("lora_selection", current.lora_selection)),
        prompt_audio_path=_clean_text(source.get("prompt_audio_path", current.prompt_audio_path)),
        prompt_text=_clean_text(source.get("prompt_text", current.prompt_text)),
        reference_wav_path=_clean_text(source.get("reference_wav_path", current.reference_wav_path)),
        cfg_value=_number_or_none(source.get("cfg_value", current.cfg_value), float),
        inference_timesteps=_number_or_none(source.get("inference_timesteps", current.inference_timesteps), int),
        seed=_number_or_none(source.get("seed", current.seed), int),
    )


def _bool_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def create_app(config: RealtimeAppConfig | None = None):
    config = config or RealtimeAppConfig()
    runtime = RealtimeRuntime(config)
    app = FastAPI(title="MiniMind Realtime Backend", version="0.1.0")
    app.state.realtime_runtime = runtime

    @app.get("/health")
    async def health():
        models = []
        upstream_ok = True
        error_message = None
        try:
            models = runtime.llm.list_models()
        except Exception as exc:
            upstream_ok = False
            error_message = str(exc)
        notes = [
            "text.input is available in the scaffold",
            "binary audio streaming is now supported",
        ]
        if runtime.tts_enabled:
            notes.append("tts streaming is enabled when the optional VoxCPM stack is available")
        elif runtime.tts_requested:
            notes.append("tts was requested but the optional VoxCPM stack is unavailable, so the backend is running text-only")
        else:
            notes.append("tts is disabled by default in the slim runtime path")
        return {
            "status": "ok",
            "service": "minimind-realtime",
            "ws_path": "/ws/realtime",
            "upstream_ok": upstream_ok,
            "upstream_models": models,
            "llm_base_url": config.llm_base_url,
            "llm_model": config.llm_model,
            "error_message": error_message,
            "asr_model_path": config.asr_model_path,
            "asr_device": config.asr_device,
            "asr_warmed": runtime.asr.is_warmed,
            "vad_path": config.vad_path,
            "tts_requested": runtime.tts_requested,
            "tts_enabled": runtime.tts_enabled,
            "tts_warmed": runtime.tts.is_warmed if runtime.tts is not None else False,
            "tts_error": runtime.tts_error,
            "tts_model_path": config.tts_model_path,
            "tts_lora_root": config.tts_lora_root,
            "memory": runtime.memory.status_dict(),
            "notes": notes,
        }

    @app.post("/warmup")
    async def warmup(payload: dict[str, Any] | None = None):
        request = payload or {}
        force = bool(request.get("force", False))
        started_at = perf_counter()
        tts_options = _build_tts_options(request)
        asr_result = await asyncio.to_thread(runtime.asr.warmup, force)
        tts_result: dict[str, Any] = {}
        if runtime.tts is not None:
            loop = asyncio.get_running_loop()
            tts_result = await loop.run_in_executor(
                runtime.tts_executor, runtime.tts.warmup, tts_options, force
            )
        elif runtime.tts_requested and runtime.tts_error:
            tts_result = {
                "tts_skipped": True,
                "tts_error": runtime.tts_error,
            }
        cached = bool(asr_result.get("cached", False))
        if tts_result:
            cached = cached and bool(tts_result.get("cached", False))
        return {
            "ok": True,
            "cached": cached,
            **asr_result,
            **tts_result,
            "total_ms": round((perf_counter() - started_at) * 1000.0, 2),
        }

    @app.get("/tts/lora/catalog")
    async def tts_lora_catalog():
        if runtime.tts is None:
            if runtime.tts_error:
                raise HTTPException(status_code=503, detail=runtime.tts_error)
            raise HTTPException(status_code=404, detail="TTS is disabled")
        return {"models": runtime.tts.list_lora_checkpoints()}

    @app.websocket("/ws/realtime")
    async def websocket_realtime(websocket: WebSocket) -> None:
        await websocket.accept()
        send_lock = asyncio.Lock()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        session_id = f"session-{uuid.uuid4().hex[:8]}"
        state = ConversationState(session_id=session_id, max_turns=config.max_history_turns)
        vad_session = RealtimeSession(
            config.vad_path,
            VADConfig(
                threshold=config.vad_threshold,
                min_speech_ms=config.vad_min_speech_ms,
                min_silence_ms=config.vad_min_silence_ms,
            ),
        )
        active_task: asyncio.Task | None = None
        interrupt_event = asyncio.Event()
        session_tts_options = TTSRequestOptions()
        turn_capture = TurnCaptureRecorder(
            config.debug_output_dir,
            sample_rate=vad_session.config.sample_rate,
            enabled=config.debug_turns,
            session_id=session_id,
        )

        async def send_json(payload: dict) -> None:
            async with send_lock:
                await websocket.send_json(payload)

        async def receiver() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        queue.put_nowait({"type": "socket.closed"})
                        break
                    if message.get("text") is not None:
                        queue.put_nowait(json.loads(message["text"]))
                    elif message.get("bytes") is not None:
                        queue.put_nowait(message["bytes"])
            except WebSocketDisconnect:
                queue.put_nowait({"type": "socket.closed"})

        async def notify_vad() -> None:
            await send_json({"type": "vad", "speaking": vad_session.speaking})

        async def run_text_request(
            user_text: str,
            turn_id: str | None = None,
            tts_options: TTSRequestOptions | None = None,
            turn_timing: dict[str, float | None] | None = None,
        ) -> None:
            stream = None
            full_text = ""
            request_id = f"req-{uuid.uuid4().hex[:8]}"
            segmenter = StreamingTextSegmenter()
            tts_queue: asyncio.Queue[str | None] = asyncio.Queue()
            request_tts_options = tts_options or session_tts_options
            timing = dict(turn_timing or {})
            speech_finished_at = timing.get("speech_finished_at")
            vad_session.generating = True
            state.active_request_id = request_id
            interrupt_event.clear()

            def log_timing(stage: str, event_at: float | None = None, **fields: Any) -> None:
                _log_turn_timing(
                    session_id,
                    turn_id,
                    stage,
                    event_at=event_at,
                    speech_finished_at=speech_finished_at,
                    assistant_request_id=request_id,
                    **fields,
                )

            async def run_tts_worker() -> None:
                if runtime.tts is None:
                    return
                loop = asyncio.get_running_loop()
                while True:
                    text = await tts_queue.get()
                    try:
                        if text is None or interrupt_event.is_set():
                            return
                        tts_request_id = f"tts-{uuid.uuid4().hex[:8]}"
                        tts_request_started_at = perf_counter()
                        if timing.get("tts_request_started_at") is None:
                            timing["tts_request_started_at"] = tts_request_started_at
                            log_timing(
                                "tts.request_started",
                                tts_request_started_at,
                                tts_request_id=tts_request_id,
                                chunk_chars=len(text),
                                llm_first_text_to_tts_request_ms=_elapsed_ms(
                                    timing.get("llm_first_text_at"),
                                    tts_request_started_at,
                                ),
                                first_segment_to_tts_request_ms=_elapsed_ms(
                                    timing.get("tts_first_segment_ready_at"),
                                    tts_request_started_at,
                                ),
                            )
                        sample_rate = await loop.run_in_executor(
                            runtime.tts_executor, runtime.tts.sample_rate, request_tts_options
                        )
                        await send_json(
                            {
                                "type": "assistant.audio.start",
                                "request_id": tts_request_id,
                                "assistant_request_id": request_id,
                                "turn_id": turn_id,
                                "sample_rate": sample_rate,
                                "audio_format": "pcm_f32le",
                                "text": text,
                            }
                        )
                        audio_start_sent_at = perf_counter()
                        if timing.get("tts_audio_start_sent_at") is None:
                            timing["tts_audio_start_sent_at"] = audio_start_sent_at
                            log_timing(
                                "tts.audio_start_sent",
                                audio_start_sent_at,
                                tts_request_id=tts_request_id,
                                tts_request_to_audio_start_ms=_elapsed_ms(
                                    timing.get("tts_request_started_at"),
                                    audio_start_sent_at,
                                ),
                            )
                        audio_stream = runtime.tts.stream_tts(text, options=request_tts_options)
                        chunk_index = 0
                        total_samples = 0
                        captured_chunks: list[np.ndarray] | None = [] if turn_capture.enabled else None
                        while not interrupt_event.is_set():
                            done, audio = await loop.run_in_executor(
                                runtime.tts_executor, _next_stream_item, audio_stream
                            )
                            if done:
                                break
                            encoded, num_samples = _encode_pcm_f32(audio)
                            total_samples += num_samples
                            if captured_chunks is not None:
                                captured_chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                            await send_json(
                                {
                                    "type": "assistant.audio.chunk",
                                    "request_id": tts_request_id,
                                    "assistant_request_id": request_id,
                                    "turn_id": turn_id,
                                    "index": chunk_index,
                                    "sample_rate": sample_rate,
                                    "audio_format": "pcm_f32le",
                                    "num_samples": num_samples,
                                    "data": encoded,
                                }
                            )
                            chunk_sent_at = perf_counter()
                            if timing.get("tts_first_audio_chunk_at") is None:
                                timing["tts_first_audio_chunk_at"] = chunk_sent_at
                                log_timing(
                                    "tts.first_audio_chunk_sent",
                                    chunk_sent_at,
                                    tts_request_id=tts_request_id,
                                    num_samples=num_samples,
                                    tts_request_to_first_audio_ms=_elapsed_ms(
                                        timing.get("tts_request_started_at"),
                                        chunk_sent_at,
                                    ),
                                )
                                log_timing(
                                    "turn.first_audio_summary",
                                    chunk_sent_at,
                                    # vad_tail_ms: user-finished-speech -> VAD speech_end.
                                    vad_tail_ms=timing.get("vad_tail_ms"),
                                    # speech_end_to_turn_start_ms: VAD speech_end -> backend
                                    # turn task actually starts.
                                    speech_end_to_turn_start_ms=_elapsed_ms(
                                        timing.get("speech_end_detected_at"),
                                        timing.get("turn_started_at"),
                                    ),
                                    # asr_ms: ASR transcription duration.
                                    asr_ms=_elapsed_ms(
                                        timing.get("asr_started_at"),
                                        timing.get("asr_completed_at"),
                                    ),
                                    asr_done_to_llm_request_ms=_elapsed_ms(
                                        timing.get("asr_completed_at"),
                                        timing.get("llm_request_started_at"),
                                    ),
                                    memory_search_ms=_elapsed_ms(
                                        timing.get("memory_search_started_at"),
                                        timing.get("memory_search_completed_at"),
                                    ),
                                    # llm_request_to_first_text_ms: LLM request start ->
                                    # first speakable text delta from the model.
                                    llm_request_to_first_text_ms=_elapsed_ms(
                                        timing.get("llm_request_started_at"),
                                        timing.get("llm_first_text_at"),
                                    ),
                                    # llm_first_text_to_first_segment_ms: first text delta ->
                                    # first segment that the segmenter is willing to hand to TTS.
                                    llm_first_text_to_first_segment_ms=_elapsed_ms(
                                        timing.get("llm_first_text_at"),
                                        timing.get("tts_first_segment_ready_at"),
                                    ),
                                    first_segment_to_tts_request_ms=_elapsed_ms(
                                        timing.get("tts_first_segment_ready_at"),
                                        timing.get("tts_request_started_at"),
                                    ),
                                    tts_request_to_audio_start_ms=_elapsed_ms(
                                        timing.get("tts_request_started_at"),
                                        timing.get("tts_audio_start_sent_at"),
                                    ),
                                    # tts_request_to_first_audio_ms: TTS request start ->
                                    # first emitted audio chunk.
                                    tts_request_to_first_audio_ms=_elapsed_ms(
                                        timing.get("tts_request_started_at"),
                                        chunk_sent_at,
                                    ),
                                    speech_end_to_first_audio_ms=_elapsed_ms(
                                        timing.get("speech_end_detected_at"),
                                        chunk_sent_at,
                                    ),
                                    # total_to_first_audio_ms: user-finished-speech estimate ->
                                    # first emitted audio chunk.
                                    total_to_first_audio_ms=_elapsed_ms(
                                        speech_finished_at,
                                        chunk_sent_at,
                                    ),
                                )
                            chunk_index += 1
                        if captured_chunks:
                            full_audio = np.concatenate(captured_chunks)
                            await asyncio.to_thread(
                                turn_capture.capture_tts_turn,
                                turn_id or request_id,
                                full_audio,
                                text,
                                request_id=tts_request_id,
                                sample_rate=sample_rate,
                                interrupted=interrupt_event.is_set(),
                                extra={"chunk_count": chunk_index, "assistant_request_id": request_id},
                            )
                        await send_json(
                            {
                                "type": "assistant.audio.completed",
                                "request_id": tts_request_id,
                                "assistant_request_id": request_id,
                                "turn_id": turn_id,
                                "chunk_count": chunk_index,
                                "total_samples": total_samples,
                                "interrupted": interrupt_event.is_set(),
                            }
                        )
                    except Exception as exc:
                        logger.exception(
                            "tts generation failed | request_id=%s | turn_id=%s | text=%r",
                            request_id,
                            turn_id,
                            text,
                        )
                        await send_json(
                            {
                                "type": "assistant.audio.error",
                                "assistant_request_id": request_id,
                                "turn_id": turn_id,
                                "message": str(exc),
                            }
                        )
                    finally:
                        tts_queue.task_done()

            tts_task = asyncio.create_task(run_tts_worker())
            await send_json({"type": "assistant.started", "request_id": request_id, "turn_id": turn_id})
            try:
                memory_context = ""
                if runtime.memory.available:
                    timing["memory_search_started_at"] = perf_counter()
                    memory_context = await asyncio.to_thread(runtime.memory.search_context, user_text)
                    timing["memory_search_completed_at"] = perf_counter()
                    log_timing(
                        "memory.search_completed",
                        timing["memory_search_completed_at"],
                        memory_search_ms=_elapsed_ms(
                            timing.get("memory_search_started_at"),
                            timing.get("memory_search_completed_at"),
                        ),
                        memory_context_found=bool(memory_context),
                    )
                    if memory_context:
                        await send_json(
                            {
                                "type": "memory.context",
                                "request_id": request_id,
                                "turn_id": turn_id,
                                "available": True,
                            }
                        )
                messages = state.build_messages(user_text)
                timing["llm_request_started_at"] = perf_counter()
                log_timing("llm.request_started", timing["llm_request_started_at"])
                stream = runtime.llm.start_stream(messages, memory_context=memory_context)
                reasoning_text = ""
                while True:
                    done, delta = await asyncio.to_thread(_next_stream_item, stream)
                    if done:
                        break
                    if interrupt_event.is_set():
                        break
                    text_delta = str(getattr(delta, "text", "") or "")
                    reasoning_delta = str(getattr(delta, "reasoning_text", "") or "")
                    if reasoning_delta:
                        reasoning_text += reasoning_delta
                    if not text_delta:
                        continue
                    if timing.get("llm_first_text_at") is None:
                        timing["llm_first_text_at"] = perf_counter()
                        log_timing(
                            "llm.first_text_delta",
                            timing["llm_first_text_at"],
                            llm_request_to_first_text_ms=_elapsed_ms(
                                timing.get("llm_request_started_at"),
                                timing.get("llm_first_text_at"),
                            ),
                            text_chars=len(text_delta),
                        )
                    full_text += text_delta
                    await send_json({"type": "assistant.delta", "request_id": request_id, "turn_id": turn_id, "text": text_delta})
                    for chunk in segmenter.feed(text_delta):
                        if timing.get("tts_first_segment_ready_at") is None:
                            timing["tts_first_segment_ready_at"] = perf_counter()
                            log_timing(
                                "tts.first_segment_ready",
                                timing["tts_first_segment_ready_at"],
                                chunk_chars=len(chunk),
                                llm_request_to_first_segment_ms=_elapsed_ms(
                                    timing.get("llm_request_started_at"),
                                    timing.get("tts_first_segment_ready_at"),
                                ),
                                llm_first_text_to_first_segment_ms=_elapsed_ms(
                                    timing.get("llm_first_text_at"),
                                    timing.get("tts_first_segment_ready_at"),
                                ),
                            )
                        await tts_queue.put(chunk)
                if not full_text.strip() and reasoning_text.strip() and not interrupt_event.is_set():
                    await send_json(
                        {
                            "type": "assistant.error",
                            "request_id": request_id,
                            "turn_id": turn_id,
                            "message": "LLM produced reasoning tokens but no speakable text. Thinking mode may still be enabled.",
                        }
                    )
                    await tts_queue.put(None)
                    await tts_task
                    return
                final_tts_chunk = segmenter.flush()
                if final_tts_chunk:
                    if timing.get("tts_first_segment_ready_at") is None:
                        timing["tts_first_segment_ready_at"] = perf_counter()
                        log_timing(
                            "tts.first_segment_ready",
                            timing["tts_first_segment_ready_at"],
                            chunk_chars=len(final_tts_chunk),
                            llm_request_to_first_segment_ms=_elapsed_ms(
                                timing.get("llm_request_started_at"),
                                timing.get("tts_first_segment_ready_at"),
                            ),
                            llm_first_text_to_first_segment_ms=_elapsed_ms(
                                timing.get("llm_first_text_at"),
                                timing.get("tts_first_segment_ready_at"),
                            ),
                            flushed_at_stream_end=True,
                        )
                    await tts_queue.put(final_tts_chunk)
                await tts_queue.put(None)
                await tts_task
                state.add_user_turn(user_text)
                state.add_assistant_turn(full_text)
                if interrupt_event.is_set():
                    await send_json(
                        {
                            "type": "assistant.completed",
                            "request_id": request_id,
                            "turn_id": turn_id,
                            "interrupted": True,
                            "text": full_text,
                        }
                    )
                    return
                await send_json(
                    {
                        "type": "assistant.completed",
                        "request_id": request_id,
                        "turn_id": turn_id,
                        "interrupted": False,
                        "text": full_text,
                    }
                )
                if runtime.memory.available:
                    async def save_memory_turn() -> None:
                        result = await asyncio.to_thread(
                            runtime.memory.save_turn,
                            user_text,
                            full_text,
                            state.session_id,
                        )
                        with contextlib.suppress(Exception):
                            await send_json(
                                {
                                    "type": "memory.saved",
                                    "request_id": request_id,
                                    "turn_id": turn_id,
                                    "saved": bool(result),
                                }
                            )

                    asyncio.create_task(save_memory_turn())
            except Exception as exc:
                await send_json({"type": "assistant.error", "request_id": request_id, "turn_id": turn_id, "message": str(exc)})
            finally:
                if not tts_task.done():
                    await tts_queue.put(None)
                    tts_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await tts_task
                if stream is not None:
                    runtime.llm.close_stream(stream)
                vad_session.generating = False
                vad_session.interrupt = False
                state.active_request_id = None

        async def run_audio_turn(
            samples: np.ndarray,
            vad_timing: VADTurnTiming | None = None,
            tts_options: TTSRequestOptions | None = None,
        ) -> None:
            handed_off_to_llm = False
            turn_id = f"turn-{uuid.uuid4().hex[:8]}"
            user_text = ""
            capture_error: str | None = None
            timing: dict[str, float | None] = {}
            speech_finished_at = None
            if vad_timing is not None:
                speech_finished_at = vad_timing.speech_finished_at_est
                timing["speech_finished_at"] = speech_finished_at
                timing["speech_end_detected_at"] = vad_timing.speech_end_at
                timing["vad_tail_ms"] = vad_timing.vad_tail_ms

            def log_timing(stage: str, event_at: float | None = None, **fields: Any) -> None:
                _log_turn_timing(
                    session_id,
                    turn_id,
                    stage,
                    event_at=event_at,
                    speech_finished_at=speech_finished_at,
                    **fields,
                )

            try:
                vad_session.generating = True
                timing["turn_started_at"] = perf_counter()
                log_timing(
                    "turn.started",
                    timing["turn_started_at"],
                    vad_tail_ms=timing.get("vad_tail_ms"),
                    speech_duration_ms=vad_timing.speech_duration_ms if vad_timing is not None else None,
                    buffered_audio_ms=vad_timing.buffered_audio_ms if vad_timing is not None else None,
                    speech_end_to_turn_start_ms=_elapsed_ms(
                        timing.get("speech_end_detected_at"),
                        timing.get("turn_started_at"),
                    ),
                )
                timing["asr_started_at"] = perf_counter()
                log_timing(
                    "asr.started",
                    timing["asr_started_at"],
                    speech_end_to_asr_start_ms=_elapsed_ms(
                        timing.get("speech_end_detected_at"),
                        timing.get("asr_started_at"),
                    ),
                )
                await send_json({"type": "asr.started", "turn_id": turn_id})
                user_text = await asyncio.to_thread(runtime.asr.transcribe, samples)
                timing["asr_completed_at"] = perf_counter()
                log_timing(
                    "asr.completed",
                    timing["asr_completed_at"],
                    asr_ms=_elapsed_ms(
                        timing.get("asr_started_at"),
                        timing.get("asr_completed_at"),
                    ),
                    user_text_chars=len(user_text),
                )
                if interrupt_event.is_set():
                    return
                if not user_text:
                    await send_json({"type": "asr.completed", "turn_id": turn_id, "text": ""})
                    await send_json({"type": "user_text", "turn_id": turn_id, "text": ""})
                    return
                await send_json({"type": "asr.completed", "turn_id": turn_id, "text": user_text})
                await send_json({"type": "user_text", "turn_id": turn_id, "text": user_text})
                handed_off_to_llm = True
                await run_text_request(
                    user_text,
                    turn_id=turn_id,
                    tts_options=tts_options,
                    turn_timing=timing,
                )
            except Exception as exc:
                capture_error = str(exc)
                vad_session.generating = False
                await send_json({"type": "asr.error", "turn_id": turn_id, "message": str(exc)})
            finally:
                if turn_capture.enabled:
                    await asyncio.to_thread(
                        turn_capture.capture_user_turn,
                        turn_id,
                        samples,
                        user_text,
                        interrupted=interrupt_event.is_set(),
                        error_message=capture_error,
                    )
                if not handed_off_to_llm:
                    vad_session.generating = False

        def push_audio(audio_bytes: bytes) -> str:
            return vad_session.push_chunk(_decode_pcm16(audio_bytes))

        await send_json(
            {
                "type": "server.ready",
                "protocol_version": "0.1.0",
                "session_id": session_id,
                "supported_client_messages": [
                    "session.start",
                    "text.input",
                    "tts.configure",
                    "logging.configure",
                    "audio.chunk",
                    "interrupt",
                    "ping",
                    "session.stop",
                ],
            }
        )

        receiver_task = asyncio.create_task(receiver())
        try:
            while True:
                message = await queue.get()

                if isinstance(message, (bytes, bytearray)):
                    if active_task is not None and active_task.done():
                        active_task = None
                    status = push_audio(bytes(message))
                    await notify_vad()
                    if status == "interrupt":
                        if not interrupt_event.is_set():
                            interrupt_event.set()
                            await send_json(
                                {
                                    "type": "interrupt.ack",
                                    "session_id": session_id,
                                    "request_id": f"interrupt-{uuid.uuid4().hex[:8]}",
                                    "interrupted_request_id": state.active_request_id,
                                    "accepted": bool(active_task and not active_task.done()),
                                    "reason": "vad_barge_in",
                                }
                            )
                        continue
                    if status != "speech_end":
                        continue
                    if active_task is not None and not active_task.done():
                        if interrupt_event.is_set():
                            try:
                                await asyncio.wait_for(active_task, timeout=5.0)
                            except asyncio.TimeoutError:
                                active_task.cancel()
                                with contextlib.suppress(asyncio.CancelledError, Exception):
                                    await active_task
                            finally:
                                active_task = None
                        else:
                            await send_json({"type": "error", "message": "assistant request is still running"})
                            continue
                    samples = vad_session.get_audio()
                    vad_timing = vad_session.consume_completed_timing()
                    if samples.size == 0:
                        continue
                    interrupt_event.clear()
                    vad_session.interrupt = False
                    active_task = asyncio.create_task(
                        run_audio_turn(samples, vad_timing=vad_timing, tts_options=session_tts_options)
                    )
                    continue

                if not isinstance(message, dict):
                    await send_json({"type": "error", "message": "received unsupported queue payload"})
                    continue

                message_type = str(message.get("type") or "").strip()

                if message_type == "socket.closed":
                    break

                if message_type == "ping":
                    await send_json({"type": "pong", "session_id": session_id})
                    continue

                if message_type == "session.start":
                    requested = str(message.get("session_id") or "").strip()
                    if requested:
                        session_id = requested
                        state.session_id = requested
                        turn_capture.set_session_id(requested)
                    session_tts_options = _build_tts_options(message, session_tts_options)
                    turn_capture.set_enabled(_bool_flag(message.get("logging_enabled"), config.debug_turns))
                    await send_json({"type": "session.ready", "session_id": session_id, "state": "ready"})
                    await send_json(
                        {
                            "type": "logging.configured",
                            "session_id": session_id,
                            "enabled": turn_capture.enabled,
                            "capture_dir": config.debug_output_dir,
                        }
                    )
                    continue

                if message_type == "tts.configure":
                    session_tts_options = _build_tts_options(message, session_tts_options)
                    await send_json(
                        {
                            "type": "tts.configured",
                            "session_id": session_id,
                            "model_version": session_tts_options.model_version,
                            "lora_selection": session_tts_options.lora_selection,
                            "reference_enabled": bool(session_tts_options.prompt_audio_path and session_tts_options.prompt_text),
                            "reference_wav_enabled": bool(session_tts_options.reference_wav_path),
                            "reference_ignored": bool(session_tts_options.prompt_audio_path and not session_tts_options.prompt_text),
                            "cfg_value": session_tts_options.cfg_value,
                            "inference_timesteps": session_tts_options.inference_timesteps,
                            "seed": session_tts_options.seed,
                        }
                    )
                    continue

                if message_type == "logging.configure":
                    turn_capture.set_enabled(_bool_flag(message.get("enabled"), False))
                    await send_json(
                        {
                            "type": "logging.configured",
                            "session_id": session_id,
                            "enabled": turn_capture.enabled,
                            "capture_dir": config.debug_output_dir,
                        }
                    )
                    continue

                if message_type == "session.stop":
                    interrupt_event.set()
                    if active_task is not None:
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await active_task
                    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                        await send_json({"type": "session.closed", "session_id": session_id})
                    break

                if message_type == "interrupt":
                    interrupt_event.set()
                    vad_session.interrupt = True
                    await send_json(
                        {
                            "type": "interrupt.ack",
                            "session_id": session_id,
                            "request_id": message.get("request_id"),
                            "interrupted_request_id": state.active_request_id,
                            "accepted": bool(state.active_request_id),
                        }
                    )
                    continue

                if message_type == "audio.chunk":
                    await send_json({"type": "error", "message": "send raw websocket binary frames for audio, not JSON audio.chunk"})
                    continue

                if message_type == "text.input":
                    text = str(message.get("text") or "").strip()
                    if not text:
                        await send_json({"type": "error", "message": "text.input requires a non-empty text field"})
                        continue
                    if active_task is not None and not active_task.done():
                        await send_json({"type": "error", "message": "assistant request is still running"})
                        continue
                    request_options = _build_tts_options(message, session_tts_options)
                    active_task = asyncio.create_task(run_text_request(text, tts_options=request_options))
                    continue

                await send_json({"type": "error", "message": f"unsupported message type: {message_type}"})
                continue
        finally:
            interrupt_event.set()
            if active_task is not None:
                active_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await active_task
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await receiver_task

    return app


def parse_args() -> RealtimeAppConfig:
    parser = argparse.ArgumentParser(description="MiniMind realtime backend scaffold")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8011, type=int)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--max-history-turns", default=6, type=int)
    parser.add_argument("--asr-model-path", default=str(REPO_ROOT / "model" / "SenseVoiceSmall"))
    parser.add_argument("--asr-device", default="cuda")
    parser.add_argument("--vad-path", default=str(REPO_ROOT / "model" / "vad" / "silero_vad.onnx"))
    parser.add_argument("--vad-threshold", default=0.8, type=float)
    parser.add_argument("--vad-min-speech-ms", default=128, type=int)
    parser.add_argument("--vad-min-silence-ms", default=800, type=int)
    tts_group = parser.add_mutually_exclusive_group()
    tts_group.add_argument("--enable-tts", action="store_true")
    tts_group.add_argument("--disable-tts", action="store_true")
    parser.add_argument("--tts-model-path", default=None)
    parser.add_argument("--tts-lora-root", default=None)
    parser.add_argument("--debug-turns", action="store_true")
    parser.add_argument("--debug-output-dir", default=None)
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--enable-memory", action="store_true")
    memory_group.add_argument("--disable-memory", action="store_true")
    args = parser.parse_args()
    memory_enabled = _bool_flag(os.getenv("MINIMIND_MEMORY_ENABLED"), False)
    if args.enable_memory:
        memory_enabled = True
    if args.disable_memory:
        memory_enabled = False
    return RealtimeAppConfig(
        host=args.host,
        port=args.port,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        max_history_turns=args.max_history_turns,
        asr_model_path=args.asr_model_path,
        asr_device=args.asr_device,
        vad_path=args.vad_path,
        vad_threshold=args.vad_threshold,
        vad_min_speech_ms=args.vad_min_speech_ms,
        vad_min_silence_ms=args.vad_min_silence_ms,
        tts_enabled=bool(args.enable_tts and not args.disable_tts),
        tts_model_path=args.tts_model_path,
        tts_lora_root=args.tts_lora_root,
        debug_turns=bool(args.debug_turns),
        debug_output_dir=args.debug_output_dir,
        memory_enabled=memory_enabled,
    )


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run realtime/app.py") from exc

    config = parse_args()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
