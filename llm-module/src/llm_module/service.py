from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from .llm import (
    CHAT_COMPATIBLE_PROMPT_PRESET,
    DEFAULT_PROMPT_PRESET,
    LLMConfig,
    LlamaServerLLM,
    VOICE_STRICT_PROMPT_PRESET,
    resolve_system_prompt,
)


WEBSOCKET_PATH = "/ws/llm"


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _to_optional_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in {None, ""}:
        return None if default is None else int(default)
    return int(value)


def _to_optional_text(value: Any, default: Optional[str] = None) -> Optional[str]:
    candidate = default if value in {None, ""} else value
    if candidate is None:
        return None
    text = str(candidate).strip()
    return text or None


def _normalize_prompt_preset(value: Any, default: str = DEFAULT_PROMPT_PRESET) -> str:
    preset = str(value or default).strip().lower()
    return preset or default


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
class LLMServiceConfig:
    host: str = "127.0.0.1"
    port: int = 8004
    upstream_base_url: str = "http://127.0.0.1:8080"
    upstream_model: Optional[str] = None
    api_key: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt_preset: str = DEFAULT_PROMPT_PRESET
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    timeout: float = 600.0
    reasoning_budget: Optional[int] = None
    reasoning_format: Optional[str] = None
    log_level: str = "info"


class LLMService:
    def __init__(self, config: LLMServiceConfig):
        self.config = config
        resolved_system_prompt = resolve_system_prompt(config.system_prompt, config.prompt_preset)
        self.llm = LlamaServerLLM(
            LLMConfig(
                base_url=config.upstream_base_url,
                model=config.upstream_model,
                system_prompt=resolved_system_prompt,
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                timeout=config.timeout,
                api_key=config.api_key,
                reasoning_budget=config.reasoning_budget,
                reasoning_format=config.reasoning_format,
            )
        )

    def list_models(self) -> list[str]:
        return self.llm.list_models()

    def resolve_model(self, override_model: Optional[str] = None) -> str:
        return self.llm.resolve_model(override_model)

    def generate_stream(self, request: Dict[str, Any]) -> tuple[str, Iterable[Any]]:
        messages = request.get("messages")
        text = str(request.get("text", "") or "").strip()
        if messages is None and not text:
            raise ValueError("'text' must be non-empty when 'messages' is not supplied")

        prompt_preset = _normalize_prompt_preset(request.get("prompt_preset"), self.config.prompt_preset)
        system_prompt = resolve_system_prompt(
            request.get("system_prompt") if request.get("system_prompt") not in {None, ""} else self.config.system_prompt,
            prompt_preset,
        )
        reasoning_budget = _to_optional_int(request.get("reasoning_budget"), self.config.reasoning_budget)
        reasoning_format = _to_optional_text(request.get("reasoning_format"), self.config.reasoning_format)
        resolved_model = self.resolve_model(request.get("model"))
        stream = self.llm.generate_stream(
            text=text,
            model=resolved_model,
            system_prompt=system_prompt,
            temperature=_to_float(request.get("temperature"), self.config.temperature),
            top_p=_to_float(request.get("top_p"), self.config.top_p),
            max_tokens=_to_int(request.get("max_tokens"), self.config.max_tokens),
            messages=messages,
            reasoning_budget=reasoning_budget,
            reasoning_format=reasoning_format,
        )
        return resolved_model, stream

    def generate_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        model, stream = self.generate_stream(request)
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        started_at = time.perf_counter()
        count = 0
        try:
            for delta in stream:
                text_parts.append(str(getattr(delta, "text", "") or ""))
                reasoning_parts.append(str(getattr(delta, "reasoning_text", "") or ""))
                count += 1
        finally:
            _close_stream(stream)
        return {
            "model": model,
            "text": "".join(text_parts),
            "reasoning": "".join(reasoning_parts).strip() or None,
            "delta_count": count,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        }


def create_app(config: Optional[LLMServiceConfig] = None, service: Optional[LLMService] = None):
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import StreamingResponse
    except ImportError as exc:  # pragma: no cover - depends on optional server dependencies
        raise RuntimeError("server dependencies are missing; install with: pip install -e .[server]") from exc

    config = config or LLMServiceConfig()
    service = service or LLMService(config)
    app = FastAPI(title="LLM Module", version="0.1.0")
    app.state.llm_service = service

    @app.get("/")
    async def root() -> Dict[str, Any]:
        return {
            "service": "llm-module",
            "websocket_path": WEBSOCKET_PATH,
            "upstream_base_url": config.upstream_base_url,
            "upstream_model": config.upstream_model,
        }

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        upstream_ok = True
        upstream_models: list[str] = []
        error_message: Optional[str] = None
        try:
            upstream_models = service.list_models()
        except Exception as exc:  # pragma: no cover - depends on external upstream server
            upstream_ok = False
            error_message = str(exc)

        return {
            "status": "ok",
            "service": "llm-module",
            "upstream_base_url": config.upstream_base_url,
            "upstream_model": config.upstream_model,
            "upstream_ok": upstream_ok,
            "upstream_models": upstream_models,
            "error_message": error_message,
        }

    @app.get("/upstream/models")
    async def upstream_models() -> Dict[str, Any]:
        models = service.list_models()
        return {"count": len(models), "models": models}

    @app.post("/chat")
    async def chat(request: Dict[str, Any]) -> Dict[str, Any]:
        return service.generate_text(request)

    @app.post("/chat/stream")
    async def chat_stream(request: Dict[str, Any]) -> StreamingResponse:
        def event_iter():
            model, stream = service.generate_stream(request)
            try:
                yield f"data: {json.dumps({'type': 'llm.started', 'model': model}, ensure_ascii=False)}\n\n"
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                count = 0
                started_at = time.perf_counter()
                for delta in stream:
                    delta_text = str(getattr(delta, "text", "") or "")
                    reasoning_text = str(getattr(delta, "reasoning_text", "") or "")
                    text_parts.append(delta_text)
                    reasoning_parts.append(reasoning_text)
                    payload = {
                        "type": "llm.delta",
                        "index": int(getattr(delta, "index", count)),
                        "text": delta_text,
                        "reasoning": reasoning_text or None,
                        "model": str(getattr(delta, "model", model) or model),
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    count += 1
                completed = {
                    "type": "llm.completed",
                    "model": model,
                    "text": "".join(text_parts),
                    "reasoning": "".join(reasoning_parts).strip() or None,
                    "delta_count": count,
                    "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
                }
                yield f"data: {json.dumps(completed, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                _close_stream(stream)

        return StreamingResponse(event_iter(), media_type="text/event-stream")

    @app.websocket(WEBSOCKET_PATH)
    async def websocket_llm(websocket: WebSocket) -> None:
        await websocket.accept()
        session_id: Optional[str] = None
        await websocket.send_json(
            {
                "type": "server.ready",
                "protocol_version": "0.1.0",
                "websocket_path": WEBSOCKET_PATH,
                "supported_client_messages": ["session.start", "chat.text", "ping"],
            }
        )

        try:
            while True:
                message = await websocket.receive_json()
                message_type = message.get("type")
                if message_type == "ping":
                    await websocket.send_json({"type": "pong", "session_id": session_id})
                    continue

                if message_type == "session.start":
                    session_id = str(message.get("session_id") or f"session-{uuid.uuid4().hex[:8]}")
                    await websocket.send_json(
                        {
                            "type": "session.ready",
                            "session_id": session_id,
                            "state": "ready",
                            "request_count": 0,
                            "active_request_id": None,
                        }
                    )
                    continue

                if message_type != "chat.text":
                    await websocket.send_json(
                        {
                            "type": "error",
                            "session_id": session_id,
                            "request_id": message.get("request_id"),
                            "message": "unsupported message type, expected 'chat.text'",
                        }
                    )
                    continue

                session_id = session_id or str(message.get("session_id") or f"session-{uuid.uuid4().hex[:8]}")
                request_id = str(message.get("request_id") or f"req-{uuid.uuid4().hex[:8]}")
                model, stream = service.generate_stream(message)
                text_parts: list[str] = []
                reasoning_parts: list[str] = []
                started_at = time.perf_counter()
                count = 0

                await websocket.send_json(
                    {"type": "llm.started", "session_id": session_id, "request_id": request_id, "model": model}
                )
                try:
                    while True:
                        done, delta = await asyncio.to_thread(_next_stream_item, stream)
                        if done:
                            break
                        delta_text = str(getattr(delta, "text", "") or "")
                        reasoning_text = str(getattr(delta, "reasoning_text", "") or "")
                        text_parts.append(delta_text)
                        reasoning_parts.append(reasoning_text)
                        await websocket.send_json(
                            {
                                "type": "llm.delta",
                                "session_id": session_id,
                                "request_id": request_id,
                                "index": int(getattr(delta, "index", count)),
                                "text": delta_text,
                                "reasoning": reasoning_text or None,
                                "model": str(getattr(delta, "model", model) or model),
                            }
                        )
                        count += 1
                finally:
                    await asyncio.to_thread(_close_stream, stream)

                await websocket.send_json(
                    {
                        "type": "llm.completed",
                        "session_id": session_id,
                        "request_id": request_id,
                        "model": model,
                        "text": "".join(text_parts),
                        "reasoning": "".join(reasoning_parts).strip() or None,
                        "delta_count": count,
                        "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
                    }
                )
        except WebSocketDisconnect:
            return

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight LLM proxy service")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8004, help="Port to bind")
    parser.add_argument("--upstream-base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument("--model", default=None, help="Optional upstream model alias/id")
    parser.add_argument("--api-key", default=None, help="Optional upstream API key")
    parser.add_argument("--system-prompt", default=None, help="Optional custom system prompt")
    parser.add_argument(
        "--prompt-preset",
        default=DEFAULT_PROMPT_PRESET,
        choices=[CHAT_COMPATIBLE_PROMPT_PRESET, VOICE_STRICT_PROMPT_PRESET],
        help="Prompt preset used when no custom system prompt is provided",
    )
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum output tokens")
    parser.add_argument("--timeout", type=float, default=600.0, help="Upstream request timeout in seconds")
    parser.add_argument("--reasoning-budget", type=int, default=None, help="Optional default reasoning budget")
    parser.add_argument("--reasoning-format", default=None, help="Optional default reasoning format")
    parser.add_argument("--log-level", type=str, default="info", help="uvicorn log level")
    return parser


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional server dependencies
        raise RuntimeError("uvicorn is missing; install with: pip install -e .[server]") from exc

    args = build_arg_parser().parse_args()
    config = LLMServiceConfig(
        host=args.host,
        port=args.port,
        upstream_base_url=args.upstream_base_url,
        upstream_model=args.model,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        prompt_preset=args.prompt_preset,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        reasoning_budget=args.reasoning_budget,
        reasoning_format=args.reasoning_format,
        log_level=args.log_level,
    )
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
