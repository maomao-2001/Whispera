import argparse
import asyncio
import json
import time
import uuid

import websockets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WebSocket verification client for llm-module")
    parser.add_argument("--url", default="ws://127.0.0.1:8004/ws/llm", help="WebSocket LLM service URL")
    parser.add_argument("--text", required=True, help="Input text sent as chat.text")
    parser.add_argument("--session-id", default=None, help="Optional session id")
    parser.add_argument("--request-id", default=None, help="Optional request id")
    parser.add_argument("--model", default=None, help="Optional upstream model alias/id")
    parser.add_argument("--system-prompt", default=None, help="Optional system prompt override")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens")
    return parser


async def run_client(args: argparse.Namespace) -> int:
    session_id = args.session_id or f"llm-demo-{uuid.uuid4().hex[:8]}"
    request_id = args.request_id or f"req-llm-{uuid.uuid4().hex[:8]}"
    payload = {
        "type": "chat.text",
        "session_id": session_id,
        "request_id": request_id,
        "text": args.text,
        "model": args.model,
        "system_prompt": args.system_prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }

    full_text_parts: list[str] = []
    first_delta_latency_ms = None
    start_time = time.perf_counter()

    async with websockets.connect(args.url, max_size=None) as websocket:
        greeting = json.loads(await websocket.recv())
        print(f"[llm-client] server greeting: {greeting.get('type')}")

        await websocket.send(json.dumps({"type": "session.start", "session_id": session_id}, ensure_ascii=False))
        while True:
            message = json.loads(await websocket.recv())
            if message.get("type") == "session.ready":
                print(f"[llm-client] session ready | session_id={message.get('session_id')}")
                break

        await websocket.send(json.dumps(payload, ensure_ascii=False))

        while True:
            message = json.loads(await websocket.recv())
            message_type = message.get("type")

            if message_type == "llm.started":
                print(f"[llm-client] started | model={message.get('model')}")
                continue

            if message_type == "llm.delta":
                if first_delta_latency_ms is None:
                    first_delta_latency_ms = (time.perf_counter() - start_time) * 1000.0
                delta_text = str(message.get("text") or "")
                full_text_parts.append(delta_text)
                print(delta_text, end="", flush=True)
                continue

            if message_type == "llm.completed":
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                full_text = "".join(full_text_parts)
                print()
                print(f"[llm-client] completed | delta_count={message.get('delta_count')}")
                if first_delta_latency_ms is not None:
                    print(f"[llm-client] first_delta_latency_ms={first_delta_latency_ms:.2f}")
                print(f"[llm-client] elapsed_ms={elapsed_ms:.2f}")
                print(f"[llm-client] full_text={full_text}")
                return 0

            if message_type == "error":
                raise RuntimeError(message.get("message", "unknown LLM server error"))

            print(f"[llm-client] ignored message: {message_type}")


def main() -> None:
    args = build_arg_parser().parse_args()
    raise SystemExit(asyncio.run(run_client(args)))


if __name__ == "__main__":
    main()
