import argparse
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from llm_module import DEFAULT_SYSTEM_PROMPT, LLMConfig, LlamaServerLLM


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct llama-server verification demo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="llama-server base URL")
    parser.add_argument("--model", default=None, help="Optional upstream model alias/id")
    parser.add_argument("--text", required=False, default="你好，简单介绍一下你自己。", help="Input text for the LLM")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=256, help="Maximum output tokens")
    parser.add_argument("--timeout", type=float, default=600.0, help="Request timeout in seconds")
    parser.add_argument("--list-models", action="store_true", help="List upstream models and exit")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    llm = LlamaServerLLM(
        LLMConfig(
            base_url=args.base_url,
            model=args.model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
        )
    )

    if args.list_models:
        models = llm.list_models()
        print(f"[llm-demo] models={models}")
        return

    start_time = time.perf_counter()
    resolved_model = llm.resolve_model(args.model)
    print(f"[llm-demo] base_url={args.base_url} | model={resolved_model}")

    parts: list[str] = []
    first_delta_latency_ms = None

    for delta in llm.generate_stream(
        text=args.text,
        model=resolved_model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    ):
        if first_delta_latency_ms is None:
            first_delta_latency_ms = (time.perf_counter() - start_time) * 1000.0
        parts.append(delta.text)
        print(delta.text, end="", flush=True)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    full_text = "".join(parts)
    print()
    if first_delta_latency_ms is not None:
        print(f"[llm-demo] first_delta_latency_ms={first_delta_latency_ms:.2f}")
    print(f"[llm-demo] elapsed_ms={elapsed_ms:.2f}")
    print(f"[llm-demo] full_text={full_text}")


if __name__ == "__main__":
    main()
