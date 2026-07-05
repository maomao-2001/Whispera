from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .local_modules import ensure_llm_module_on_path


ensure_llm_module_on_path()

from llm_module import LLMConfig, LlamaServerLLM  # type: ignore  # noqa: E402
from llm_module.llm import VOICE_STRICT_PROMPT_PRESET, resolve_system_prompt  # type: ignore  # noqa: E402


@dataclass
class LLMRuntimeConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str | None = None
    system_prompt: str | None = None
    temperature: float = 0.65
    top_p: float = 0.88
    max_tokens: int = 256
    timeout: float = 200.0
    reasoning_budget: int | None = 0
    enable_thinking: bool | None = False


class LocalLLMClient:
    def __init__(self, config: LLMRuntimeConfig | None = None):
        self.config = config or LLMRuntimeConfig()
        self._client = LlamaServerLLM(
            LLMConfig(
                base_url=self.config.base_url,
                model=self.config.model,
                system_prompt=resolve_system_prompt(self.config.system_prompt, VOICE_STRICT_PROMPT_PRESET),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout,
                reasoning_budget=self.config.reasoning_budget,
                enable_thinking=self.config.enable_thinking,
            )
        )

    def _resolve_system_prompt(self) -> str:
        return resolve_system_prompt(self.config.system_prompt, VOICE_STRICT_PROMPT_PRESET)

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def start_stream(self, messages: Iterable[dict[str, str]], memory_context: str | None = None):
        system_prompt = self._resolve_system_prompt()
        context = str(memory_context or "").strip()
        if context:
            system_prompt = f"{system_prompt}\n\n{context}"
        return self._client.generate_stream(messages=messages, system_prompt=system_prompt)

    @staticmethod
    def close_stream(stream) -> None:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
