from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional


_BUILTIN_DEFAULT_SYSTEM_PROMPT = (
    "你正在进行实时语音对话。\n"
    "这不是写作，不是客服，不是课堂讲解，而是一次自然、真实的当面交流。\n"
    "\n"
    "你的整体气质是：成熟、知性、温和、可靠，像一个情绪稳定、很会倾听、也很会照顾人感受的姐姐。\n"
    "说话要有分寸，温柔但不讨好，关心但不黏腻，亲切但不轻浮。\n"
    "\n"
    "请优先做到：\n"
    "先理解用户当下真正想要的，是答案、建议、安慰，还是单纯想有人听他说。\n"
    "不要复述、改写或总结用户的问题，不要用“你是想问”“如果你是说”“关于你刚才说的”这类铺垫开头。\n"
    "如果用户在表达情绪，先接住情绪，再回应内容。\n"
    "如果用户是在提问，第一句话直接给答案或结论，再补一句自然解释。\n"
    "能一句说清就一句，用户没有要求展开时，不主动讲太多背景。\n"
    "一次回复通常控制在 1 到 3 句，默认简短，不长篇大论。\n"
    "语气自然放松，像真人说话，不要有明显书面腔。\n"
    "\n"
    "表达风格：\n"
    "多用自然短句，少用生硬的长句、套话和官方话术。\n"
    "可以温和、平静、略带安抚感，但不要过度热情，也不要刻意哄人。\n"
    "可以适度使用成熟自然的表达，比如“别着急，我们慢慢看”“这件事我理解你的感受”“如果是我的判断，我会更偏向……”\n"
    "可以有少量自然停顿，比如“让我想想”“你这么说的话”，但不要频繁重复。\n"
    "不要故意卖萌，不要撒娇，不要使用“宝贝”“亲爱的”这类称呼。\n"
    "不要总是用固定句式开头，比如“我觉得”“在我看来”“其实你这个问题”。\n"
    "不要像老师讲课，也不要像心理咨询模板。\n"
    "\n"
    "互动原则：\n"
    "当用户脆弱、委屈、焦虑、疲惫时，先让对方感觉被理解，再给建议。\n"
    "当用户只是想倾诉时，少一点解决方案，多一点陪伴感。\n"
    "当用户明确要方法、判断或结论时，直接回答，不要绕。\n"
    "当用户问地点、做法、选择或判断时，直接说推荐项或结论，不要先把问题换一种说法。\n"
    "如果用户表达不完整、信息不够，或者你没听清，不要硬猜，先用一句温和的话确认。\n"
    "\n"
    "输出限制：\n"
    "回答必须适合直接送入 TTS 播报，读起来顺口、自然。\n"
    "不要输出思考过程、推理标签、括号旁白。\n"
    "不要使用 emoji、颜文字、星号、markdown、列表、表格。\n"
    "不要分点，不要解释结构，直接说内容。\n"
    "不要自称自己是 AI、语音助手或模型，除非用户明确追问身份。\n"
    "\n"
    "目标：让用户感觉你成熟、温柔、聪明、有分寸，像一个真正会听人说话、也能把话说得恰到好处的人。"
)
_SYSTEM_PROMPT_FILE = Path("assets") / "llm" / "system_prompt.txt"


def _iter_system_prompt_candidates() -> Iterable[Path]:
    module_path = Path(__file__).resolve()
    for parent in module_path.parents:
        yield parent / _SYSTEM_PROMPT_FILE


def _load_external_system_prompt() -> Optional[str]:
    # Search upward so both repo runs and packaged builds can override the
    # builtin prompt with assets/llm/system_prompt.txt.
    for candidate in _iter_system_prompt_candidates():
        try:
            if not candidate.is_file():
                continue
            prompt_text = candidate.read_text(encoding="utf-8-sig").strip()
        except OSError:
            continue
        if prompt_text:
            return prompt_text
    return None


def get_default_system_prompt() -> str:
    return _load_external_system_prompt() or _BUILTIN_DEFAULT_SYSTEM_PROMPT


DEFAULT_SYSTEM_PROMPT = get_default_system_prompt()
CHAT_COMPATIBLE_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
VOICE_STRICT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
CHAT_COMPATIBLE_PROMPT_PRESET = "chat_compatible"
VOICE_STRICT_PROMPT_PRESET = "voice_strict"
DEFAULT_PROMPT_PRESET = CHAT_COMPATIBLE_PROMPT_PRESET


def _normalize_base_url(base_url: str) -> str:
    value = str(base_url or "http://127.0.0.1:8080").strip()
    return value.rstrip("/")


def _coalesce_float(value: Optional[float], default: float) -> float:
    if value is None:
        return float(default)
    return float(value)


def _coalesce_int(value: Optional[int], default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _coalesce_optional_int(value: Optional[int], default: Optional[int]) -> Optional[int]:
    if value is None:
        return None if default is None else int(default)
    return int(value)


def _coalesce_optional_str(value: Optional[str], default: Optional[str]) -> Optional[str]:
    candidate = default if value is None else value
    if candidate is None:
        return None
    normalized = str(candidate).strip()
    return normalized or None


def resolve_system_prompt(system_prompt: Optional[str], prompt_preset: str = DEFAULT_PROMPT_PRESET) -> str:
    explicit_prompt = str(system_prompt or "").strip()
    if explicit_prompt:
        return explicit_prompt

    preset = str(prompt_preset or DEFAULT_PROMPT_PRESET).strip().lower()
    if preset == VOICE_STRICT_PROMPT_PRESET:
        return get_default_system_prompt()
    return get_default_system_prompt()


def _extract_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("code") or payload.get("type")
        if message:
            return str(message)
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def _extract_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    return ""


def _extract_delta_parts(choice: Dict[str, Any]) -> tuple[str, str]:
    delta = choice.get("delta") or {}
    content_text = _extract_content_text(delta.get("content"))
    reasoning_text = _extract_content_text(
        delta.get("reasoning_content") if delta.get("reasoning_content") is not None else delta.get("reasoning")
    )

    if not content_text:
        content_text = _extract_content_text(choice.get("text"))

    return content_text, reasoning_text


@dataclass
class LLMDelta:
    index: int
    text: str
    reasoning_text: str = ""
    model: Optional[str] = None
    raw_event: Optional[Dict[str, Any]] = None


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: Optional[str] = None
    system_prompt: str = field(default_factory=get_default_system_prompt)
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    timeout: float = 600.0
    api_key: Optional[str] = None
    user_agent: str = "llm-module/0.1"
    reasoning_budget: Optional[int] = None
    reasoning_format: Optional[str] = None
    enable_thinking: Optional[bool] = None


class LlamaServerLLM:
    """Small OpenAI-compatible client for llama.cpp's llama-server."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.config.base_url = _normalize_base_url(self.config.base_url)

    @property
    def models_url(self) -> str:
        return f"{self.config.base_url}/v1/models"

    @property
    def chat_completions_url(self) -> str:
        return f"{self.config.base_url}/v1/chat/completions"

    def _build_headers(self, stream: bool = False) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if stream else "application/json",
            "User-Agent": self.config.user_agent,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_messages(self, text: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        user_text = str(text or "").strip()
        if not user_text:
            raise ValueError("'text' must be a non-empty string")

        prompt = self.config.system_prompt if system_prompt is None else str(system_prompt)
        prompt = prompt.strip()

        messages: list[Dict[str, str]] = []
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _normalize_messages(self, messages: Iterable[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized: list[Dict[str, str]] = []
        for item in messages:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role") or "").strip()
            if role not in {"system", "user", "assistant", "tool"}:
                continue

            content = item.get("content")
            content_text = _extract_content_text(content).strip() if isinstance(content, list) else str(content or "").strip()
            if not content_text:
                continue
            normalized.append({"role": role, "content": content_text})

        if not normalized:
            raise ValueError("'messages' must contain at least one non-empty chat message")
        return normalized

    def _build_chat_messages(
        self,
        messages: Iterable[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        normalized = self._normalize_messages(messages)
        if any(item["role"] == "system" for item in normalized):
            return normalized

        prompt = self.config.system_prompt if system_prompt is None else str(system_prompt)
        prompt = prompt.strip()
        if not prompt:
            return normalized

        return [{"role": "system", "content": prompt}, *normalized]

    def _iter_sse_events(self, response: Any) -> Generator[str, None, None]:
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="ignore").strip()

            if not line:
                if data_lines:
                    yield "\n".join(data_lines)
                    data_lines = []
                continue

            if line.startswith(":"):
                continue

            if line.lower().startswith("data:"):
                data_lines.append(line[5:].lstrip())

        if data_lines:
            yield "\n".join(data_lines)

    def list_models(self) -> List[str]:
        request = urllib.request.Request(
            self.models_url,
            headers=self._build_headers(stream=False),
            method="GET",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama-server models request failed: {exc.code} {exc.reason} | {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to connect to llama-server at {self.config.base_url}: {exc.reason}") from exc

        models = payload.get("data") or []
        return [str(item.get("id")) for item in models if isinstance(item, dict) and item.get("id")]

    def resolve_model(self, override_model: Optional[str] = None) -> str:
        if override_model is not None and str(override_model).strip():
            return str(override_model).strip()
        if self.config.model is not None and str(self.config.model).strip():
            return str(self.config.model).strip()

        models = self.list_models()
        if not models:
            raise RuntimeError("no models available from llama-server; please specify a model or start llama-server correctly")
        return models[0]

    def generate_stream(
        self,
        text: str = "",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[Iterable[Dict[str, Any]]] = None,
        reasoning_budget: Optional[int] = None,
        reasoning_format: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> Generator[LLMDelta, None, None]:
        resolved_model = self.resolve_model(model)
        payload = {
            "model": resolved_model,
            "messages": self._build_chat_messages(messages, system_prompt) if messages is not None else self._build_messages(text, system_prompt),
            "stream": True,
            "temperature": _coalesce_float(temperature, self.config.temperature),
            "top_p": _coalesce_float(top_p, self.config.top_p),
            "max_tokens": _coalesce_int(max_tokens, self.config.max_tokens),
        }
        resolved_reasoning_budget = _coalesce_optional_int(reasoning_budget, self.config.reasoning_budget)
        resolved_reasoning_format = _coalesce_optional_str(reasoning_format, self.config.reasoning_format)
        if resolved_reasoning_budget is not None:
            payload["reasoning_budget"] = resolved_reasoning_budget
        if resolved_reasoning_format is not None:
            payload["reasoning_format"] = resolved_reasoning_format
        resolved_enable_thinking = self.config.enable_thinking if enable_thinking is None else enable_thinking
        if resolved_enable_thinking is not None:
            # llama.cpp forwards these kwargs to Jinja chat templates. Different
            # Qwen-family templates use different flag names, so send both.
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(resolved_enable_thinking),
                "open_thinking": bool(resolved_enable_thinking),
            }

        request = urllib.request.Request(
            self.chat_completions_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=self._build_headers(stream=True),
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                index = 0
                for event_payload in self._iter_sse_events(response):
                    if not event_payload:
                        continue
                    if event_payload == "[DONE]":
                        break

                    try:
                        event = json.loads(event_payload)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"invalid SSE payload from llama-server: {event_payload}") from exc

                    if event.get("error") is not None:
                        raise RuntimeError(_extract_error_message(event["error"]))

                    event_model = str(event.get("model") or resolved_model)
                    choices = event.get("choices") or []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        delta_text, reasoning_text = _extract_delta_parts(choice)
                        if not delta_text and not reasoning_text:
                            continue
                        yield LLMDelta(
                            index=index,
                            text=delta_text,
                            reasoning_text=reasoning_text,
                            model=event_model,
                            raw_event=event,
                        )
                        index += 1
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"llama-server chat request failed: {exc.code} {exc.reason} | {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"failed to connect to llama-server at {self.config.base_url}: {exc.reason}") from exc

    def generate_text(
        self,
        text: str = "",
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        messages: Optional[Iterable[Dict[str, Any]]] = None,
        reasoning_budget: Optional[int] = None,
        reasoning_format: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
    ) -> str:
        return "".join(
            delta.text
            for delta in self.generate_stream(
                text=text,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                messages=messages,
                reasoning_budget=reasoning_budget,
                reasoning_format=reasoning_format,
                enable_thinking=enable_thinking,
            )
        )
