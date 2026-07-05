from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .local_modules import REPO_ROOT, ensure_mem0_on_path


logger = logging.getLogger(__name__)


DEFAULT_MEMORY_CUSTOM_INSTRUCTIONS = (
    "Extract only durable long-term user facts. "
    "Keep stable preferences, personal constraints, identity details, recurring habits, and long-term projects. "
    "Do not store transient requests, temporary decisions, invitations, current-session recall questions, or generic advice-seeking. "
    "If no durable fact exists, return an empty memory list."
)


def _normalize_openai_base_url(value: str | None, default: str) -> str:
    text = str(value or default).strip().rstrip("/")
    if not text:
        text = default.rstrip("/")
    return text if text.endswith("/v1") else f"{text}/v1"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in {None, ""}:
        return int(default)
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in {None, ""}:
        return float(default)
    return float(value)


def _normalize_mem0_init_error(exc: Exception) -> str:
    message = str(exc)
    if "No package metadata was found for mem0ai" in message:
        return "mem0 dependencies are missing; run: python -m pip install -r requirements-mem0.txt"
    if isinstance(exc, ModuleNotFoundError):
        missing_name = getattr(exc, "name", "") or "unknown module"
        return (
            f"mem0 dependency '{missing_name}' is missing; "
            "run: python -m pip install -r requirements-mem0.txt"
        )
    return message


@dataclass
class MemoryRuntimeConfig:
    enabled: bool = False
    user_id: str = "local_user"
    agent_id: str = "realtime_chat2"
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_model: str | None = None
    embedder_base_url: str = "http://127.0.0.1:8081/v1"
    embedder_model: str = "local-embedding"
    embedding_dims: int = 1024
    collection_name: str = "realtime_chat_memory"
    store_path: str = str(REPO_ROOT / "runtime" / "mem0" / "qdrant")
    history_db_path: str = str(REPO_ROOT / "runtime" / "mem0" / "history.db")
    top_k: int = 5
    threshold: float = 0.1
    infer: bool = True
    context_char_limit: int = 1200
    custom_instructions: str | None = DEFAULT_MEMORY_CUSTOM_INSTRUCTIONS

    @classmethod
    def from_env(
        cls,
        llm_base_url: str,
        llm_model: str | None = None,
        enabled: bool | None = None,
    ) -> "MemoryRuntimeConfig":
        return cls(
            enabled=_env_bool("MINIMIND_MEMORY_ENABLED", False) if enabled is None else bool(enabled),
            user_id=os.getenv("MINIMIND_MEMORY_USER_ID", "local_user"),
            agent_id=os.getenv("MINIMIND_MEMORY_AGENT_ID", "realtime_chat2"),
            llm_base_url=_normalize_openai_base_url(os.getenv("MINIMIND_MEMORY_LLM_BASE_URL"), llm_base_url),
            llm_model=os.getenv("MINIMIND_MEMORY_LLM_MODEL") or llm_model,
            embedder_base_url=_normalize_openai_base_url(
                os.getenv("MINIMIND_MEMORY_EMBEDDER_BASE_URL"),
                "http://127.0.0.1:8081",
            ),
            embedder_model=os.getenv("MINIMIND_MEMORY_EMBEDDER_MODEL", "local-embedding"),
            embedding_dims=_env_int("MINIMIND_MEMORY_EMBEDDING_DIMS", 768),
            collection_name=os.getenv("MINIMIND_MEMORY_COLLECTION", "realtime_chat_memory"),
            store_path=os.getenv("MINIMIND_MEMORY_STORE_PATH", str(REPO_ROOT / "runtime" / "mem0" / "qdrant")),
            history_db_path=os.getenv("MINIMIND_MEMORY_HISTORY_DB", str(REPO_ROOT / "runtime" / "mem0" / "history.db")),
            top_k=_env_int("MINIMIND_MEMORY_TOP_K", 5),
            threshold=_env_float("MINIMIND_MEMORY_THRESHOLD", 0.1),
            infer=_env_bool("MINIMIND_MEMORY_INFER", True),
            context_char_limit=_env_int("MINIMIND_MEMORY_CONTEXT_CHARS", 1200),
            custom_instructions=os.getenv("MINIMIND_MEMORY_CUSTOM_INSTRUCTIONS", DEFAULT_MEMORY_CUSTOM_INSTRUCTIONS),
        )


@dataclass
class MemoryStatus:
    enabled: bool
    available: bool
    user_id: str
    collection_name: str
    store_path: str
    embedder_base_url: str
    embedder_model: str
    embedding_dims: int
    last_error: str | None = None


class RealtimeMemoryService:
    def __init__(self, config: MemoryRuntimeConfig):
        self.config = config
        self._memory: Any | None = None
        self._last_error: str | None = None

        logger.info("[memory] init: enabled=%s, user_id=%s, embedder_url=%s, dims=%d, store=%s",
                    config.enabled, config.user_id, config.embedder_base_url,
                    config.embedding_dims, config.store_path)

        if not config.enabled:
            logger.info("[memory] disabled by config — skipping initialization")
            return

        os.environ.setdefault("MEM0_TELEMETRY", "false")
        os.environ.setdefault("MEM0_DIR", str(REPO_ROOT / "runtime" / "mem0" / "config"))

        try:
            mem0_path = ensure_mem0_on_path()
            logger.info("[memory] mem0 path resolved: %s", mem0_path)
            from mem0 import Memory  # type: ignore
            logger.info("[memory] mem0.Memory imported successfully")

            Path(config.store_path).mkdir(parents=True, exist_ok=True)
            Path(config.history_db_path).parent.mkdir(parents=True, exist_ok=True)
            mem0_config = self._build_mem0_config()
            logger.info("[memory] mem0 config: %s", mem0_config)
            self._memory = Memory.from_config(mem0_config)
            logger.info("[memory] Memory instance created successfully — available=True")
        except Exception as exc:
            import traceback
            self._last_error = _normalize_mem0_init_error(exc)
            logger.warning("[memory] INIT FAILED: %s\n%s", exc, traceback.format_exc())

    @property
    def available(self) -> bool:
        return self._memory is not None

    def status(self) -> MemoryStatus:
        return MemoryStatus(
            enabled=self.config.enabled,
            available=self.available,
            user_id=self.config.user_id,
            collection_name=self.config.collection_name,
            store_path=self.config.store_path,
            embedder_base_url=self.config.embedder_base_url,
            embedder_model=self.config.embedder_model,
            embedding_dims=self.config.embedding_dims,
            last_error=self._last_error,
        )

    def status_dict(self) -> dict[str, Any]:
        return asdict(self.status())

    def _build_mem0_config(self) -> dict[str, Any]:
        llm_config: dict[str, Any] = {
            "model": self.config.llm_model or "local-chat",
            "openai_base_url": self.config.llm_base_url,
            "api_key": os.getenv("MINIMIND_MEMORY_LLM_API_KEY", "dummy"),
            "temperature": 0.1,
            "max_tokens": 800,
        }
        return {
            "llm": {
                "provider": "openai",
                "config": llm_config,
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.config.embedder_model,
                    "openai_base_url": self.config.embedder_base_url,
                    "api_key": os.getenv("MINIMIND_MEMORY_EMBEDDER_API_KEY", "dummy"),
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.config.collection_name,
                    "path": self.config.store_path,
                    "embedding_model_dims": self.config.embedding_dims,
                    "on_disk": True,
                },
            },
            "history_db_path": self.config.history_db_path,
            "custom_instructions": self.config.custom_instructions,
        }

    def _filters(self) -> dict[str, str]:
        return {"user_id": self.config.user_id}

    def _remember_error(self, exc: Exception) -> None:
        self._last_error = str(exc)
        logger.warning("mem0 operation failed: %s", exc)

    def search_context(self, query: str) -> str:
        text = str(query or "").strip()
        logger.info("[MEMORY-SEARCH] called with query=%r, memory_available=%s", text[:80] if text else "", self._memory is not None)
        if not self._memory or not text:
            logger.info("[MEMORY-SEARCH] early return: memory=%s, text_empty=%s", self._memory is not None, not text)
            return ""

        try:
            response = self._memory.search(
                text,
                filters=self._filters(),
                top_k=self.config.top_k,
                threshold=self.config.threshold,
            )
            logger.info("[MEMORY-SEARCH] raw response type=%s, keys=%s", type(response).__name__, list(response.keys()) if isinstance(response, dict) else "N/A")
        except Exception as exc:
            logger.error("[MEMORY-SEARCH] search failed: %s", exc, exc_info=True)
            self._remember_error(exc)
            return ""

        results = response.get("results", response) if isinstance(response, dict) else response
        memories: list[str] = []
        for item in results if isinstance(results, list) else []:
            if not isinstance(item, dict):
                continue
            memory_text = str(item.get("memory") or item.get("text") or "").strip()
            if memory_text:
                memories.append(memory_text)

        logger.info("[MEMORY-SEARCH] found %d memories for query", len(memories))
        if not memories:
            return ""

        context = "\n".join(f"- {memory}" for memory in memories)
        if len(context) > self.config.context_char_limit:
            context = context[: self.config.context_char_limit].rstrip()
        return "以下是关于当前用户的长期记忆。仅在相关时自然参考，不要主动说明你在读取记忆：\n" + context

    def save_turn(self, user_text: str, assistant_text: str, session_id: str) -> dict[str, Any] | None:
        logger.info("[MEMORY-SAVE] called: memory_available=%s, user_text=%r, assistant_text=%r", self._memory is not None, (user_text or "")[:60], (assistant_text or "")[:60])
        if not self._memory:
            logger.info("[MEMORY-SAVE] skipped: memory not available")
            return None

        user_value = str(user_text or "").strip()
        assistant_value = str(assistant_text or "").strip()
        if not user_value or not assistant_value:
            logger.info("[MEMORY-SAVE] skipped: empty user or assistant text")
            return None

        messages = [
            {"role": "user", "content": user_value},
        ]
        metadata = {
            "source": "Whispera",
            "session_id": session_id,
            "agent_id": self.config.agent_id,
        }
        try:
            result = self._memory.add(
                messages,
                user_id=self.config.user_id,
                metadata=metadata,
                infer=self.config.infer,
            )
            logger.info("[MEMORY-SAVE] success: result=%s", result)
            return result
        except Exception as exc:
            self._remember_error(exc)
            return None
