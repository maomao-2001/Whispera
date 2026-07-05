from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepend_if_missing(path: Path) -> None:
    value = str(path)
    if value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)


def ensure_repo_root_on_path() -> Path:
    _prepend_if_missing(REPO_ROOT)
    return REPO_ROOT


def ensure_llm_module_on_path() -> Path:
    ensure_repo_root_on_path()
    src_path = REPO_ROOT / "llm-module" / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"llm-module src path not found: {src_path}")
    _prepend_if_missing(src_path)
    return src_path


def ensure_mem0_on_path() -> Path | None:
    ensure_repo_root_on_path()
    src_path = REPO_ROOT / "mem0"
    if not src_path.exists():
        return None
    _prepend_if_missing(src_path)
    return src_path


def ensure_voxcpm_on_path() -> Path:
    ensure_repo_root_on_path()
    src_path = REPO_ROOT / "voxcpm-tts-streaming-module" / "src"
    if not src_path.exists():
        raise FileNotFoundError(f"voxcpm src path not found: {src_path}")
    _prepend_if_missing(src_path)
    return src_path
