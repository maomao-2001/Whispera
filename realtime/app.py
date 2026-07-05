from __future__ import annotations

from .backend_main import RealtimeAppConfig, create_app, main, parse_args

__all__ = [
    "RealtimeAppConfig",
    "create_app",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()
