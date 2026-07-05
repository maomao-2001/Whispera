import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = REPO_ROOT / "llama" / "bin" / "llama-server.exe"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the bundled llama-server.exe")
    parser.add_argument("--server", type=Path, default=DEFAULT_SERVER, help="Path to llama-server.exe")
    parser.add_argument("--model", type=Path, help="Path to a GGUF model")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="GPU layers passed to llama-server")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra arguments passed through to llama-server")
    return parser


def _iter_torch_lib_candidates() -> list[Path]:
    python_root = Path(sys.executable).resolve().parent
    repo_runtime_root = REPO_ROOT.parent / "runtime" / "python"
    candidates = [
        python_root / "Lib" / "site-packages" / "torch" / "lib",
        repo_runtime_root / "Lib" / "site-packages" / "torch" / "lib",
    ]

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(candidate)
    return deduped


def _build_child_env(server_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    path_parts: list[str] = []
    seen: set[str] = set()

    for candidate in [server_dir, *_iter_torch_lib_candidates()]:
        if not candidate.is_dir():
            continue
        normalized = str(candidate).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        path_parts.append(str(candidate))

    existing_path = env.get("PATH", "")
    if existing_path:
        path_parts.append(existing_path)
    env["PATH"] = os.pathsep.join(path_parts)
    return env


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.server.exists():
        raise SystemExit(f"llama-server.exe not found: {args.server}")
    if not args.model:
        raise SystemExit("Missing required --model path")
    if not args.model.exists():
        raise SystemExit(f"model file not found: {args.model}")

    command = [
        str(args.server),
        "--model",
        str(args.model),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--ctx-size",
        str(args.ctx_size),
        "--n-gpu-layers",
        str(args.n_gpu_layers),
    ]
    if args.extra:
        command.extend(args.extra)

    child_env = _build_child_env(args.server.parent)
    torch_lib_dirs = [candidate for candidate in _iter_torch_lib_candidates() if candidate.is_dir()]
    print("[llama-server] " + " ".join(command), flush=True)
    if torch_lib_dirs:
        print("[llama-server] added DLL search paths: " + ", ".join(str(path) for path in torch_lib_dirs), flush=True)
    else:
        print("[llama-server] warning: torch CUDA runtime DLL directory not found; CUDA backend may fall back to CPU", flush=True)
    raise SystemExit(subprocess.run(command, cwd=args.server.parent, env=child_env).returncode)


if __name__ == "__main__":
    main()
