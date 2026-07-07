from __future__ import annotations

import argparse
import json
import platform
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS_ROOT = REPO_ROOT / "assets"

DEFAULT_LLAMA_RELEASE = "b8171"
LLAMA_REPO = "ggml-org/llama.cpp"


@dataclass(frozen=True)
class HfSnapshotAsset:
    name: str
    repo_id: str
    target: Path
    allow_patterns: tuple[str, ...]
    required_files: tuple[str, ...]
    revision: str = "main"


@dataclass(frozen=True)
class HfFileAsset:
    name: str
    repo_id: str
    filename: str
    target: Path
    required_files: tuple[str, ...]
    revision: str = "main"


def build_hf_assets(assets_root: Path) -> tuple[HfSnapshotAsset, HfSnapshotAsset, HfFileAsset]:
    return (
        HfSnapshotAsset(
            name="asr",
            repo_id="FunAudioLLM/SenseVoiceSmall",
            target=assets_root / "asr" / "SenseVoiceSmall",
            allow_patterns=(
                "README.md",
                "am.mvn",
                "chn_jpn_yue_eng_ko_spectok.bpe.model",
                "config.yaml",
                "configuration.json",
                "model.pt",
            ),
            required_files=(
                "am.mvn",
                "chn_jpn_yue_eng_ko_spectok.bpe.model",
                "config.yaml",
                "model.pt",
            ),
        ),
        HfSnapshotAsset(
            name="tts",
            repo_id="openbmb/VoxCPM2",
            target=assets_root / "tts" / "openbmb__VoxCPM2",
            allow_patterns=(
                "README.md",
                "audiovae.pth",
                "config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenization_voxcpm2.py",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
            required_files=(
                "audiovae.pth",
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        ),
        HfFileAsset(
            name="embedding",
            repo_id="nomic-ai/nomic-embed-text-v1.5-GGUF",
            filename="nomic-embed-text-v1.5.Q8_0.gguf",
            target=assets_root / "embedding",
            required_files=("nomic-embed-text-v1.5.Q8_0.gguf",),
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Whispera runtime assets into the local assets directory.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=DEFAULT_ASSETS_ROOT,
        help="Target assets directory. Defaults to ./assets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing files.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the expected assets; do not download.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload managed assets even if required files already exist.",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip Hugging Face model assets.",
    )
    parser.add_argument(
        "--skip-llama",
        action="store_true",
        help="Skip llama.cpp binary assets.",
    )
    parser.add_argument(
        "--llama-release",
        default=DEFAULT_LLAMA_RELEASE,
        help="llama.cpp release tag to download, or 'latest'.",
    )
    parser.add_argument(
        "--llama-backend",
        default="win-cuda12-x64",
        choices=(
            "auto",
            "win-cuda12-x64",
            "win-cuda13-x64",
            "win-cpu-x64",
            "win-vulkan-x64",
            "linux-cpu-x64",
            "macos-arm64",
            "macos-x64",
        ),
        help="llama.cpp release asset family to select.",
    )
    parser.add_argument(
        "--llama-asset-substring",
        action="append",
        default=[],
        help=(
            "Select llama.cpp release assets by filename substring. "
            "Can be repeated and overrides --llama-backend selection."
        ),
    )
    parser.add_argument(
        "--keep-download-cache",
        action="store_true",
        help="Keep downloaded GitHub release archives under assets/.cache.",
    )
    return parser.parse_args()


def require_huggingface_hub() -> tuple[object, object]:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  python -m pip install huggingface_hub\n"
            "or install the project requirements first."
        ) from exc
    return hf_hub_download, snapshot_download


def is_complete(target: Path, required_files: Iterable[str]) -> bool:
    return all((target / relative_path).is_file() for relative_path in required_files)


def remove_target_if_forced(target: Path, force: bool, dry_run: bool) -> None:
    if not force or not target.exists():
        return
    print(f"[force] removing {target}")
    if dry_run:
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def download_hf_snapshot(asset: HfSnapshotAsset, force: bool, dry_run: bool) -> None:
    if is_complete(asset.target, asset.required_files) and not force:
        print(f"[skip] {asset.name}: already complete at {asset.target}")
        return

    print(f"[hf] {asset.name}: {asset.repo_id}@{asset.revision} -> {asset.target}")
    if dry_run:
        for pattern in asset.allow_patterns:
            print(f"      {pattern}")
        return

    remove_target_if_forced(asset.target, force, dry_run=False)
    asset.target.mkdir(parents=True, exist_ok=True)
    _, snapshot_download = require_huggingface_hub()
    snapshot_download(
        repo_id=asset.repo_id,
        revision=asset.revision,
        local_dir=str(asset.target),
        allow_patterns=list(asset.allow_patterns),
    )


def download_hf_file(asset: HfFileAsset, force: bool, dry_run: bool) -> None:
    if is_complete(asset.target, asset.required_files) and not force:
        print(f"[skip] {asset.name}: already complete at {asset.target}")
        return

    print(f"[hf] {asset.name}: {asset.repo_id}/{asset.filename}@{asset.revision} -> {asset.target}")
    if dry_run:
        return

    remove_target_if_forced(asset.target / asset.filename, force, dry_run=False)
    asset.target.mkdir(parents=True, exist_ok=True)
    hf_hub_download, _ = require_huggingface_hub()
    hf_hub_download(
        repo_id=asset.repo_id,
        filename=asset.filename,
        revision=asset.revision,
        local_dir=str(asset.target),
    )


def request_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Whispera-assets-downloader",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"GitHub API request failed: {url}\n{exc}\n{detail}") from exc


def resolve_llama_release(release: str) -> dict:
    if release == "latest":
        url = f"https://api.github.com/repos/{LLAMA_REPO}/releases/latest"
    else:
        url = f"https://api.github.com/repos/{LLAMA_REPO}/releases/tags/{release}"
    return request_json(url)


def effective_backend(backend: str) -> str:
    if backend != "auto":
        return backend

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "win-cuda12-x64"
    if system == "darwin":
        return "macos-arm64" if "arm" in machine or "aarch64" in machine else "macos-x64"
    if system == "linux":
        return "linux-cpu-x64"
    raise SystemExit(f"Unsupported platform for --llama-backend auto: {platform.platform()}")


def asset_name_matches_backend(name: str, backend: str) -> bool:
    lower = name.lower()
    if not lower.endswith((".zip", ".tar.gz", ".tgz")):
        return False

    if backend == "win-cuda12-x64":
        return "win" in lower and "x64" in lower and "cuda" in lower and ("12" in lower or "cu12" in lower)
    if backend == "win-cuda13-x64":
        return "win" in lower and "x64" in lower and "cuda" in lower and ("13" in lower or "cu13" in lower)
    if backend == "win-cpu-x64":
        excluded = ("cuda", "vulkan", "openvino", "sycl", "hip", "kompute")
        return "win" in lower and "x64" in lower and all(part not in lower for part in excluded)
    if backend == "win-vulkan-x64":
        return "win" in lower and "x64" in lower and "vulkan" in lower
    if backend == "linux-cpu-x64":
        excluded = ("cuda", "vulkan", "openvino", "sycl", "hip", "kompute", "rocm")
        return "linux" in lower and ("x64" in lower or "x86_64" in lower) and all(part not in lower for part in excluded)
    if backend == "macos-arm64":
        return ("macos" in lower or "darwin" in lower) and ("arm64" in lower or "aarch64" in lower)
    if backend == "macos-x64":
        return ("macos" in lower or "darwin" in lower) and ("x64" in lower or "x86_64" in lower)
    return False


def select_llama_assets(release_payload: dict, backend: str, substrings: list[str]) -> list[dict]:
    assets = release_payload.get("assets") or []
    if substrings:
        lowered = [item.lower() for item in substrings]
        selected = [
            asset for asset in assets
            if any(part in str(asset.get("name", "")).lower() for part in lowered)
        ]
    else:
        selected = [
            asset for asset in assets
            if asset_name_matches_backend(str(asset.get("name", "")), backend)
        ]

    if selected:
        return sorted(selected, key=lambda item: str(item.get("name", "")).lower())

    available = "\n".join(f"  - {asset.get('name')}" for asset in assets) or "  <no assets>"
    raise SystemExit(
        f"No llama.cpp release assets matched backend '{backend}'.\n"
        "Use --llama-asset-substring with one of these assets:\n"
        f"{available}"
    )


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        partial.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "Whispera-assets-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            expected = int(response.headers.get("Content-Length") or "0")
            downloaded = 0
            with partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    downloaded += len(chunk)

            if expected and downloaded != expected:
                raise RuntimeError(
                    f"Incomplete download for {destination.name}: "
                    f"{downloaded} bytes downloaded, expected {expected} bytes"
                )
        partial.replace(destination)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def archive_can_open(archive_path: Path) -> bool:
    try:
        if archive_path.suffix == ".zip":
            with zipfile.ZipFile(archive_path) as archive:
                archive.infolist()
            return True
        if archive_path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path) as archive:
                archive.getmembers()
            return True
    except (OSError, tarfile.TarError, zipfile.BadZipFile):
        return False
    return False


def safe_extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError:
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        archive.extractall(destination)


def safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            try:
                target.relative_to(destination_root)
            except ValueError:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        archive.extractall(destination)


def extract_archive(archive_path: Path, destination: Path) -> None:
    if archive_path.suffix == ".zip":
        safe_extract_zip(archive_path, destination)
        return
    if archive_path.name.endswith((".tar.gz", ".tgz")):
        safe_extract_tar(archive_path, destination)
        return
    raise RuntimeError(f"Unsupported archive type: {archive_path.name}")


def find_llama_payload_root(extracted_root: Path) -> Path:
    candidates = [
        path.parent
        for path in extracted_root.rglob("*")
        if path.is_file() and path.name in {"llama-server.exe", "llama-server"}
    ]
    if candidates:
        return sorted(candidates, key=lambda item: len(item.parts))[0]

    entries = list(extracted_root.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return extracted_root


def copy_tree_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def download_llama_bin(
    assets_root: Path,
    release: str,
    backend: str,
    substrings: list[str],
    force: bool,
    dry_run: bool,
    keep_download_cache: bool,
) -> None:
    target = assets_root / "llama-bin"
    required_server = "llama-server.exe" if platform.system().lower() == "windows" else "llama-server"
    if is_complete(target, (required_server,)) and not force:
        print(f"[skip] llama-bin: already complete at {target}")
        return

    resolved_backend = effective_backend(backend)
    print(f"[github] llama.cpp {release} ({resolved_backend}) -> {target}")
    if substrings:
        for substring in substrings:
            print(f"         asset substring: {substring}")
    if dry_run:
        print("         release assets will be resolved from the GitHub Releases API")
        return

    release_payload = resolve_llama_release(release)
    selected_assets = select_llama_assets(release_payload, resolved_backend, substrings)
    for asset in selected_assets:
        print(f"         {asset.get('name')}")

    remove_target_if_forced(target, force, dry_run=False)
    target.mkdir(parents=True, exist_ok=True)
    cache_dir = assets_root / ".cache" / "llama.cpp"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for asset in selected_assets:
        name = str(asset["name"])
        url = str(asset["browser_download_url"])
        archive_path = cache_dir / name

        if archive_path.exists() and not archive_can_open(archive_path):
            print(f"[cache] invalid archive, redownloading: {archive_path}")
            archive_path.unlink()

        if not archive_path.exists() or force:
            print(f"[download] {name}")
            download_file(url, archive_path)
            if not archive_can_open(archive_path):
                archive_path.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded archive is not readable: {name}")
        else:
            print(f"[cache] {archive_path}")

        with tempfile.TemporaryDirectory(prefix="whispera-llama-") as temp:
            extracted = Path(temp) / "extracted"
            extract_archive(archive_path, extracted)
            payload_root = find_llama_payload_root(extracted)
            copy_tree_contents(payload_root, target)

        if not keep_download_cache and archive_path.exists():
            archive_path.unlink()

    if not keep_download_cache:
        try:
            cache_dir.rmdir()
            cache_dir.parent.rmdir()
        except OSError:
            pass


def create_placeholders(assets_root: Path, dry_run: bool) -> None:
    for relative in ("llm", "lora", "reference"):
        target = assets_root / relative
        print(f"[dir] {target}")
        if not dry_run:
            target.mkdir(parents=True, exist_ok=True)
            gitkeep = target / ".gitkeep"
            if not gitkeep.exists():
                gitkeep.write_text("", encoding="utf-8")


def verify_assets(assets_root: Path, include_hf: bool, include_llama: bool) -> int:
    checks: list[tuple[str, Path, tuple[str, ...]]] = []
    if include_hf:
        for asset in build_hf_assets(assets_root):
            checks.append((asset.name, asset.target, asset.required_files))
    if include_llama:
        server_name = "llama-server.exe" if platform.system().lower() == "windows" else "llama-server"
        checks.append(("llama-bin", assets_root / "llama-bin", (server_name,)))

    failures = 0
    for name, target, required_files in checks:
        missing = [relative for relative in required_files if not (target / relative).is_file()]
        if missing:
            failures += 1
            print(f"[missing] {name}: {target}")
            for relative in missing:
                print(f"          {relative}")
        else:
            print(f"[ok] {name}: {target}")

    for relative in ("llm", "lora", "reference"):
        target = assets_root / relative
        if target.is_dir():
            print(f"[ok] placeholder: {target}")
        else:
            failures += 1
            print(f"[missing] placeholder: {target}")

    return failures


def main() -> int:
    args = parse_args()
    assets_root = args.assets_root.resolve()
    include_hf = not args.skip_hf
    include_llama = not args.skip_llama

    print(f"Whispera assets root: {assets_root}")

    if args.verify_only:
        return 1 if verify_assets(assets_root, include_hf, include_llama) else 0

    if not args.dry_run:
        assets_root.mkdir(parents=True, exist_ok=True)

    create_placeholders(assets_root, dry_run=args.dry_run)

    if include_hf:
        for asset in build_hf_assets(assets_root):
            if isinstance(asset, HfSnapshotAsset):
                download_hf_snapshot(asset, force=args.force, dry_run=args.dry_run)
            else:
                download_hf_file(asset, force=args.force, dry_run=args.dry_run)

    if include_llama:
        download_llama_bin(
            assets_root=assets_root,
            release=args.llama_release,
            backend=args.llama_backend,
            substrings=args.llama_asset_substring,
            force=args.force,
            dry_run=args.dry_run,
            keep_download_cache=args.keep_download_cache,
        )

    if args.dry_run:
        return 0

    return 1 if verify_assets(assets_root, include_hf, include_llama) else 0


if __name__ == "__main__":
    raise SystemExit(main())
