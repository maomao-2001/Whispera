from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover - exercised by the build host
    raise SystemExit(
        "Cython is required for compiled backend builds. Install it into the packaging environment first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "build" / "compiled-backend"
DEFAULT_TEMP_ROOT = REPO_ROOT / ".cb"


@dataclass(frozen=True)
class CompileTarget:
    source_root: Path
    module_prefix: str
    bundle_root: Path
    preserve_sources: set[str]


TARGETS = (
    CompileTarget(
        source_root=REPO_ROOT / "realtime",
        module_prefix="realtime",
        bundle_root=Path("realtime"),
        preserve_sources={"__init__.py", "app.py"},
    ),
    CompileTarget(
        source_root=REPO_ROOT / "llm-module" / "src" / "llm_module",
        module_prefix="llm_module",
        bundle_root=Path("llm-module") / "src" / "llm_module",
        preserve_sources={"__init__.py"},
    ),
)

SOURCE_ONLY_TREES = (
    (
        REPO_ROOT / "voxcpm-tts-streaming-module" / "src" / "voxcpm",
        Path("voxcpm-tts-streaming-module") / "src" / "voxcpm",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile MiniMind Python modules into a release bundle without raw source.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Directory that will receive the compiled backend bundle.")
    parser.add_argument("--temp-root", default=str(DEFAULT_TEMP_ROOT), help="Temporary build directory used by setuptools/Cython.")
    parser.add_argument("--keep-temp", action="store_true", help="Keep the intermediate setuptools build directories.")
    return parser.parse_args()


def iter_compile_sources(target: CompileTarget) -> list[Path]:
    files: list[Path] = []
    for path in target.source_root.rglob("*.py"):
        if path.name in target.preserve_sources:
            continue
        files.append(path)
    return sorted(files)


def module_name_for(target: CompileTarget, source_path: Path) -> str:
    relative = source_path.relative_to(target.source_root).with_suffix("")
    parts = [target.module_prefix, *relative.parts]
    return ".".join(parts)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree_filtered(source_root: Path, destination_root: Path, include_predicate) -> int:
    copied = 0
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        if not include_predicate(path):
            continue
        destination = destination_root / path.relative_to(source_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    return copied


def build_extensions(temp_root: Path, sources: list[tuple[CompileTarget, Path]]) -> Path:
    build_temp = temp_root / "t"
    build_lib = temp_root / "l"
    ensure_clean_dir(build_temp)
    ensure_clean_dir(build_lib)

    ext_modules = [
        Extension(
            name=module_name_for(target, source_path),
            sources=[str(source_path)],
        )
        for target, source_path in sources
    ]

    setup(
        name="minimind-compiled-backend",
        script_args=["build_ext", "--build-lib", str(build_lib), "--build-temp", str(build_temp)],
        ext_modules=cythonize(
            ext_modules,
            compiler_directives={"language_level": 3},
            build_dir=str(temp_root / "c"),
            quiet=True,
        ),
    )
    return build_lib


def copy_compiled_outputs(build_lib: Path, output_root: Path) -> int:
    copied = 0

    realtime_build = build_lib / "realtime"
    if realtime_build.exists():
        copied += copy_tree_filtered(
            realtime_build,
            output_root / "realtime",
            lambda path: path.suffix.lower() in {".pyd", ".so", ".dll", ".pyi"},
        )

    llm_build = build_lib / "llm_module"
    if llm_build.exists():
        copied += copy_tree_filtered(
            llm_build,
            output_root / "llm-module" / "src" / "llm_module",
            lambda path: path.suffix.lower() in {".pyd", ".so", ".dll", ".pyi"},
        )

    return copied


def copy_runtime_wrappers(output_root: Path) -> int:
    copied = 0
    for target in TARGETS:
        destination = output_root / target.bundle_root
        copied += copy_tree_filtered(
            target.source_root,
            destination,
            lambda path, preserved=target.preserve_sources: path.name in preserved,
        )

    for source_root, bundle_root in SOURCE_ONLY_TREES:
        destination = output_root / bundle_root
        copied += copy_tree_filtered(
            source_root,
            destination,
            lambda path: "__pycache__" not in path.parts and path.suffix.lower() in {".py", ".pyi", ".json"},
        )

    llm_scripts_source = REPO_ROOT / "llm-module" / "scripts"
    llm_scripts_destination = output_root / "llm-module" / "scripts"
    copied += copy_tree_filtered(llm_scripts_source, llm_scripts_destination, lambda path: path.suffix.lower() == ".py")

    return copied


def write_manifest(output_root: Path, compiled_count: int, wrapper_count: int) -> None:
    manifest = {
        "compiled_files": compiled_count,
        "wrapper_files": wrapper_count,
        "bundle_root": str(output_root),
        "targets": [
            {
                "source_root": str(target.source_root),
                "bundle_root": str(output_root / target.bundle_root),
                "preserve_sources": sorted(target.preserve_sources),
            }
            for target in TARGETS
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    temp_root = Path(args.temp_root).resolve()

    ensure_clean_dir(output_root)
    sources = [(target, source_path) for target in TARGETS for source_path in iter_compile_sources(target)]
    build_lib = build_extensions(temp_root, sources)
    compiled_count = copy_compiled_outputs(build_lib, output_root)
    wrapper_count = copy_runtime_wrappers(output_root)
    write_manifest(output_root, compiled_count=compiled_count, wrapper_count=wrapper_count)

    if not args.keep_temp and temp_root.exists():
        shutil.rmtree(temp_root)

    print(f"Compiled backend bundle written to: {output_root}")
    print(f"Compiled modules: {compiled_count}")
    print(f"Wrapper/source files kept: {wrapper_count}")


if __name__ == "__main__":
    main()
