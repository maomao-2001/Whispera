[简体中文](./README.md) | [English](./README.en.md)

# Whispera

Whispera is a local real-time voice assistant desktop app for Windows.

The project currently supports local microphone input, VAD interruption, SenseVoice ASR, local LLM inference through `llama-server`, optional VoxCPM streaming TTS, and optional long-term memory integration through `mem0`.

Demo video: <https://www.bilibili.com/video/BV1nFE36mEmc>

## Overview

- Electron handles the desktop UI, configuration, logs, and local service orchestration.
- Python in `realtime/` handles VAD, ASR, LLM, TTS, the WebSocket protocol, and session state.
- Supports local `llama-server`, optional VoxCPM streaming TTS, and optional `mem0` long-term memory.
- Large model files, ASR/TTS weights, and distribution assets are not committed to Git and must be prepared separately.

## Requirements

- Windows 10/11
- PowerShell
- Node.js
- A local Python runtime
- Local model and resource files

It is recommended to use your own Python or conda environment for development. `runtime/python/` is mainly used for packaging the portable runtime. If you want to force a specific runtime, you can also set `MINIMIND_PYTHON`.

ASR uses `cuda` by default. If your machine does not have a usable GPU, or if you want to force CPU inference, explicitly set `MINIMIND_ASR_DEVICE=cpu`.

## Quick Start

### 0. Get the code

This repository uses Git LFS for some large files. Install and initialize Git LFS before cloning:

```powershell
git lfs install
git clone <repo-url>
cd <repo-dir>
git lfs pull
```

If you already cloned the repository but only got LFS pointer files, run `git lfs pull` once more.

### 1. Prepare Python and install dependencies

Whispera requires the GPU build of PyTorch by default. Do not switch to a CPU-only `torch` in the default setup, or ASR will not work correctly with the default configuration.

Run from the repository root:

```powershell
conda create -n whispera python=3.11 -y
conda activate whispera
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm install --prefix electron-app
```

After installation, it is recommended to verify that PyTorch is really using a CUDA build before starting the project:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected result:

- `torch.__version__` looks like `2.6.0+cu126`
- `torch.version.cuda` has a value such as `12.6`
- `torch.cuda.is_available()` prints `True`

If you do not use conda, you can install with your system Python directly. Only set this when you want to force a specific Python executable:

```powershell
$env:MINIMIND_PYTHON="C:\Users\you\anaconda3\envs\your_env\python.exe"
```

Then install dependencies and run the project.

Before the first launch, it is also recommended to explicitly clear the following environment variable so Electron is not accidentally started as a plain Node process:

```powershell
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
```

### 2. Prepare local assets

The following large assets are not included in the Git repository. You can download them into `assets/` with one command:

```powershell
npm run setup:assets
```

This is equivalent to:

```powershell
python scripts/download_assets.py
```

The command downloads:

- `llama-bin/`: the `llama-server` runtime from `ggml-org/llama.cpp` GitHub Releases
- `asr/SenseVoiceSmall/`: SenseVoiceSmall from Hugging Face
- `tts/openbmb__VoxCPM2/`: VoxCPM2 from Hugging Face
- `embedding/`: `nomic-embed-text-v1.5.Q8_0.gguf` from Hugging Face
- `lora/`, `reference/`, and `llm/`: placeholder directories only

```text
assets/
  llama-bin/
    llama-server.exe
  llm/
    *.gguf
  embedding/
    *.gguf                     # only needed for memory mode
  asr/
    SenseVoiceSmall/
  tts/
    openbmb__VoxCPM2/
  lora/
  reference/
```

Notes:

- `model/vad/silero_vad.onnx` is provided by this repository.
- LLM GGUF files are not downloaded by default. Developers should choose their own model and put it under `assets/llm/`, or select a local GGUF file in Settings.
- If there is only one `*.gguf` file under `assets/llm/`, Electron will use it automatically.
- `assets/llm/` must contain at least one `*.gguf`, otherwise `llama-server` cannot start.
- If Hugging Face is slow in your network, set `HF_ENDPOINT` to an available mirror before running the command.

To verify the local assets:

```powershell
npm run verify:assets
```

### 3. Start the development build

```powershell
npm run dev
```

This command forwards to `npm run dev` inside `electron-app/`.

After startup, the core service endpoints are usually:

```text
llama-server:      http://127.0.0.1:8080
memory embedding:  http://127.0.0.1:8081
realtime backend:  http://127.0.0.1:8011
realtime ws:       ws://127.0.0.1:8011/ws/realtime
web client:        http://127.0.0.1:8012/
```

## Core Features

- Electron desktop frontend with resource status, logs, and configuration panels
- Local `llama-server` startup and reuse
- Silero VAD with barge-in interruption
- SenseVoice ASR
- Streaming text output with sentence-level segmentation
- Optional VoxCPM streaming TTS
- Optional `mem0` long-term memory retrieval and persistence
- Windows portable directory packaging workflow

## System Architecture

```text
Electron renderer / web client
  -> WebSocket
  -> Python realtime backend
  -> Silero VAD
  -> SenseVoice ASR
  -> optional mem0 memory search
  -> llama-server (OpenAI-compatible API)
  -> text segmenter
  -> optional VoxCPM streaming TTS
  -> audio chunks back to Electron
```

When started in development mode, Electron brings up these local services as needed:

1. `llm-module/scripts/start_llama_server.py`
2. Optional memory embedding `llama-server`
3. `python -m realtime.app`
4. Local web client and Electron renderer

## Repository Structure

```text
electron-app/                    # Electron frontend, packaging config, process orchestration
realtime/                        # Python real-time backend
llm-module/                      # llama-server startup and local client
voxcpm-tts-streaming-module/     # VoxCPM TTS module
mem0/                            # vendored mem0 SDK source
model/vad/                       # Silero VAD model
runtime/                         # portable Python runtime docs and output directory
scripts/                         # packaging, runtime export, and backend build scripts
assets/                          # local resource directory, not committed to Git
distribution-assets/             # distribution resource staging directory, not committed to Git
```

## Common Environment Variables

```powershell
$env:MINIMIND_PYTHON="C:\Users\you\python.exe"
$env:MINIMIND_ASSETS_ROOT="D:\assets"
$env:MINIMIND_LLM_MODEL="D:\models\chat.gguf"
$env:MINIMIND_LLM_BASE_URL="http://127.0.0.1:8080"
$env:MINIMIND_BACKEND_HTTP_BASE="http://127.0.0.1:8011"
$env:MINIMIND_BACKEND_WS_URL="ws://127.0.0.1:8011/ws/realtime"
$env:MINIMIND_ASR_DEVICE="cuda"
$env:MINIMIND_DEBUG_TURNS="1"
```

If `MINIMIND_ASR_DEVICE` is not set, the default is `cuda`.

## Optional Memory Module

The repository already vendors `mem0/` as source code for local long-term memory support. See [mem0/LOCAL_CHANGES.md](./mem0/LOCAL_CHANGES.md) for project-specific local modifications.

There are two switches to be aware of:

- `MINIMIND_MEMORY_ENABLED`: master switch for the whole memory module
- `MINIMIND_MEMORY_INFER`: whether saved content should go through memory extraction inference

If you do not want memory enabled, it is recommended to disable it explicitly:

```powershell
$env:MINIMIND_MEMORY_ENABLED="0"
npm run dev
```

If you want to enable it:

```powershell
$env:MINIMIND_MEMORY_ENABLED="1"
$env:MINIMIND_MEMORY_EMBEDDER_MODEL_PATH="D:\models\embedding.gguf"
$env:MINIMIND_MEMORY_EMBEDDING_DIMS="1024"
npm run dev
```

Additional notes:

- `requirements.txt` already includes `requirements-mem0.txt`
- `requirements-mem0.txt` installs local `./mem0` as an editable package
- Memory data is written by default to `runtime/mem0/qdrant` and `runtime/mem0/history.db`
- If you only want to disable memory extraction, but not the whole module, set `MINIMIND_MEMORY_INFER=0`

## Packaging and Distribution

If you want to build the Windows portable version, it is recommended to follow this order:

### 1. Prepare the portable Python runtime

```powershell
.\scripts\pack_runtime.ps1 -ReplaceExisting
```

Optional slimming:

```powershell
.\scripts\slim_runtime_for_distribution.ps1 -RuntimeRoot runtime\python
```

### 2. Build the backend

```powershell
.\scripts\build_compiled_backend.ps1
```

If you need to specify Python explicitly:

```powershell
.\scripts\build_compiled_backend.ps1 -PythonExe C:\Users\you\python.exe
```

### 3. Generate the portable directory

```powershell
npm run dist --prefix electron-app
```

Output directory:

```text
electron-app/dist/win-unpacked/
```

The final distribution usually contains two parts:

1. `electron-app/dist/win-unpacked/`
2. A separately prepared `assets/` resource directory

## Asset and Git Strategy

The following content is intentionally excluded from Git:

- `assets/`
- `distribution-assets/`
- `runtime/python/`
- `runtime/mem0/`
- `build/compiled-backend/`
- `electron-app/dist/`
- `logs/`

The repository keeps source code, scripts, packaging workflows, and documentation. Actual models, weights, and large assets must be prepared by the user.

## Referenced Projects

This project references or integrates the following open-source projects:

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM): source of the streaming TTS capability
- [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o): lower-level reference project
- [mem0ai/mem0](https://github.com/mem0ai/mem0): source of the long-term memory module

## Related Documents

- [codebase-onboarding-report.md](./codebase-onboarding-report.md): codebase runtime flow analysis
- [electron-app/README.md](./electron-app/README.md): Electron-side notes
- [runtime/README.md](./runtime/README.md): portable runtime notes
- [mem0/LOCAL_CHANGES.md](./mem0/LOCAL_CHANGES.md): vendored mem0 local change log

## License

This project is licensed under the [Apache License 2.0](./LICENSE).

Third-party modules vendored or referenced in this repository continue to follow their own licenses. Please review the license and attribution notes in the corresponding directories when redistributing, reusing, or modifying them.
