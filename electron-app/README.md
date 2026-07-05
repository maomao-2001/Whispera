# Electron Frontend

This folder contains the desktop frontend shell for the slimmed realtime assistant path.

## Current status

What exists now:

- custom Electron window shell
- backend status panel
- Start Voice Session starts `llama-server`, starts the Python realtime backend, then opens the realtime WebSocket
- Stop Voice Session closes microphone/WebSocket first, then stops Electron-managed backend processes
- microphone streaming, transcript transport, and text-only assistant streaming

## Run

Install dependencies:

```bash
npm install
```

Start the app:

```bash
npm run dev
```

Electron resolves Python from `runtime/python/python.exe` by default. Set `MINIMIND_PYTHON` only when you intentionally want to override that runtime.

When memory is enabled and `MINIMIND_MEMORY_INFER` is not turned off, Electron also starts the chat llama server with a larger default context window (`16384`) so mem0 extraction can run without immediately hitting the 8192-token limit. You can still override this with `MINIMIND_LLAMA_CTX_SIZE`.

Click `Start Voice Session` in the app. Electron will start services in this order:

1. `llm-module/scripts/start_llama_server.py`
2. `python -m realtime.app`
3. renderer microphone and `ws://127.0.0.1:8011/ws/realtime`

Click `Stop Voice Session` before closing if you want to release the local model processes immediately. App quit also cleans up services started by Electron.

Service stdout/stderr is mirrored into the in-app Logs panel and full log files under Electron user data. Use `Logs -> Open Service Log Folder` to inspect complete `realtime` and `llama` output.

Optional overrides:

```bash
# only set this when you want to override the default runtime/python/python.exe
set MINIMIND_PYTHON=C:\Users\you\anaconda3\envs\your_env\python.exe
set MINIMIND_LLM_MODEL=D:\models\your-model.gguf
set MINIMIND_LLM_BASE_URL=http://127.0.0.1:8080
set MINIMIND_BACKEND_HTTP_BASE=http://127.0.0.1:8011
set MINIMIND_BACKEND_WS_URL=ws://127.0.0.1:8011/ws/realtime
npm run dev
```

Other useful overrides:

- `MINIMIND_ASR_DEVICE=cpu`
- `MINIMIND_DEBUG_TURNS=1`
- `MINIMIND_DEBUG_OUTPUT_DIR=D:\project\minimind-base-chat\debug_turns`

## Packaging

Before packaging, compile the backend bundle that will be copied into `resources/app-bundle/`:

```powershell
.\scripts\build_compiled_backend.ps1
```

If the default Python selection is not the one you want, pass it explicitly:

```powershell
.\scripts\build_compiled_backend.ps1 -PythonExe C:\Users\you\anaconda3\envs\your_env\python.exe
```

This keeps the portable `win-unpacked` flow, but packages compiled `.pyd` modules instead of the raw `realtime`, `llm_module`, and `voxcpm` source trees.

`npm run dist` now produces the portable `win-unpacked` directory only. It does not build an NSIS installer or an extra zip archive.

For Windows offline distribution, see:

- [PACKAGING.zh-CN.md](./PACKAGING.zh-CN.md)
