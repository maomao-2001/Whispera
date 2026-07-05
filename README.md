[简体中文](./README.md) | [English](./README.en.md)

# Whispera

Whispera 是一个面向 Windows 的本地实时语音助手桌面应用。

项目当前支持本地麦克风输入、VAD 打断、SenseVoice ASR、`llama-server` 本地大模型推理、可选的 VoxCPM 流式 TTS，以及可选的 `mem0` 长期记忆集成。

演示视频：<https://www.bilibili.com/video/BV1nFE36mEmc>

## 概览

- Electron 负责桌面界面、配置、日志和本地服务编排。
- Python `realtime/` 负责 VAD、ASR、LLM、TTS、WebSocket 协议和会话状态。
- 支持本地 `llama-server`、可选 VoxCPM 流式 TTS、可选 `mem0` 长期记忆。
- 大模型、ASR/TTS 权重和分发资源不随 Git 仓库提交，需要单独准备。

## 运行要求

- Windows 10/11
- PowerShell
- Node.js
- 本地可用的 Python 运行时
- 本地模型与资源文件

推荐使用你自己的 Python 或 conda 环境进行开发。`runtime/python/` 主要用于便携 runtime 打包；如果你明确想指定运行时，也可以设置 `MINIMIND_PYTHON`。

ASR 默认使用 `cuda`。如果机器没有可用 GPU，或你想强制走 CPU，请显式设置 `MINIMIND_ASR_DEVICE=cpu`。

## 快速开始

### 0. 获取代码

本仓库使用 Git LFS 管理部分大文件。拉取代码前请先安装并初始化 Git LFS，然后按下面步骤获取仓库：

```powershell
git lfs install
git clone <repo-url>
cd <repo-dir>
git lfs pull
```

如果你已经 clone 过仓库，但发现拿到的是 LFS pointer 文件，再补执行一次 `git lfs pull` 即可。

### 1. 准备 Python 环境并安装依赖

Whispera 默认要求使用 GPU 版 PyTorch。首次安装时不要换成 CPU-only 的 `torch`，否则 ASR 无法按默认配置正常工作。

根目录执行：

```powershell
conda create -n whispera python=3.11 -y
conda activate whispera
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
npm install --prefix electron-app
```

安装完成后，建议先验证一次 PyTorch 确实是 CUDA 版本，再启动项目：

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

期望结果：

- `torch.__version__` 类似 `2.6.0+cu126`
- `torch.version.cuda` 有值，例如 `12.6`
- `torch.cuda.is_available()` 输出 `True`

如果你不使用 conda，也可以直接用系统 Python 安装。只有在你想强制指定某个 Python 时，才需要设置：

```powershell
$env:MINIMIND_PYTHON="C:\Users\you\anaconda3\envs\your_env\python.exe"
```

然后再安装依赖和运行。

另外，首次启动前建议显式清理一次下面这个环境变量，避免 Electron 被误当成普通 Node 进程启动：

```powershell
Remove-Item Env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
```

### 2. 准备本地资源

下面这些大资源默认不在 Git 仓库里，需要你自行准备：

- 资源包下载：<https://pan.quark.cn/s/e3c9253ba232?pwd=TLpM>
- 提取码：`TLpM`
- 下载后请解压到仓库根目录，使目录结构变为 `assets/`、`assets/llama-bin/`、`assets/llm/` 等。

```text
assets/
  llama-bin/
    llama-server.exe
  llm/
    *.gguf
  embedding/
    *.gguf                     # 仅 memory 模式需要
  asr/
    SenseVoiceSmall/
  tts/
    openbmb__VoxCPM1.5/
  lora/
  reference/
```

说明：

- `model/vad/silero_vad.onnx` 由仓库提供
- 其余 LLM、ASR、TTS、embedding 等大资源需要本地拷贝
- 如果 `assets/llm/` 下只有一个 `*.gguf`，Electron 会自动使用它
- `assets/llm/` 下必须至少有一个 `*.gguf`，否则 `llama-server` 无法启动

### 3. 启动开发版

```powershell
npm run dev
```

这条命令会转发到 `electron-app/` 下的 `npm run dev`。

启动成功后，核心服务端口通常是：

```text
llama-server:      http://127.0.0.1:8080
memory embedding:  http://127.0.0.1:8081
realtime backend:  http://127.0.0.1:8011
realtime ws:       ws://127.0.0.1:8011/ws/realtime
web client:        http://127.0.0.1:8012/
```

## 核心功能

- Electron 桌面前端，带资源状态、日志和配置面板
- 本地 `llama-server` 服务拉起与复用
- Silero VAD + barge-in 打断
- SenseVoice ASR
- 流式文本输出与按句切分
- 可选 VoxCPM 流式 TTS
- 可选 `mem0` 长期记忆检索与保存
- Windows 便携目录打包流程

## 系统架构

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

Electron 开发态启动时会按需拉起这些本地服务：

1. `llm-module/scripts/start_llama_server.py`
2. 可选 memory embedding `llama-server`
3. `python -m realtime.app`
4. 本地 Web client 和 Electron renderer

## 仓库结构

```text
electron-app/                    # Electron 前端、打包配置、进程编排
realtime/                        # Python 实时后端
llm-module/                      # llama-server 启动与本地客户端
voxcpm-tts-streaming-module/     # VoxCPM TTS 模块
mem0/                            # vendored mem0 SDK 源码
model/vad/                       # Silero VAD 模型
runtime/                         # 便携 Python runtime 说明与产物目录
scripts/                         # 打包、runtime 导出、后端编译脚本
assets/                          # 本地资源目录，不提交 Git
distribution-assets/             # 分发资源整理目录，不提交 Git
```

## 常用环境变量

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

其中 `MINIMIND_ASR_DEVICE` 不设置时默认是 `cuda`。

## 可选记忆模块

仓库已经将 `mem0/` 作为源码目录纳入主仓库管理，用于本地长期记忆能力。相关本地改动说明见 [mem0/LOCAL_CHANGES.md](./mem0/LOCAL_CHANGES.md)。

需要注意两层开关：

- `MINIMIND_MEMORY_ENABLED`：整个记忆模块总开关
- `MINIMIND_MEMORY_INFER`：是否对保存内容做记忆提取推理

如果你不想启用记忆模块，建议显式关闭：

```powershell
$env:MINIMIND_MEMORY_ENABLED="0"
npm run dev
```

如果你想启用它：

```powershell
$env:MINIMIND_MEMORY_ENABLED="1"
$env:MINIMIND_MEMORY_EMBEDDER_MODEL_PATH="D:\models\embedding.gguf"
$env:MINIMIND_MEMORY_EMBEDDING_DIMS="1024"
npm run dev
```

补充说明：

- `requirements.txt` 已经包含 `requirements-mem0.txt`
- `requirements-mem0.txt` 会把本地 `./mem0` 作为 editable package 安装
- memory 数据默认写入 `runtime/mem0/qdrant` 和 `runtime/mem0/history.db`
- 只想关闭“记忆提取”，而不是整个模块时，可设置 `MINIMIND_MEMORY_INFER=0`

## 打包与发布

如果你要构建 Windows 便携版，建议按下面顺序执行：

### 1. 准备便携 Python runtime

```powershell
.\scripts\pack_runtime.ps1 -ReplaceExisting
```

可选瘦身：

```powershell
.\scripts\slim_runtime_for_distribution.ps1 -RuntimeRoot runtime\python
```

### 2. 编译后端

```powershell
.\scripts\build_compiled_backend.ps1
```

如果需要显式指定 Python：

```powershell
.\scripts\build_compiled_backend.ps1 -PythonExe C:\Users\you\python.exe
```

### 3. 生成便携目录

```powershell
npm run dist --prefix electron-app
```

输出目录：

```text
electron-app/dist/win-unpacked/
```

最终分发通常包含两部分：

1. `electron-app/dist/win-unpacked/`
2. 单独准备好的 `assets/` 资源目录

## 资源与 Git 策略

以下内容按当前设计不进入 Git：

- `assets/`
- `distribution-assets/`
- `runtime/python/`
- `runtime/mem0/`
- `build/compiled-backend/`
- `electron-app/dist/`
- `logs/`

仓库保留的是代码、脚本、打包流程和文档；真正的模型、权重和大资源由使用者自行准备。

## 参考项目

本项目参考或集成了以下开源项目：

- [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM)：流式 TTS 能力来源
- [jingyaogong/minimind-o](https://github.com/jingyaogong/minimind-o)：底层参考
- [mem0ai/mem0](https://github.com/mem0ai/mem0)：长期记忆模块来源

## 相关文档

- [codebase-onboarding-report.md](./codebase-onboarding-report.md)：代码库运行链路分析
- [electron-app/README.md](./electron-app/README.md)：Electron 侧说明
- [runtime/README.md](./runtime/README.md)：便携 runtime 说明
- [mem0/LOCAL_CHANGES.md](./mem0/LOCAL_CHANGES.md)：vendored mem0 本地改动记录
