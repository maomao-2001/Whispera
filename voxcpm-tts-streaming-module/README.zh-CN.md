# VoxCPM 最小运行版

这个仓库已经被精简成一个可嵌入的最小运行模块，只保留两项能力：

- SenseVoice ASR
- VoxCPM 流式 TTS

这个精简版已经移除：

- `voxcpm --text ...` 命令行入口
- Web UI
- 训练脚本
- VAD / LLM / assistant 整体链路
- 演示脚本和大部分外围文档

## 核心文件

当前保留的主要运行文件：

- `src/voxcpm/asr.py`
- `src/voxcpm/asr_service.py`
- `src/voxcpm/core.py`
- `src/voxcpm/streaming_service.py`
- `src/voxcpm/session_protocol.py`

## 环境准备

建议继续使用现有 conda 环境：

```powershell
conda activate voxcpm
cd D:\project\voxcpm-tts-streaming
pip install -e ".[streaming]"
```

安装完成后，可以确认当前导入路径是否已经指向这个仓库：

```powershell
python -c "import voxcpm; print(voxcpm.__file__)"
```

期望输出类似：

```text
D:\project\voxcpm-tts-streaming\src\voxcpm\__init__.py
```

## 模型文件

### TTS 模型

默认本地模型目录：

```text
models/openbmb__VoxCPM1.5
```

至少需要以下文件：

```text
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
audiovae.pth
model.safetensors
```

如果没有 `model.safetensors`，也可以使用：

```text
pytorch_model.bin
```

### ASR 模型

ASR 默认使用：

```text
iic/SenseVoiceSmall
```

如果环境中已经缓存过，会直接加载。

## 启动服务

### 启动流式 TTS

```powershell
conda activate voxcpm
cd D:\project\voxcpm-tts-streaming

////////首次启动
pip uninstall -y voxcpm
pip install -e ".[streaming]"
////////

python -m voxcpm.streaming_service --host 127.0.0.1 --port 8000
```

默认 WebSocket 路径：

```text
/ws/tts
```

### 启动 ASR

```powershell
conda activate voxcpm
cd D:\project\voxcpm-tts-streaming
python -m voxcpm.asr_service --host 127.0.0.1 --port 8003
```

默认 WebSocket 路径：

```text
/ws/asr
```

## 健康检查

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8003/health
```

正常情况下返回 JSON，其中会包含：

- TTS: `websocket_path = /ws/tts`
- ASR: `websocket_path = /ws/asr`

## 最小验证

### TTS 验证

```powershell
conda activate voxcpm
python -c "from voxcpm.core import VoxCPM; import soundfile as sf; m=VoxCPM(voxcpm_model_path='models/openbmb__VoxCPM1.5', enable_denoiser=False); wav=m.generate(text='你好啊，这是一次最小版 TTS 验证。'); sf.write('tts_test.wav', wav, m.tts_model.sample_rate); print('ok -> tts_test.wav')"
```

成功后会生成：

```text
tts_test.wav
```

### ASR 验证

```powershell
conda activate voxcpm
python -c "from voxcpm.asr import SenseVoiceASR; a=SenseVoiceASR(); r=a.transcribe_path('tts_test.wav'); print(r.text)"
```

## 作为子模块调用

### TTS

```python
from voxcpm.core import VoxCPM

model = VoxCPM(
    voxcpm_model_path="models/openbmb__VoxCPM1.5",
    enable_denoiser=False,
)

audio = model.generate(text="你好，欢迎使用 VoxCPM。")
```

### ASR

```python
from voxcpm.asr import SenseVoiceASR

asr = SenseVoiceASR()
result = asr.transcribe_path("tts_test.wav")
print(result.text)
```

### FastAPI / WebSocket 应用

```python
from voxcpm.streaming_service import create_app as create_tts_app
from voxcpm.asr_service import create_app as create_asr_app

tts_app = create_tts_app()
asr_app = create_asr_app()
```

## 说明

- 这个精简仓库的定位是被别的项目嵌入调用。
- 如果你还需要 CLI 合成、训练、Web UI、VAD、LLM 或完整 assistant 链路，需要回到原始完整工程。
