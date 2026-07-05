# VoxCPM Minimal Runtime

This repository has been reduced to a small runtime module that keeps only:

- SenseVoice ASR
- VoxCPM streaming TTS

Removed from this trimmed version:

- CLI command `voxcpm --text ...`
- Web UI
- Training scripts
- VAD / LLM / assistant pipeline
- Demo scripts and extra docs

## Core Files

Main runtime files:

- `src/voxcpm/asr.py`
- `src/voxcpm/asr_service.py`
- `src/voxcpm/core.py`
- `src/voxcpm/streaming_service.py`
- `src/voxcpm/session_protocol.py`

## Setup

Use the existing conda environment:

```powershell
conda activate voxcpm
cd D:\project\voxcpm-tts-streaming
pip install -e ".[streaming]"
```

Verify that the package points to this repository:

```powershell
python -c "import voxcpm; print(voxcpm.__file__)"
```

Expected output:

```text
D:\project\voxcpm-tts-streaming\src\voxcpm\__init__.py
```

## Model Files

### TTS Model

Default local model directory:

```text
models/openbmb__VoxCPM1.5
```

Required files:

```text
config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
audiovae.pth
model.safetensors
```

If `model.safetensors` is not present, `pytorch_model.bin` can be used instead.

### ASR Model

Default ASR model:

```text
iic/SenseVoiceSmall
```

If it is already cached in the environment, it will be loaded directly.

## Start Services

### Streaming TTS

```powershell
conda activate voxcpm
cd D:\project\voxcpm-tts-streaming
python -m voxcpm.streaming_service --host 127.0.0.1 --port 8000
```

Default WebSocket path:

```text
/ws/tts
```

### ASR

```powershell
conda activate voxcpm
cd D:\project\voxcpm-tts-streaming
python -m voxcpm.asr_service --host 127.0.0.1 --port 8003
```

Default WebSocket path:

```text
/ws/asr
```

## Health Checks

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8003/health
```

Expected fields:

- TTS: `websocket_path = /ws/tts`
- ASR: `websocket_path = /ws/asr`

## Minimal Verification

### TTS Check

```powershell
conda activate voxcpm
python -c "from voxcpm.core import VoxCPM; import soundfile as sf; m=VoxCPM(voxcpm_model_path='models/openbmb__VoxCPM1.5', enable_denoiser=False); wav=m.generate(text='Hello, this is a minimal TTS check.'); sf.write('tts_test.wav', wav, m.tts_model.sample_rate); print('ok -> tts_test.wav')"
```

This should create:

```text
tts_test.wav
```

### ASR Check

```powershell
conda activate voxcpm
python -c "from voxcpm.asr import SenseVoiceASR; a=SenseVoiceASR(); r=a.transcribe_path('tts_test.wav'); print(r.text)"
```

## Use as a Module

### TTS

```python
from voxcpm.core import VoxCPM

model = VoxCPM(
    voxcpm_model_path="models/openbmb__VoxCPM1.5",
    enable_denoiser=False,
)

audio = model.generate(text="Hello from VoxCPM.")
```

### ASR

```python
from voxcpm.asr import SenseVoiceASR

asr = SenseVoiceASR()
result = asr.transcribe_path("tts_test.wav")
print(result.text)
```

### FastAPI / WebSocket Apps

```python
from voxcpm.streaming_service import create_app as create_tts_app
from voxcpm.asr_service import create_app as create_asr_app

tts_app = create_tts_app()
asr_app = create_asr_app()
```

## Notes

- This reduced repository is intended to be embedded into another project.
- For CLI synthesis, training, Web UI, VAD, LLM, or the full assistant pipeline, use the original full project.
