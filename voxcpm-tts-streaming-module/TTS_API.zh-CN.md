# VoxCPM TTS / ASR 接口文档

本文档描述本仓库中两类 WebSocket 服务的对接方式：

- 流式 TTS：`/ws/tts`
- 流式 ASR：`/ws/asr`

适合直接提供给上层应用或其他 AI，按协议接入。

## 1. 服务地址

### 1.1 流式 TTS

启动命令：

```powershell
conda activate voxcpm
cd voxcpm-tts-streaming-module
python -m voxcpm.streaming_service --host 127.0.0.1 --port 8000
```

WebSocket 地址：

```text
ws://127.0.0.1:8000/ws/tts
```

健康检查：

```text
GET http://127.0.0.1:8000/health
```

### 1.2 流式 ASR

启动命令：

```powershell
conda activate voxcpm
cd voxcpm-tts-streaming-module
python -m voxcpm.asr_service --host 127.0.0.1 --port 8003
```

WebSocket 地址：

```text
ws://127.0.0.1:8003/ws/asr
```

健康检查：

```text
GET http://127.0.0.1:8003/health
```

## 2. 通用协议约定

这两类服务都基于 WebSocket，消息体使用 JSON。

连接成功后，服务端都会先发送一条 `server.ready`。

两类服务都支持：

- `session.start`
- `ping`

都会返回：

- `server.ready`
- `session.ready`
- `request.state`
- `pong`
- `error`

### 2.1 `server.ready`

示例：

```json
{
  "type": "server.ready",
  "protocol_version": "1.0",
  "websocket_path": "/ws/tts",
  "supported_client_messages": ["session.start", "tts.start", "tts.interrupt", "ping"],
  "message": "send 'session.start' to establish a session, then send 'tts.start' to begin streaming synthesis"
}
```

字段说明：

- `protocol_version`：协议版本
- `websocket_path`：当前服务的 WebSocket 路径
- `supported_client_messages`：允许发送的客户端消息类型
- `message`：服务端给调用方的说明文本

### 2.2 `session.start`

用于建立一个会话。

请求示例：

```json
{
  "type": "session.start",
  "session_id": "demo-session-001"
}
```

字段说明：

- `type`：固定为 `session.start`
- `session_id`：可选。建议客户端自己传，便于追踪

### 2.3 `session.ready`

在 `session.start` 后返回。

示例：

```json
{
  "type": "session.ready",
  "session_id": "demo-session-001",
  "state": "active",
  "request_count": 0,
  "active_request_id": null
}
```

### 2.4 `request.state`

用于描述某次请求的状态变化。

示例：

```json
{
  "type": "request.state",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "request_type": "tts",
  "state": "streaming",
  "chunk_count": 3,
  "total_samples": 44100,
  "sample_rate": 44100,
  "elapsed_ms": 1530.2,
  "audio_duration_ms": 1000.0,
  "rtf": 1.53,
  "interrupted_at": null,
  "stop_reason": null,
  "error_message": null
}
```

常见状态值：

- `queued`
- `started`
- `streaming`
- `completed`
- `interrupted`
- `failed`

### 2.5 `ping` / `pong`

心跳请求：

```json
{
  "type": "ping"
}
```

返回：

```json
{
  "type": "pong",
  "session_id": "demo-session-001"
}
```

### 2.6 `error`

当消息非法、参数不完整或处理失败时，服务端会返回：

```json
{
  "type": "error",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "message": "detailed error message"
}
```

## 3. 流式 TTS 接口

### 3.1 客户端可发送的消息

- `session.start`
- `tts.start`
- `tts.interrupt`
- `ping`

### 3.2 服务端可能返回的消息

- `server.ready`
- `session.ready`
- `request.state`
- `tts.started`
- `tts.chunk`
- `tts.completed`
- `interrupt.ack`
- `pong`
- `error`

### 3.3 推荐时序

1. 连接 `ws://127.0.0.1:8000/ws/tts`
2. 接收 `server.ready`
3. 发送 `session.start`
4. 接收 `session.ready`
5. 发送 `tts.start`
6. 接收若干 `request.state`
7. 接收 `tts.started`
8. 循环接收 `tts.chunk`
9. 接收 `tts.completed`

### 3.4 `tts.start`

最小请求：

```json
{
  "type": "tts.start",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "text": "你好，这是一次流式 TTS 验证。"
}
```

完整请求示例：

```json
{
  "type": "tts.start",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "text": "你好，这是一次流式 TTS 验证。",
  "prompt_audio_path": "D:/audio/reference.wav",
  "prompt_text": "你好，这是参考音频对应的文本。",
  "cfg_value": 2.0,
  "inference_timesteps": 10,
  "min_len": 2,
  "max_len": 4096,
  "normalize": false,
  "denoise": false,
  "streaming_prefix_len": 3,
  "streaming_emit_interval": 4,
  "lora_selection": "",
  "lora_path": ""
}
```

字段说明：

- `type`：固定为 `tts.start`
- `session_id`：可选，但建议始终传
- `request_id`：可选，但建议始终传
- `text`：必填，待合成文本
- `prompt_audio_path`：可选，参考音频路径
- `prompt_text`：可选，参考音频对应文本
- `cfg_value`：可选，默认 `2.0`
- `inference_timesteps`：可选，默认 `10`
- `min_len`：可选，默认 `2`
- `max_len`：可选，默认 `4096`
- `normalize`：可选，默认 `false`
- `denoise`：可选，默认 `false`
- `streaming_prefix_len`：可选，默认 `3`
- `streaming_emit_interval`：可选，默认 `4`
- `lora_selection` / `lora_path`：可选，LoRA 相关参数

注意：

- `text` 不能为空
- `prompt_audio_path` 和 `prompt_text` 最好成对传
- 同一 `session` 同时只能有一个活动中的 TTS 请求
- 如果上一个请求还没完成，新 `tts.start` 会收到错误

### 3.5 `tts.started`

示例：

```json
{
  "type": "tts.started",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "sample_rate": 44100,
  "audio_format": "pcm_f32le"
}
```

字段说明：

- `sample_rate`：本次音频采样率
- `audio_format`：当前固定为 `pcm_f32le`

### 3.6 `tts.chunk`

示例：

```json
{
  "type": "tts.chunk",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "index": 0,
  "audio_format": "pcm_f32le",
  "num_samples": 8192,
  "data": "BASE64_ENCODED_PCM_BYTES"
}
```

字段说明：

- `index`：chunk 序号，从 `0` 递增
- `audio_format`：当前固定为 `pcm_f32le`
- `num_samples`：当前 chunk 内 sample 数量
- `data`：base64 编码后的原始 PCM 数据

### 3.7 `tts.completed`

示例：

```json
{
  "type": "tts.completed",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "chunk_count": 12,
  "total_samples": 96256,
  "elapsed_ms": 3280.4,
  "audio_duration_ms": 2182.68,
  "rtf": 1.5
}
```

### 3.8 `tts.interrupt`

请求示例：

```json
{
  "type": "tts.interrupt",
  "session_id": "demo-session-001",
  "request_id": "req-interrupt-001"
}
```

### 3.9 `interrupt.ack`

示例：

```json
{
  "type": "interrupt.ack",
  "session_id": "demo-session-001",
  "request_id": "req-interrupt-001",
  "interrupted_request_id": "req-001",
  "request_type": "tts",
  "accepted": true,
  "reason": "client_interrupt"
}
```

说明：

- `accepted = true`：表示中断成功
- `accepted = false`：表示当前没有可中断请求

### 3.10 TTS 音频格式

`tts.chunk.data` 的处理方式：

1. 对 `data` 做 base64 解码
2. 按 `float32 little-endian` 解释二进制
3. 得到 1 维 PCM 浮点数组
4. 按 `tts.started.sample_rate` 播放或写入文件

即：

- `audio_format = pcm_f32le`
- 每个 sample 占 4 字节
- 数值范围通常为 `[-1.0, 1.0]`

### 3.11 TTS Python 对接示例

```python
import asyncio
import base64
import json
import numpy as np
import soundfile as sf
import websockets


async def main():
    uri = "ws://127.0.0.1:8000/ws/tts"
    chunks = []
    sample_rate = None

    async with websockets.connect(uri, max_size=None) as ws:
        print(json.loads(await ws.recv()))

        await ws.send(json.dumps({
            "type": "session.start",
            "session_id": "demo-session-001",
        }))
        print(json.loads(await ws.recv()))

        await ws.send(json.dumps({
            "type": "tts.start",
            "session_id": "demo-session-001",
            "request_id": "req-001",
            "text": "你好，这是一次流式 TTS 接口验证。"
        }))

        while True:
            msg = json.loads(await ws.recv())
            msg_type = msg.get("type")
            print("recv:", msg_type)

            if msg_type == "tts.started":
                sample_rate = int(msg["sample_rate"])

            elif msg_type == "tts.chunk":
                raw = base64.b64decode(msg["data"])
                audio = np.frombuffer(raw, dtype=np.float32)
                chunks.append(audio.copy())

            elif msg_type == "tts.completed":
                break

            elif msg_type == "error":
                raise RuntimeError(msg["message"])

        if chunks:
            wav = np.concatenate(chunks)
            sf.write("tts_ws_output.wav", wav, sample_rate)


asyncio.run(main())
```

## 4. 流式 ASR 接口

### 4.1 客户端可发送的消息

- `session.start`
- `audio.append`
- `audio.commit`
- `ping`

### 4.2 服务端可能返回的消息

- `server.ready`
- `session.ready`
- `request.state`
- `asr.started`
- `asr.final`
- `asr.completed`
- `pong`
- `error`

### 4.3 推荐时序

1. 连接 `ws://127.0.0.1:8003/ws/asr`
2. 接收 `server.ready`
3. 发送 `session.start`
4. 接收 `session.ready`
5. 持续发送 `audio.append`
6. 接收若干 `request.state`
7. 接收 `asr.started`
8. 发送 `audio.commit`
9. 接收 `asr.final`
10. 接收 `request.state` 的完成状态
11. 接收 `asr.completed`

### 4.4 `audio.append`

用于持续上传音频块。

最小请求示例：

```json
{
  "type": "audio.append",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "sample_rate": 16000,
  "audio_format": "pcm_f32le",
  "data": "BASE64_ENCODED_PCM_BYTES"
}
```

也支持兼容字段：

```json
{
  "type": "audio.append",
  "request_id": "req-001",
  "sample_rate": 16000,
  "audio_format": "pcm16",
  "pcm16": "BASE64_ENCODED_PCM16_BYTES"
}
```

字段说明：

- `type`：固定为 `audio.append`
- `session_id`：可选，但建议传
- `request_id`：可选，但建议固定一个请求 ID，整段音频保持一致
- `sample_rate`：必填，当前上传音频的采样率
- `audio_format`：可选，支持：
  - `pcm_f32le`
  - `float32`
  - `f32`
  - `pcm16`
  - `pcm_s16le`
  - `pcm16le`
- `data`：base64 编码后的音频数据
- `pcm16` / `pcm_f32le`：兼容写法，也可以作为载荷字段
- `language`：可选，默认 `auto`
- `use_itn`：可选，默认 `true`
- `target_sample_rate`：可选，默认 `16000`

注意：

- 同一个 ASR 请求内，`sample_rate` 必须保持一致
- 音频不能为空
- 多个 `audio.append` 要使用同一个 `request_id`

### 4.5 `audio.commit`

表示当前这段音频已经传完，要求服务端开始识别并返回最终文本。

请求示例：

```json
{
  "type": "audio.commit",
  "session_id": "demo-session-001",
  "request_id": "req-001"
}
```

注意：

- 如果 `request_id` 未知，服务端会返回错误
- `audio.commit` 不上传音频本体，只是结束标记

### 4.6 `asr.started`

表示服务端已接受当前 ASR 请求并开始收流。

示例：

```json
{
  "type": "asr.started",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "sample_rate": 16000,
  "target_sample_rate": 16000,
  "language": "auto",
  "use_itn": true
}
```

### 4.7 `asr.final`

表示最终识别文本。

示例：

```json
{
  "type": "asr.final",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "text": "你好，这是一次 ASR 接口验证。",
  "language": "auto"
}
```

说明：

- 当前实现返回的是最终文本，不是增量转写

### 4.8 `asr.completed`

表示当前识别请求已经结束。

示例：

```json
{
  "type": "asr.completed",
  "session_id": "demo-session-001",
  "request_id": "req-001",
  "sample_rate": 16000,
  "target_sample_rate": 16000,
  "chunk_count": 8,
  "total_samples": 32000,
  "elapsed_ms": 740.5,
  "audio_duration_ms": 2000.0,
  "rtf": 0.37
}
```

### 4.9 ASR 音频格式

服务端支持两类输入音频：

#### `pcm_f32le`

- base64 解码后按 `float32 little-endian` 解释
- 数值通常在 `[-1.0, 1.0]`

#### `pcm16`

- base64 解码后按 `int16 little-endian` 解释
- 服务端会自动归一化到 `float32 / [-1.0, 1.0]`

### 4.10 ASR Python 对接示例

```python
import asyncio
import base64
import json
import numpy as np
import soundfile as sf
import websockets


async def main():
    uri = "ws://127.0.0.1:8003/ws/asr"

    audio, sample_rate = sf.read("tts_test.wav", dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    chunk_size = sample_rate // 2

    async with websockets.connect(uri, max_size=None) as ws:
        print(json.loads(await ws.recv()))

        await ws.send(json.dumps({
            "type": "session.start",
            "session_id": "demo-session-001",
        }))
        print(json.loads(await ws.recv()))

        request_id = "req-001"

        for start in range(0, len(audio), chunk_size):
            chunk = np.ascontiguousarray(audio[start:start + chunk_size], dtype=np.float32)
            payload = base64.b64encode(chunk.tobytes()).decode("ascii")
            await ws.send(json.dumps({
                "type": "audio.append",
                "session_id": "demo-session-001",
                "request_id": request_id,
                "sample_rate": int(sample_rate),
                "audio_format": "pcm_f32le",
                "data": payload
            }))

        await ws.send(json.dumps({
            "type": "audio.commit",
            "session_id": "demo-session-001",
            "request_id": request_id
        }))

        while True:
            msg = json.loads(await ws.recv())
            print("recv:", msg["type"])
            if msg["type"] == "asr.final":
                print("text:", msg["text"])
            elif msg["type"] == "asr.completed":
                break
            elif msg["type"] == "error":
                raise RuntimeError(msg["message"])


asyncio.run(main())
```

## 5. 对接建议

- 建议客户端始终自己传 `session_id` 和 `request_id`
- TTS 和 ASR 都建议按“一个请求一个固定 request_id”处理
- TTS 如果需要再次发起请求，先确认旧请求已完成；或先发送 `tts.interrupt`
- ASR 要保证同一请求内 `sample_rate` 一致
- TTS 播放端要按 `tts.started.sample_rate` 处理音频
- ASR 上传端要显式传 `audio_format`

## 6. 当前实现限制

- 同一 `session` 下，TTS 同时只支持一个活动请求
- TTS 输出当前固定为 `pcm_f32le`
- ASR 当前返回最终识别结果，不返回中间增量文本
- `prompt_audio_path` 是服务端本地路径，不是文件上传接口
- 这套接口面向本地子模块集成，不包含鉴权、限流、租户隔离和持久化队列
