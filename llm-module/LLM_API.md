# llm-module API 接口文档

本文档描述 `llm-module` 暴露给其他项目使用的接口，包括 Python 客户端、HTTP 接口、SSE 流式接口和 WebSocket 流式接口。

## 1. 服务架构

`llm-module` 本身不直接加载 GGUF 模型，而是调用 llama.cpp 的 `llama-server`。

默认端口：

| 服务 | 默认地址 | 说明 |
| --- | --- | --- |
| llama-server | `http://127.0.0.1:8080` | 上游 OpenAI-compatible LLM 服务 |
| llm-module service | `http://127.0.0.1:8004` | 本项目提供的轻量代理服务 |

启动本地 llama-server：

```bash
python scripts/start_llama_server.py
```

启动 llm-module 代理服务：

```bash
python scripts/llm_server.py --upstream-base-url http://127.0.0.1:8080 --port 8004
```

## 2. Python 客户端接口

### 2.1 导入

```python
from llm_module import LLMConfig, LlamaServerLLM
```

如果调用脚本不在本项目目录下，请先安装：

```bash
python -m pip install -e D:\project\llm-module
```

### 2.2 LLMConfig

```python
LLMConfig(
    base_url="http://127.0.0.1:8080",
    model=None,
    system_prompt="你是一个简洁、自然、可靠的中文助手。请直接回答用户问题。",
    temperature=0.7,
    top_p=0.9,
    max_tokens=512,
    timeout=600.0,
    api_key=None,
    reasoning_budget=None,
    reasoning_format=None,
)
```

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `base_url` | `str` | `http://127.0.0.1:8080` | 上游 llama-server 地址 |
| `model` | `str / None` | `None` | 模型 ID；为空时自动取 `/v1/models` 第一个模型 |
| `system_prompt` | `str` | 默认中文助手提示词 | 默认系统提示词 |
| `temperature` | `float` | `0.7` | 采样温度 |
| `top_p` | `float` | `0.9` | nucleus sampling 参数 |
| `max_tokens` | `int` | `512` | 最大输出 token 数 |
| `timeout` | `float` | `600.0` | 上游请求超时时间，单位秒 |
| `api_key` | `str / None` | `None` | 透传给上游的 Bearer Token |
| `reasoning_budget` | `int / None` | `None` | llama-server reasoning budget |
| `reasoning_format` | `str / None` | `None` | llama-server reasoning format |

### 2.3 非流式生成

```python
llm = LlamaServerLLM(LLMConfig(base_url="http://127.0.0.1:8080"))

text = llm.generate_text(
    text="用一句话介绍一下本地大模型。",
    max_tokens=128,
)

print(text)
```

### 2.4 流式生成

```python
for delta in llm.generate_stream("写一句简短欢迎语。"):
    print(delta.text, end="", flush=True)
```

`generate_stream()` 返回 `LLMDelta`：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `index` | `int` | delta 序号 |
| `text` | `str` | 可见输出文本 |
| `reasoning_text` | `str` | 推理文本，如果上游返回 |
| `model` | `str / None` | 实际使用的模型 |
| `raw_event` | `dict / None` | 上游原始 SSE 事件 |

### 2.5 多轮 messages

```python
messages = [
    {"role": "system", "content": "你是一个简洁的中文助手。"},
    {"role": "user", "content": "给我三个项目命名建议。"},
]

text = llm.generate_text(messages=messages)
```

支持的 role：

```text
system, user, assistant, tool
```

## 3. HTTP 接口

以下接口由 `scripts/llm_server.py` 启动的代理服务提供。

### 3.1 GET `/`

服务基础信息。

响应示例：

```json
{
  "service": "llm-module",
  "websocket_path": "/ws/llm",
  "upstream_base_url": "http://127.0.0.1:8080",
  "upstream_model": null
}
```

### 3.2 GET `/health`

检查代理服务和上游 llama-server 状态。

响应示例：

```json
{
  "status": "ok",
  "service": "llm-module",
  "upstream_base_url": "http://127.0.0.1:8080",
  "upstream_model": null,
  "upstream_ok": true,
  "upstream_models": ["Qwen3.5-4B-Q4_K_M.gguf"],
  "error_message": null
}
```

### 3.3 GET `/upstream/models`

列出上游 llama-server 模型。

响应示例：

```json
{
  "count": 1,
  "models": ["Qwen3.5-4B-Q4_K_M.gguf"]
}
```

### 3.4 POST `/chat`

非流式文本生成。

请求体：

```json
{
  "text": "用一句话介绍一下本地大模型。",
  "model": null,
  "system_prompt": null,
  "prompt_preset": "chat_compatible",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256,
  "reasoning_budget": null,
  "reasoning_format": null
}
```

也可以用 `messages` 替代 `text`：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个简洁的中文助手。"},
    {"role": "user", "content": "你好"}
  ],
  "max_tokens": 128
}
```

请求字段：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `text` | `str` | 与 `messages` 二选一 | 用户输入文本 |
| `messages` | `array` | 与 `text` 二选一 | OpenAI chat messages |
| `model` | `str / null` | 否 | 指定上游模型 ID |
| `system_prompt` | `str / null` | 否 | 覆盖默认系统提示词 |
| `prompt_preset` | `str` | 否 | `chat_compatible` 或 `voice_strict` |
| `temperature` | `float` | 否 | 采样温度 |
| `top_p` | `float` | 否 | top-p 参数 |
| `max_tokens` | `int` | 否 | 最大输出 token 数 |
| `reasoning_budget` | `int / null` | 否 | 上游 reasoning budget |
| `reasoning_format` | `str / null` | 否 | 上游 reasoning format |

响应体：

```json
{
  "model": "Qwen3.5-4B-Q4_K_M.gguf",
  "text": "本地大模型是在你自己的设备上运行、可离线调用的语言模型。",
  "reasoning": null,
  "delta_count": 12,
  "elapsed_ms": 156.68
}
```

curl 示例：

```bash
curl -X POST http://127.0.0.1:8004/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"用一句话介绍一下本地大模型。\",\"max_tokens\":128}"
```

### 3.5 POST `/chat/stream`

SSE 流式文本生成。

请求体与 `/chat` 相同。

curl 示例：

```bash
curl -N -X POST http://127.0.0.1:8004/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"写一句简短欢迎语。\",\"max_tokens\":128}"
```

事件格式：

```text
data: {"type":"llm.started","model":"Qwen3.5-4B-Q4_K_M.gguf"}

data: {"type":"llm.delta","index":0,"text":"你好","reasoning":null,"model":"Qwen3.5-4B-Q4_K_M.gguf"}

data: {"type":"llm.completed","model":"Qwen3.5-4B-Q4_K_M.gguf","text":"你好，欢迎使用本地大模型。","reasoning":null,"delta_count":8,"elapsed_ms":120.5}

data: [DONE]
```

事件类型：

| type | 说明 |
| --- | --- |
| `llm.started` | 生成开始 |
| `llm.delta` | 增量文本 |
| `llm.completed` | 生成完成 |
| `[DONE]` | SSE 结束标记 |

## 4. WebSocket 接口

地址：

```text
ws://127.0.0.1:8004/ws/llm
```

### 4.1 连接建立

服务端连接后立即发送：

```json
{
  "type": "server.ready",
  "protocol_version": "0.1.0",
  "websocket_path": "/ws/llm",
  "supported_client_messages": ["session.start", "chat.text", "ping"]
}
```

### 4.2 开始会话

客户端发送：

```json
{
  "type": "session.start",
  "session_id": "demo-session"
}
```

服务端返回：

```json
{
  "type": "session.ready",
  "session_id": "demo-session",
  "state": "ready",
  "request_count": 0,
  "active_request_id": null
}
```

### 4.3 发送文本生成请求

客户端发送：

```json
{
  "type": "chat.text",
  "session_id": "demo-session",
  "request_id": "req-001",
  "text": "用一句话介绍一下本地大模型。",
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 256
}
```

服务端事件：

```json
{
  "type": "llm.started",
  "session_id": "demo-session",
  "request_id": "req-001",
  "model": "Qwen3.5-4B-Q4_K_M.gguf"
}
```

```json
{
  "type": "llm.delta",
  "session_id": "demo-session",
  "request_id": "req-001",
  "index": 0,
  "text": "本地",
  "reasoning": null,
  "model": "Qwen3.5-4B-Q4_K_M.gguf"
}
```

```json
{
  "type": "llm.completed",
  "session_id": "demo-session",
  "request_id": "req-001",
  "model": "Qwen3.5-4B-Q4_K_M.gguf",
  "text": "本地大模型是在你自己的设备上运行、可离线调用的语言模型。",
  "reasoning": null,
  "delta_count": 12,
  "elapsed_ms": 156.68
}
```

### 4.4 ping

客户端发送：

```json
{"type": "ping"}
```

服务端返回：

```json
{"type": "pong", "session_id": "demo-session"}
```

### 4.5 WebSocket 客户端示例

```bash
python scripts/llm_client.py --text "你好，讲一个一句话冷知识。"
```

## 5. 错误响应

HTTP 接口遇到上游连接失败、请求格式错误等情况时，会返回 FastAPI 默认错误响应。

WebSocket 错误事件格式：

```json
{
  "type": "error",
  "session_id": "demo-session",
  "request_id": "req-001",
  "message": "unsupported message type, expected 'chat.text'"
}
```

常见问题：

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'llm_module'` | 当前 Python 环境没有安装本项目 | 执行 `python -m pip install -e D:\project\llm-module` |
| `failed to connect to llama-server` | 上游 llama-server 没启动 | 先运行 `python scripts/start_llama_server.py` |
| Git clone 后缺少 DLL/EXE | 没拉取 LFS 对象 | 执行 `git lfs pull` |
