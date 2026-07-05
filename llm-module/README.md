# llm-module

API 文档见：[LLM_API.md](LLM_API.md)

`llama-server` 相关能力，不再包含 TTS、ASR、VAD、SQLite 记忆、训练代码或浏览器语音前端。

## 保留内容

- `llm_module.LlamaServerLLM`: 纯标准库 Python 客户端，调用 llama.cpp 的 OpenAI-compatible 接口。
- `llm_module.service`: 可选 FastAPI 代理服务，提供 `/chat`、`/chat/stream` 和 `/ws/llm`。
- `scripts/start_llama_server.py`: 启动内置 `llama/bin/llama-server.exe`。
- `scripts/llm_demo.py`: 直接验证上游 llama-server。
- `scripts/llm_server.py`: 启动轻量 LLM 代理服务。
- `scripts/llm_client.py`: WebSocket 流式验证客户端。

## 安装

只使用 Python 客户端时不需要额外依赖：

```bash
pip install -e .
```

如果你的调用脚本不在本项目目录里，比如 `D:\project\test.py`，也需要先在当前 Python 环境里执行：

```bash
python -m pip install -e D:\project\llm-module
```

如果要启动 HTTP/SSE/WebSocket 代理服务：

```bash
pip install -e .[server]
```

## 启动 llama-server

仓库使用 Git LFS 保存 `llama/bin/llama-server.exe` 和所需 DLL。克隆后请确保已经安装 Git LFS，并执行 `git lfs pull` 拉取二进制文件。

```bash
python scripts/start_llama_server.py
```

也可以指定其他 GGUF：

```bash
python scripts/start_llama_server.py --model D:\models\your-model.gguf --port 8080
```

## 直接在 Python 中使用

```python
from llm_module import LLMConfig, LlamaServerLLM

llm = LlamaServerLLM(
    LLMConfig(
        base_url="http://127.0.0.1:8080",
        max_tokens=256,
        temperature=0.7,
    )
)

print(llm.generate_text("用一句话介绍一下本地大模型。"))
```

流式：

```python
for delta in llm.generate_stream("写一句简短的欢迎语。"):
    print(delta.text, end="", flush=True)
```

多轮消息：

```python
messages = [
    {"role": "system", "content": "你是一个简洁的中文助手。"},
    {"role": "user", "content": "给我三个项目命名建议。"},
]

text = llm.generate_text(messages=messages)
```

## 运行代理服务

```bash
python scripts/llm_server.py --upstream-base-url http://127.0.0.1:8080 --port 8004
```

普通 HTTP：

```bash
curl -X POST http://127.0.0.1:8004/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"你好，简单介绍一下你自己。\",\"max_tokens\":128}"
```

SSE 流式：

```bash
curl -N -X POST http://127.0.0.1:8004/chat/stream ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"写一段很短的开场白。\"}"
```

WebSocket 验证：

```bash
python scripts/llm_client.py --text "你好，讲个一句话冷知识。"
```

## 直接验证上游 llama-server

```bash
python scripts/llm_demo.py --list-models
python scripts/llm_demo.py --text "你好，简单介绍一下你自己。"
```

## 项目结构

```text
.
├── llama/bin/                 # llama-server.exe 和运行所需 DLL
├── scripts/
│   ├── start_llama_server.py
│   ├── llm_demo.py
│   ├── llm_server.py
│   └── llm_client.py
├── src/llm_module/
│   ├── __init__.py
│   ├── llm.py
│   └── service.py
├── pyproject.toml
└── README.md
```
