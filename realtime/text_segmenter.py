from __future__ import annotations


class StreamingTextSegmenter:
    """Cut incremental LLM text into TTS-friendly chunks."""

    def __init__(self, soft_limit: int = 24):
        self.soft_limit = soft_limit
        self.buffer = ""
        self.hard_punct = set(".!?。！？")
        self.soft_punct = set(",;:，、；：")

    def feed(self, text_delta: str) -> list[str]:
        chunks: list[str] = []
        if not text_delta:
            return chunks
        for char in text_delta:
            self.buffer += char
            if char in self.hard_punct:
                chunk = self._flush()
                if chunk:
                    chunks.append(chunk)
            elif char in self.soft_punct and len(self.buffer.strip()) >= self.soft_limit:
                chunk = self._flush()
                if chunk:
                    chunks.append(chunk)
            elif len(self.buffer.strip()) >= self.soft_limit * 2:
                chunk = self._flush()
                if chunk:
                    chunks.append(chunk)
        return chunks

    def flush(self) -> str:
        return self._flush()

    def reset(self) -> None:
        self.buffer = ""

    def _flush(self) -> str:
        value = self.buffer.strip()
        self.buffer = ""
        return value

