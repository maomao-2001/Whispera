import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional


PROTOCOL_VERSION = "0.7.0"


class SessionState(str, Enum):
    READY = "ready"
    CLOSED = "closed"


class RequestState(str, Enum):
    QUEUED = "queued"
    STARTED = "started"
    STREAMING = "streaming"
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"
    FAILED = "failed"


def create_session_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


def create_request_id() -> str:
    return f"req-{uuid.uuid4().hex[:8]}"


@dataclass
class RequestRecord:
    session_id: str
    request_id: str
    request_type: str = "tts"
    state: RequestState = RequestState.QUEUED
    text: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    sample_rate: Optional[int] = None
    chunk_count: int = 0
    total_samples: int = 0
    interrupted_at: Optional[float] = None
    stop_reason: Optional[str] = None
    error_message: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        elapsed_ms = None
        if self.started_at is not None and self.completed_at is not None:
            elapsed_ms = round((self.completed_at - self.started_at) * 1000.0, 2)

        audio_duration_ms = None
        if self.sample_rate is not None and self.sample_rate > 0:
            audio_duration_ms = round((self.total_samples / self.sample_rate) * 1000.0, 2)

        rtf = None
        if elapsed_ms is not None and elapsed_ms > 0 and audio_duration_ms is not None:
            rtf = round(audio_duration_ms / elapsed_ms, 4)

        return {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "request_type": self.request_type,
            "state": self.state.value,
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "sample_rate": self.sample_rate,
            "chunk_count": self.chunk_count,
            "total_samples": self.total_samples,
            "interrupted_at": self.interrupted_at,
            "stop_reason": self.stop_reason,
            "error_message": self.error_message,
            "elapsed_ms": elapsed_ms,
            "audio_duration_ms": audio_duration_ms,
            "rtf": rtf,
        }


@dataclass
class SessionRecord:
    session_id: str
    state: SessionState = SessionState.READY
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    active_request_id: Optional[str] = None
    request_ids: List[str] = field(default_factory=list)

    def snapshot(self, requests: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        request_snapshots = requests if requests is not None else []
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "active_request_id": self.active_request_id,
            "request_count": len(self.request_ids),
            "requests": request_snapshots,
        }


class SessionStore:
    """Minimal in-memory protocol/session store for module 04.

    Scope intentionally stays small:
    - track one or more lightweight websocket sessions
    - track request lifecycle transitions
    - expose serializable snapshots for HTTP inspection and demo validation
    """

    def __init__(self):
        self._lock = Lock()
        self._sessions: Dict[str, SessionRecord] = {}
        self._requests: Dict[str, RequestRecord] = {}

    def start_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            session_id = session_id or create_session_id()
            now = time.time()

            if session_id in self._sessions:
                session = self._sessions[session_id]
                session.state = SessionState.READY
                session.updated_at = now
            else:
                session = SessionRecord(session_id=session_id, created_at=now, updated_at=now)
                self._sessions[session_id] = session

            return self._session_snapshot_unlocked(session_id)

    def close_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None

            session.state = SessionState.CLOSED
            session.updated_at = time.time()
            return self._session_snapshot_unlocked(session_id)

    def register_request(
        self,
        session_id: str,
        request_id: Optional[str] = None,
        request_type: str = "tts",
        text: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            session = self._require_session_unlocked(session_id)
            request_id = request_id or create_request_id()
            if request_id in self._requests:
                raise ValueError(f"request_id already exists: {request_id}")

            now = time.time()
            request = RequestRecord(
                session_id=session_id,
                request_id=request_id,
                request_type=request_type,
                text=text,
                created_at=now,
                updated_at=now,
            )
            self._requests[request_id] = request
            session.request_ids.append(request_id)
            session.active_request_id = request_id
            session.updated_at = now
            return request.snapshot()

    def mark_request_started(self, session_id: str, request_id: str, sample_rate: Optional[int] = None) -> Dict[str, Any]:
        with self._lock:
            request = self._require_request_unlocked(session_id, request_id)
            if request.state in {RequestState.COMPLETED, RequestState.INTERRUPTED, RequestState.FAILED}:
                return request.snapshot()
            now = time.time()
            request.state = RequestState.STARTED
            request.started_at = request.started_at or now
            request.updated_at = now
            request.sample_rate = sample_rate
            self._touch_session_unlocked(session_id, request_id)
            return request.snapshot()

    def mark_request_streaming(self, session_id: str, request_id: str, num_samples: int) -> Dict[str, Any]:
        with self._lock:
            request = self._require_request_unlocked(session_id, request_id)
            if request.state in {RequestState.COMPLETED, RequestState.INTERRUPTED, RequestState.FAILED}:
                return request.snapshot()
            now = time.time()
            request.state = RequestState.STREAMING
            request.started_at = request.started_at or now
            request.updated_at = now
            request.chunk_count += 1
            request.total_samples += int(num_samples)
            self._touch_session_unlocked(session_id, request_id)
            return request.snapshot()

    def mark_request_completed(self, session_id: str, request_id: str, text: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            request = self._require_request_unlocked(session_id, request_id)
            if request.state in {RequestState.INTERRUPTED, RequestState.FAILED}:
                return request.snapshot()
            if request.state == RequestState.COMPLETED:
                if text is not None:
                    request.text = text
                return request.snapshot()
            now = time.time()
            request.state = RequestState.COMPLETED
            request.started_at = request.started_at or now
            request.completed_at = now
            request.updated_at = now
            if text is not None:
                request.text = text

            session = self._require_session_unlocked(session_id)
            if session.active_request_id == request_id:
                session.active_request_id = None
            session.updated_at = now

            return request.snapshot()

    def mark_request_interrupted(
        self,
        session_id: str,
        request_id: str,
        reason: str = "client_interrupt",
    ) -> Dict[str, Any]:
        with self._lock:
            request = self._require_request_unlocked(session_id, request_id)
            now = time.time()
            request.state = RequestState.INTERRUPTED
            request.started_at = request.started_at or now
            request.completed_at = now
            request.interrupted_at = now
            request.updated_at = now
            request.stop_reason = reason
            request.error_message = None

            session = self._require_session_unlocked(session_id)
            if session.active_request_id == request_id:
                session.active_request_id = None
            session.updated_at = now

            return request.snapshot()

    def mark_request_failed(self, session_id: str, request_id: str, error_message: str) -> Dict[str, Any]:
        with self._lock:
            request = self._require_request_unlocked(session_id, request_id)
            if request.state in {RequestState.COMPLETED, RequestState.INTERRUPTED, RequestState.FAILED}:
                return request.snapshot()
            now = time.time()
            request.state = RequestState.FAILED
            request.started_at = request.started_at or now
            request.completed_at = now
            request.updated_at = now
            request.error_message = error_message

            session = self._require_session_unlocked(session_id)
            if session.active_request_id == request_id:
                session.active_request_id = None
            session.updated_at = now

            return request.snapshot()

    def interrupt_active_request(
        self,
        session_id: str,
        request_type: Optional[str] = None,
        reason: str = "client_interrupt",
    ) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.active_request_id is None:
                return None

            request = self._requests.get(session.active_request_id)
            if request is None:
                session.active_request_id = None
                session.updated_at = time.time()
                return None

            if request_type is not None and request.request_type != request_type:
                return None

            if request.state in {RequestState.COMPLETED, RequestState.INTERRUPTED, RequestState.FAILED}:
                session.active_request_id = None
                session.updated_at = time.time()
                return request.snapshot()

            now = time.time()
            request.state = RequestState.INTERRUPTED
            request.started_at = request.started_at or now
            request.completed_at = now
            request.interrupted_at = now
            request.updated_at = now
            request.stop_reason = reason
            request.error_message = None
            session.active_request_id = None
            session.updated_at = now
            return request.snapshot()

    def get_request_snapshot(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return None
            return request.snapshot()

    def get_active_request_id(self, session_id: str) -> Optional[str]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return session.active_request_id

    def is_request_interrupted(self, session_id: str, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.session_id != session_id:
                return False
            return request.state == RequestState.INTERRUPTED

    def get_session_snapshot(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if session_id not in self._sessions:
                return None
            return self._session_snapshot_unlocked(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._session_snapshot_unlocked(session_id) for session_id in sorted(self._sessions)]

    def _touch_session_unlocked(self, session_id: str, active_request_id: Optional[str]) -> None:
        session = self._require_session_unlocked(session_id)
        session.updated_at = time.time()
        session.active_request_id = active_request_id

    def _require_session_unlocked(self, session_id: str) -> SessionRecord:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"session not found: {session_id}")
        return session

    def _require_request_unlocked(self, session_id: str, request_id: str) -> RequestRecord:
        request = self._requests.get(request_id)
        if request is None:
            raise ValueError(f"request not found: {request_id}")
        if request.session_id != session_id:
            raise ValueError(f"request {request_id} does not belong to session {session_id}")
        return request

    def _session_snapshot_unlocked(self, session_id: str) -> Dict[str, Any]:
        session = self._require_session_unlocked(session_id)
        requests = [self._requests[request_id].snapshot() for request_id in session.request_ids if request_id in self._requests]
        return session.snapshot(requests=requests)


def build_server_ready_message(
    websocket_path: str,
    supported_client_messages: Optional[List[str]] = None,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    supported_client_messages = supported_client_messages or ["session.start", "tts.start", "ping"]
    message = message or "send 'session.start' to establish a session, then send 'tts.start' to begin streaming synthesis"
    return {
        "type": "server.ready",
        "protocol_version": PROTOCOL_VERSION,
        "websocket_path": websocket_path,
        "supported_client_messages": supported_client_messages,
        "message": message,
    }


def build_session_ready_message(session_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "session.ready",
        "session_id": session_snapshot["session_id"],
        "state": session_snapshot["state"],
        "request_count": session_snapshot["request_count"],
        "active_request_id": session_snapshot["active_request_id"],
    }


def build_request_state_message(request_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "request.state",
        "session_id": request_snapshot["session_id"],
        "request_id": request_snapshot["request_id"],
        "request_type": request_snapshot["request_type"],
        "state": request_snapshot["state"],
        "chunk_count": request_snapshot["chunk_count"],
        "total_samples": request_snapshot["total_samples"],
        "sample_rate": request_snapshot["sample_rate"],
        "elapsed_ms": request_snapshot["elapsed_ms"],
        "audio_duration_ms": request_snapshot["audio_duration_ms"],
        "rtf": request_snapshot["rtf"],
        "interrupted_at": request_snapshot["interrupted_at"],
        "stop_reason": request_snapshot["stop_reason"],
        "error_message": request_snapshot["error_message"],
    }


def build_tts_started_message(session_id: str, request_id: str, sample_rate: int) -> Dict[str, Any]:
    return {
        "type": "tts.started",
        "session_id": session_id,
        "request_id": request_id,
        "sample_rate": sample_rate,
        "audio_format": "pcm_f32le",
    }


def build_tts_chunk_message(
    session_id: str,
    request_id: str,
    index: int,
    num_samples: int,
    data: str,
    audio_format: str = "pcm_f32le",
) -> Dict[str, Any]:
    return {
        "type": "tts.chunk",
        "session_id": session_id,
        "request_id": request_id,
        "index": index,
        "audio_format": audio_format,
        "num_samples": num_samples,
        "data": data,
    }


def build_tts_completed_message(request_snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "tts.completed",
        "session_id": request_snapshot["session_id"],
        "request_id": request_snapshot["request_id"],
        "chunk_count": request_snapshot["chunk_count"],
        "total_samples": request_snapshot["total_samples"],
        "elapsed_ms": request_snapshot["elapsed_ms"],
        "audio_duration_ms": request_snapshot["audio_duration_ms"],
        "rtf": request_snapshot["rtf"],
    }


def build_asr_started_message(
    session_id: str,
    request_id: str,
    sample_rate: int,
    target_sample_rate: int,
    language: str,
    use_itn: bool,
) -> Dict[str, Any]:
    return {
        "type": "asr.started",
        "session_id": session_id,
        "request_id": request_id,
        "sample_rate": sample_rate,
        "target_sample_rate": target_sample_rate,
        "language": language,
        "use_itn": use_itn,
    }


def build_asr_final_message(
    session_id: str,
    request_id: str,
    text: str,
    language: str,
) -> Dict[str, Any]:
    return {
        "type": "asr.final",
        "session_id": session_id,
        "request_id": request_id,
        "text": text,
        "language": language,
    }


def build_asr_completed_message(request_snapshot: Dict[str, Any], target_sample_rate: int) -> Dict[str, Any]:
    return {
        "type": "asr.completed",
        "session_id": request_snapshot["session_id"],
        "request_id": request_snapshot["request_id"],
        "sample_rate": request_snapshot["sample_rate"],
        "target_sample_rate": target_sample_rate,
        "chunk_count": request_snapshot["chunk_count"],
        "total_samples": request_snapshot["total_samples"],
        "elapsed_ms": request_snapshot["elapsed_ms"],
        "audio_duration_ms": request_snapshot["audio_duration_ms"],
        "rtf": request_snapshot["rtf"],
    }


def build_interrupt_ack_message(
    session_id: Optional[str],
    request_id: str,
    interrupted_request_id: Optional[str],
    request_type: Optional[str],
    accepted: bool,
    reason: str,
) -> Dict[str, Any]:
    return {
        "type": "interrupt.ack",
        "session_id": session_id,
        "request_id": request_id,
        "interrupted_request_id": interrupted_request_id,
        "request_type": request_type,
        "accepted": accepted,
        "reason": reason,
    }


def build_error_message(message: str, session_id: Optional[str] = None, request_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        "type": "error",
        "session_id": session_id,
        "request_id": request_id,
        "message": message,
    }
