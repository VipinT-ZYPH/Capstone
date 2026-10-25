"""Offline-testable orchestration and official Gnani STT/TTS adapters."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .conversation import ConversationTurnResult, ConversationalRAGAgent


class VoiceIntegrationError(RuntimeError):
    """Safe error raised when a voice pipeline stage cannot complete."""

    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category
        self.safe_message = message


class SpeechToText(Protocol):
    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str,
    ) -> dict[str, Any]:
        ...


class TextToSpeech(Protocol):
    def synthesize(self, text: str) -> bytes:
        ...


@runtime_checkable
class StreamingSpeechToText(Protocol):
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterable[bytes],
        *,
        language_code: str = "en-IN",
    ) -> str:
        ...


@runtime_checkable
class StreamingTextToSpeech(Protocol):
    def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        ...


@dataclass(frozen=True)
class VoiceTurnResult:
    session_id: str
    transcript: str
    conversation: ConversationTurnResult
    audio: bytes


@dataclass(frozen=True)
class StreamingVoiceTurnResult:
    session_id: str
    transcript: str
    conversation: ConversationTurnResult
    audio_chunks: tuple[bytes, ...]
    audio_complete: bool


@dataclass(frozen=True)
class VoiceAudioEvent:
    """One incremental audio event for a completed conversational turn."""

    session_id: str
    audio: bytes
    complete: bool
    transcript: str = ""
    conversation: ConversationTurnResult | None = None


class GnaniSTTAdapter:
    """Adapter for the official ``gnani.stt.GnaniSTTClient`` contract."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.vachana.ai",
        timeout: int = 60,
    ):
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        from gnani.stt import GnaniSTTClient

        self._client = GnaniSTTClient(api_key=api_key, base_url=base_url, timeout=timeout)

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str,
        language_code: str,
    ) -> dict[str, Any]:
        return self._client.transcribe_bytes(
            audio_bytes,
            filename=filename,
            language_code=language_code,
            format="transcribe",
        )


class GnaniTTSAdapter:
    """Adapter for the official ``gnani.tts.GnaniTTSClient`` contract."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.vachana.ai",
        timeout: int = 60,
        voice: str = "Pranav",
        model: str = "timbre-v2.0",
    ):
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        from gnani.tts import GnaniTTSClient, AudioConfig

        self._client = GnaniTTSClient(api_key=api_key, base_url=base_url, timeout=timeout)
        self._audio_config = AudioConfig(
            sample_rate=48000,
            encoding="linear_pcm",
            container="wav",
        )
        self._voice = voice
        self._model = model

    def synthesize(self, text: str) -> bytes:
        return self._client.synthesize(
            text,
            voice=self._voice,
            model=self._model,
            audio_config=self._audio_config,
        )


class GnaniSTTStreamingAdapter:
    """Adapter for Gnani's verified STT WebSocket streaming client."""

    def __init__(
        self,
        api_key: str,
        *,
        language_code: str = "en-IN",
        sample_rate: int = 16000,
        base_url: str = "https://api.vachana.ai",
        format: str = "transcribe",
    ):
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        from gnani.stt import GnaniSTTStreamClient

        self._client = GnaniSTTStreamClient(
            api_key=api_key,
            language_code=language_code,
            sample_rate=sample_rate,
            base_url=base_url,
            format=format,
        )

    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterable[bytes],
        *,
        language_code: str = "en-IN",
    ) -> str:
        from gnani.stt import StreamTranscriptEvent

        if language_code != self._client.language_code:
            raise ValueError("language_code must match the configured streaming STT client")
        async with self._client as stream:
            events = await stream.stream_audio(audio_chunks, realtime_pace=False)
        transcripts = [
            event.text.strip()
            for event in events
            if isinstance(event, StreamTranscriptEvent) and event.text.strip()
        ]
        if not transcripts:
            raise VoiceIntegrationError("empty-transcript", "speech transcription was empty")
        return " ".join(transcripts)


class GnaniTTSStreamingAdapter:
    """Adapter for Gnani's verified realtime TTS WebSocket client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.vachana.ai",
        voice: str = "Pranav",
        model: str = "timbre-v2.0",
    ):
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        from gnani.tts import AudioConfig, GnaniTTSRealtimeClient

        self._client = GnaniTTSRealtimeClient(api_key=api_key, base_url=base_url)
        self._audio_config = AudioConfig(
            sample_rate=48000,
            encoding="linear_pcm",
            container="wav",
        )
        self._voice = voice
        self._model = model

    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        from gnani.tts import TTSAudioChunkEvent, TTSCompletedEvent

        completed = False
        async for event in self._client.synthesize_events(
            text,
            voice=self._voice,
            model=self._model,
            audio_config=self._audio_config,
        ):
            if isinstance(event, TTSAudioChunkEvent):
                if not isinstance(event.data, bytes) or not event.data:
                    raise VoiceIntegrationError("tts-parse", "speech audio chunk was malformed")
                yield event.data
            elif isinstance(event, TTSCompletedEvent):
                completed = True
        if not completed:
            raise VoiceIntegrationError("tts-parse", "speech synthesis did not complete")


class VoiceConversationService:
    """Connect voice I/O to the existing isolated conversational agent."""

    def __init__(
        self,
        agent: ConversationalRAGAgent,
        stt: SpeechToText,
        tts: TextToSpeech,
        *,
        max_attempts: int = 2,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._agent = agent
        self._stt = stt
        self._tts = tts
        self._max_attempts = max_attempts

    def process_audio(
        self,
        session_id: str,
        audio_bytes: bytes,
        *,
        filename: str = "audio.wav",
        language_code: str = "en-IN",
    ) -> VoiceTurnResult:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        if not audio_bytes:
            raise VoiceIntegrationError("invalid-audio", "audio input is empty")
        transcription = self._with_retries(
            lambda: self._stt.transcribe_bytes(audio_bytes, filename, language_code),
            "stt",
        )
        transcript = _transcript_from_response(transcription)
        if not transcript:
            raise VoiceIntegrationError("empty-transcript", "speech transcription was empty")

        conversation = self._agent.process_turn(session_id, transcript)
        if not conversation.result.answer.strip():
            raise VoiceIntegrationError("empty-response", "agent response was empty")
        audio = self._with_retries(
            lambda: self._tts.synthesize(conversation.result.answer),
            "tts",
        )
        if not audio:
            raise VoiceIntegrationError("empty-audio", "speech synthesis returned no audio")
        return VoiceTurnResult(session_id, transcript, conversation, audio)

    def _with_retries(self, operation: Any, stage: str) -> Any:
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                return operation()
            except Exception as exc:
                if not _is_retryable_voice_error(exc) or attempt == self._max_attempts - 1:
                    raise VoiceIntegrationError(stage, f"{stage} service failed") from exc
                last_error = exc
                _retry_delay(attempt)
        message = f"{stage} service failed"
        if last_error is not None:
            raise VoiceIntegrationError(stage, message) from last_error
        raise VoiceIntegrationError(stage, message)

    async def process_audio_stream(
        self,
        session_id: str,
        audio_chunks: AsyncIterable[bytes],
        *,
        language_code: str = "en-IN",
    ) -> StreamingVoiceTurnResult:
        """Process one completed STT stream and one streamed TTS response.

        The agent is called exactly once after STT completion. Retrying an
        external stream can repeat provider-side work after an ambiguous
        network failure, so callers must not treat this as exactly-once
        provider processing.
        """
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        chunks: list[bytes] = []
        transcript = ""
        conversation: ConversationTurnResult | None = None
        async for event in self.iter_audio_stream(session_id, audio_chunks, language_code=language_code):
            if event.complete:
                transcript = event.transcript
                conversation = event.conversation
            else:
                chunks.append(event.audio)
                if not transcript:
                    transcript = event.transcript
                    conversation = event.conversation
        if conversation is None:
            raise VoiceIntegrationError("empty-response", "conversation result was missing")
        return StreamingVoiceTurnResult(
            session_id=session_id,
            transcript=transcript,
            conversation=conversation,
            audio_chunks=tuple(chunks),
            audio_complete=True,
        )

    async def iter_audio_stream(
        self,
        session_id: str,
        audio_chunks: AsyncIterable[bytes],
        *,
        language_code: str = "en-IN",
    ) -> AsyncIterator[VoiceAudioEvent]:
        """Yield TTS audio as it arrives, followed by one completion event.

        STT completes before the agent is called, so one invocation of this
        iterator produces exactly one agent turn. TTS retries are allowed
        only before the first chunk is yielded; retrying after partial
        delivery could duplicate audio for a playback client.
        """
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        transcript = await self._with_async_retries(
            lambda: self._stream_transcript(audio_chunks, language_code),
            "stt",
        )
        if not transcript:
            raise VoiceIntegrationError("empty-transcript", "speech transcription was empty")
        conversation = self._agent.process_turn(session_id, transcript)
        if not conversation.result.answer.strip():
            raise VoiceIntegrationError("empty-response", "agent response was empty")

        emitted = False
        async for chunk in self._iter_tts_chunks(conversation.result.answer, "tts"):
            emitted = True
            yield VoiceAudioEvent(
                session_id=session_id,
                audio=chunk,
                complete=False,
                transcript=transcript,
                conversation=conversation,
            )
        if not emitted:
            raise VoiceIntegrationError("empty-audio", "speech synthesis returned no audio")
        yield VoiceAudioEvent(
            session_id=session_id,
            audio=b"",
            complete=True,
            transcript=transcript,
            conversation=conversation,
        )

    async def _stream_transcript(
        self,
        audio_chunks: AsyncIterable[bytes],
        language_code: str,
    ) -> str:
        if not isinstance(self._stt, StreamingSpeechToText):
            raise VoiceIntegrationError("stt-config", "streaming STT adapter is not configured")
        return await self._stt.transcribe_stream(audio_chunks, language_code=language_code)

    def _tts_stream(self, text: str) -> AsyncIterator[bytes]:
        if not isinstance(self._tts, StreamingTextToSpeech):
            raise VoiceIntegrationError("tts-config", "streaming TTS adapter is not configured")
        return self._tts.synthesize_stream(text)

    async def _with_async_retries(
        self,
        operation: Callable[[], Awaitable[Any]],
        stage: str,
    ) -> Any:
        for attempt in range(self._max_attempts):
            try:
                return await operation()
            except Exception as exc:
                if not _is_retryable_voice_error(exc) or attempt == self._max_attempts - 1:
                    raise VoiceIntegrationError(stage, f"{stage} service failed") from exc
                await asyncio.sleep(_retry_delay_seconds(attempt))
        raise VoiceIntegrationError(stage, f"{stage} service failed")

    async def _iter_tts_chunks(self, text: str, stage: str) -> AsyncIterator[bytes]:
        for attempt in range(self._max_attempts):
            emitted = False
            stream = self._tts_stream(text)
            try:
                async for chunk in stream:
                    if not isinstance(chunk, bytes) or not chunk:
                        raise VoiceIntegrationError("tts-parse", "speech audio chunk was malformed")
                    emitted = True
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if emitted or not _is_retryable_voice_error(exc) or attempt == self._max_attempts - 1:
                    raise VoiceIntegrationError(stage, f"{stage} service failed") from exc
                await asyncio.sleep(_retry_delay_seconds(attempt))
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()


def _transcript_from_response(response: dict[str, Any]) -> str:
    if not isinstance(response, dict):
        raise VoiceIntegrationError("stt-parse", "speech transcription response was malformed")
    transcript = response.get("transcript")
    if not isinstance(transcript, str):
        raise VoiceIntegrationError("stt-parse", "speech transcription response had no transcript")
    return transcript.strip()


def _is_retryable_voice_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    try:
        from gnani.stt import APIError as STTAPIError, StreamConnectionError as STTConnectionError
        from gnani.tts import APIError as TTSAPIError, StreamConnectionError as TTSConnectionError
    except ImportError:
        return False
    if isinstance(exc, (STTConnectionError, TTSConnectionError)):
        return True
    if isinstance(exc, (STTAPIError, TTSAPIError)):
        return getattr(exc, "status_code", 0) in {408, 429} or getattr(exc, "status_code", 0) >= 500
    return False


def _retry_delay_seconds(attempt: int) -> float:
    return min(0.25 * (2**attempt), 2.0)


def _retry_delay(attempt: int) -> None:
    import time

    time.sleep(_retry_delay_seconds(attempt))


async def _collect_audio_chunks(chunks: AsyncIterator[bytes]) -> list[bytes]:
    collected: list[bytes] = []
    async for chunk in chunks:
        if not isinstance(chunk, bytes) or not chunk:
            raise VoiceIntegrationError("tts-parse", "speech audio chunk was malformed")
        collected.append(chunk)
    return collected
