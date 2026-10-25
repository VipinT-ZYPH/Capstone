import unittest

from voiceops.chatbot import ChatResult
from voiceops.conversation import ConversationalRAGAgent
from voiceops.voice import (
    GnaniSTTAdapter,
    GnaniSTTStreamingAdapter,
    GnaniTTSAdapter,
    GnaniTTSStreamingAdapter,
    VoiceConversationService,
    VoiceIntegrationError,
)


class FakeRAG:
    def answer(self, question):
        return ChatResult("Policy response.", "answer", "{}"), []


class FakeSTT:
    def __init__(self, response=None, failures=0):
        self.response = response or {"transcript": "What is the return policy?"}
        self.failures = failures
        self.calls = 0

    def transcribe_bytes(self, audio_bytes, filename, language_code):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("temporary STT failure")
        return self.response


class FakeTTS:
    def __init__(self, failures=0):
        self.failures = failures
        self.calls = 0

    def synthesize(self, text):
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("temporary TTS failure")
        return b"wav"


class FakeStreamingSTT:
    def __init__(self, transcripts):
        self.transcripts = iter(transcripts)
        self.calls = 0

    async def transcribe_stream(self, audio_chunks, *, language_code="en-IN"):
        self.calls += 1
        received = [chunk async for chunk in audio_chunks]
        if not received:
            raise ValueError("no audio chunks")
        return next(self.transcripts)


class FakeStreamingTTS:
    def __init__(self, chunks=(b"chunk-1", b"chunk-2")):
        self.chunks = chunks
        self.calls = 0

    async def synthesize_stream(self, text):
        self.calls += 1
        for chunk in self.chunks:
            yield chunk


class FailingStreamingTTS:
    def __init__(self, *, fail_after_chunk=False):
        self.calls = 0
        self.closed = 0
        self.fail_after_chunk = fail_after_chunk

    async def synthesize_stream(self, text):
        self.calls += 1
        try:
            yield b"first"
            if self.fail_after_chunk:
                raise TimeoutError("stream interrupted")
            yield b"second"
        finally:
            self.closed += 1


async def audio_chunks(*chunks):
    for chunk in chunks:
        yield chunk


async def _collect(stream):
    return [chunk async for chunk in stream]


class VoiceTests(unittest.TestCase):
    def agent(self):
        return ConversationalRAGAgent(
            lambda: FakeRAG(),
            lookup_order=lambda order_id, verification_value=None: {"verification_required": True},
            create_ticket=lambda *args, **kwargs: {"created": True, "ticket_id": "TKT-1"},
        )

    def test_transcription_agent_response_and_tts_flow(self):
        stt, tts = FakeSTT(), FakeTTS()
        result = VoiceConversationService(self.agent(), stt, tts).process_audio("A", b"audio")
        self.assertEqual(result.transcript, "What is the return policy?")
        self.assertEqual(result.audio, b"wav")
        self.assertEqual(stt.calls, 1)
        self.assertEqual(tts.calls, 1)

    def test_empty_audio_and_empty_transcript_are_rejected(self):
        service = VoiceConversationService(self.agent(), FakeSTT({"transcript": " "}), FakeTTS())
        with self.assertRaisesRegex(VoiceIntegrationError, "audio input"):
            service.process_audio("A", b"")
        with self.assertRaisesRegex(VoiceIntegrationError, "transcription was empty"):
            service.process_audio("A", b"audio")

    def test_stt_and_tts_retries_do_not_repeat_agent_or_tools(self):
        stt, tts = FakeSTT(failures=1), FakeTTS(failures=1)
        service = VoiceConversationService(self.agent(), stt, tts)
        result = service.process_audio("A", b"audio")
        self.assertEqual(stt.calls, 2)
        self.assertEqual(tts.calls, 2)
        self.assertEqual(result.conversation.result.answer, "Policy response.")

    def test_tts_failure_does_not_hide_valid_agent_result(self):
        tts = FakeTTS(failures=2)
        with self.assertRaisesRegex(VoiceIntegrationError, "tts service"):
            VoiceConversationService(self.agent(), FakeSTT(), tts).process_audio("A", b"audio")

    def test_empty_sdk_credentials_are_rejected_without_network_access(self):
        with self.assertRaisesRegex(ValueError, "api_key"):
            GnaniSTTAdapter("")
        with self.assertRaisesRegex(ValueError, "api_key"):
            GnaniTTSAdapter("")

    def test_voice_flow_does_not_add_sensitive_order_fields_to_response(self):
        result = VoiceConversationService(self.agent(), FakeSTT(), FakeTTS()).process_audio(
            "A",
            b"audio",
        )
        self.assertNotIn("phone", result.conversation.result.answer.casefold())
        self.assertNotIn("email", result.conversation.result.answer.casefold())

    def test_sessions_remain_isolated(self):
        agent = self.agent()
        service = VoiceConversationService(agent, FakeSTT(), FakeTTS())
        first = service.process_audio("A", b"audio")
        second = service.process_audio("B", b"audio")
        self.assertEqual(first.session_id, "A")
        self.assertEqual(second.session_id, "B")
        self.assertIsNone(agent.state("A").order_id)
        self.assertIsNone(agent.state("B").order_id)

    def test_handover_result_is_preserved(self):
        class EscalatingRAG:
            def answer(self, question):
                return ChatResult("Please connect me to a human.", "escalate", "{}"), []

        tickets = []
        agent = ConversationalRAGAgent(
            lambda: EscalatingRAG(),
            lookup_order=lambda order_id, verification_value=None: {"verification_required": True},
            create_ticket=lambda *args, **kwargs: tickets.append(args) or {
                "created": True,
                "ticket_id": "TKT-1",
            },
        )
        result = VoiceConversationService(agent, FakeSTT(), FakeTTS()).process_audio("A", b"audio")
        self.assertEqual(result.conversation.result.decision, "escalate")
        self.assertEqual(len(tickets), 1)

    def test_permanent_failure_is_not_retried(self):
        class PermanentSTT(FakeSTT):
            def transcribe_bytes(self, audio_bytes, filename, language_code):
                self.calls += 1
                raise ValueError("invalid audio")

        stt = PermanentSTT()
        with self.assertRaisesRegex(VoiceIntegrationError, "stt service"):
            VoiceConversationService(self.agent(), stt, FakeTTS(), max_attempts=3).process_audio(
                "A", b"audio"
            )
        self.assertEqual(stt.calls, 1)

    def test_streaming_flow_returns_chunks_and_complete_signal(self):
        service = VoiceConversationService(
            self.agent(),
            FakeStreamingSTT(["What is the return policy?"]),
            FakeStreamingTTS(),
        )
        result = __import__("asyncio").run(
            service.process_audio_stream("A", audio_chunks(b"pcm-1", b"pcm-2"))
        )
        self.assertEqual(result.transcript, "What is the return policy?")
        self.assertEqual(result.audio_chunks, (b"chunk-1", b"chunk-2"))
        self.assertTrue(result.audio_complete)

    def test_incremental_stream_yields_chunks_then_completion(self):
        service = VoiceConversationService(
            self.agent(),
            FakeStreamingSTT(["What is the return policy?"]),
            FakeStreamingTTS((b"one", b"two")),
        )

        async def consume():
            return [
                (event.audio, event.complete)
                async for event in service.iter_audio_stream("A", audio_chunks(b"pcm"))
            ]

        events = __import__("asyncio").run(consume())
        self.assertEqual(events, [(b"one", False), (b"two", False), (b"", True)])

    def test_stream_failure_after_delivery_is_not_reported_complete_or_retried(self):
        tts = FailingStreamingTTS(fail_after_chunk=True)
        service = VoiceConversationService(
            self.agent(),
            FakeStreamingSTT(["What is the return policy?"]),
            tts,
            max_attempts=3,
        )

        async def consume():
            events = []
            stream = service.iter_audio_stream("A", audio_chunks(b"pcm"))
            try:
                async for event in stream:
                    events.append(event)
            except VoiceIntegrationError:
                return events
            self.fail("stream failure was not propagated")

        events = __import__("asyncio").run(consume())
        self.assertEqual([(event.audio, event.complete) for event in events], [(b"first", False)])
        self.assertEqual(tts.calls, 1)

    def test_early_consumer_termination_closes_tts_stream(self):
        tts = FailingStreamingTTS()
        service = VoiceConversationService(
            self.agent(),
            FakeStreamingSTT(["What is the return policy?"]),
            tts,
        )

        async def consume_one():
            stream = service.iter_audio_stream("A", audio_chunks(b"pcm"))
            first = await anext(stream)
            await stream.aclose()
            return first

        first = __import__("asyncio").run(consume_one())
        self.assertEqual(first.audio, b"first")
        self.assertFalse(first.complete)
        self.assertEqual(tts.closed, 1)

    def test_streaming_empty_transcript_is_rejected(self):
        service = VoiceConversationService(
            self.agent(),
            FakeStreamingSTT([""]),
            FakeStreamingTTS(),
        )
        with self.assertRaisesRegex(VoiceIntegrationError, "transcription was empty"):
            __import__("asyncio").run(
                service.process_audio_stream("A", audio_chunks(b"pcm"))
            )

    def test_streaming_sessions_retain_order_verification_context(self):
        lookups = []

        def lookup(order_id, verification_value=None):
            lookups.append((order_id, verification_value))
            if verification_value is None:
                return {"verification_required": True}
            return {"found": True, "verified": verification_value == "1234", "order_id": order_id}

        agent = ConversationalRAGAgent(
            lambda: FakeRAG(),
            lookup_order=lookup,
            create_ticket=lambda *args, **kwargs: {"created": True, "ticket_id": "TKT-1"},
        )
        stt = FakeStreamingSTT(["Where is order 12345678?", "1234"])
        service = VoiceConversationService(agent, stt, FakeStreamingTTS())
        first = __import__("asyncio").run(
            service.process_audio_stream("A", audio_chunks(b"pcm"))
        )
        second = __import__("asyncio").run(
            service.process_audio_stream("A", audio_chunks(b"pcm"))
        )
        self.assertEqual(first.conversation.result.decision, "clarify")
        self.assertEqual(second.conversation.conversation_id, "A")
        self.assertEqual(lookups, [("12345678", None), ("12345678", "1234")])
        self.assertTrue(agent.state("A").verified)

    def test_repeated_handover_does_not_create_duplicate_tickets(self):
        tickets = []

        class EscalatingRAG:
            def answer(self, question):
                return ChatResult("A human agent will help.", "escalate", "{}"), []

        agent = ConversationalRAGAgent(
            lambda: EscalatingRAG(),
            lookup_order=lambda order_id, verification_value=None: {
                "found": True,
                "verified": True,
                "order_id": order_id,
            },
            create_ticket=lambda *args, **kwargs: tickets.append(args)
            or {"created": True, "ticket_id": "TKT-1"},
        )
        stt = FakeStreamingSTT(["My order is 12345678.", "1234", "Please contact human support.", "Please contact human support."])
        service = VoiceConversationService(agent, stt, FakeStreamingTTS())
        __import__("asyncio").run(service.process_audio_stream("A", audio_chunks(b"pcm")))
        __import__("asyncio").run(service.process_audio_stream("A", audio_chunks(b"pcm")))
        __import__("asyncio").run(service.process_audio_stream("A", audio_chunks(b"pcm")))
        self.assertEqual(len(tickets), 1)

    def test_stt_stream_adapter_joins_events_and_closes(self):
        from gnani.stt import StreamTranscriptEvent

        class Client:
            def __init__(self):
                self.closed = False

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                self.closed = True

            async def stream_audio(self, source, **kwargs):
                self.received = [chunk async for chunk in source]
                return [
                    StreamTranscriptEvent("hello", 100, "s1", "0", 1, "now", {}),
                    StreamTranscriptEvent("world", 100, "s2", "1", 1, "now", {}),
                ]

        adapter = GnaniSTTStreamingAdapter.__new__(GnaniSTTStreamingAdapter)
        adapter._client = Client()
        adapter._client.language_code = "en-IN"
        transcript = __import__("asyncio").run(
            adapter.transcribe_stream(audio_chunks(b"a", b"b"))
        )
        self.assertEqual(transcript, "hello world")
        self.assertEqual(adapter._client.received, [b"a", b"b"])
        self.assertTrue(adapter._client.closed)

    def test_tts_stream_adapter_yields_chunks_only_after_typed_events(self):
        from gnani.tts import TTSAudioChunkEvent, TTSCompletedEvent

        class Client:
            async def synthesize_events(self, text, **kwargs):
                yield object()
                yield TTSAudioChunkEvent(b"a", 1)
                yield TTSAudioChunkEvent(b"b", 2, True)
                yield TTSCompletedEvent("request", 2)

        adapter = GnaniTTSStreamingAdapter.__new__(GnaniTTSStreamingAdapter)
        adapter._client = Client()
        adapter._voice = "Pranav"
        adapter._model = "timbre-v2.0"
        adapter._audio_config = object()
        chunks = __import__("asyncio").run(_collect(adapter.synthesize_stream("hello")))
        self.assertEqual(chunks, [b"a", b"b"])


if __name__ == "__main__":
    unittest.main()
