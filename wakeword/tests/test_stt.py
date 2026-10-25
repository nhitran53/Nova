import asyncio
import logging

import pytest

from conftest import silence, speech
from stt import RedactingFilter, SilenceEndpointer, SttError, Transcriber, redact

KEY = "sk_test_SECRET_0123456789"


def feed_all(endpointer, frames):
    """Index of the frame at which the endpointer fires, or None."""
    for i, frame in enumerate(frames):
        if endpointer.feed(frame):
            return i
    return None


def test_endpoints_after_1_5s_silence_following_speech():
    frames = speech(10) + silence(30)  # 80 ms frames: 1.5 s == 18.75 -> fires on the 19th silent frame
    assert feed_all(SilenceEndpointer(), frames) == 10 + 18


def test_brief_pause_does_not_endpoint():
    frames = speech(5) + silence(10) + speech(5) + silence(30)  # 0.8 s pause < 1.5 s
    assert feed_all(SilenceEndpointer(), frames) == 5 + 10 + 5 + 18


def test_no_speech_times_out_without_heard_speech():
    ep = SilenceEndpointer(no_speech_timeout_s=1.0)
    assert feed_all(ep, silence(40)) == 12  # 1.0 s / 0.08 = 12.5
    assert not ep.heard_speech


def test_max_duration_caps_continuous_speech():
    ep = SilenceEndpointer(max_s=2.0)
    assert feed_all(ep, speech(100)) == 24
    assert ep.done


class FakeConnection:
    def __init__(self, text="turn on the lights", fail_on=None):
        self.handlers = {}
        self.sent = 0
        self.committed = False
        self.closed = False
        self.text = text
        self.fail_on = fail_on

    def on(self, event, callback):
        self.handlers[str(getattr(event, "value", event))] = callback

    async def send(self, data):
        self.sent += 1
        if self.fail_on == "send":
            raise RuntimeError(f"boom with key {KEY}")

    async def commit(self):
        self.committed = True
        asyncio.get_running_loop().call_soon(self.handlers["committed_transcript"], {"text": f" {self.text} "})

    async def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, connection=None, connect_error=None, batch_text="batch text", batch_error=None):
        self.connection = connection
        self.batch_calls = []
        outer = self

        class Realtime:
            async def connect(self, options):
                outer.options = options
                if connect_error:
                    raise connect_error
                return outer.connection

        class SpeechToText:
            realtime = Realtime()

            def convert(self, **kwargs):
                outer.batch_calls.append(kwargs)
                if batch_error:
                    raise batch_error
                return type("R", (), {"text": batch_text})()

        self.speech_to_text = SpeechToText()


def transcriber(client, mode="auto", **kwargs):
    return Transcriber(KEY, mode=mode, client_factory=lambda key: client,
                       endpointer_factory=lambda: SilenceEndpointer(no_speech_timeout_s=1.0), **kwargs)


def test_realtime_streams_until_silence_then_commits():
    connection = FakeConnection()
    frames = iter(speech(5) + silence(40))
    text = transcriber(FakeClient(connection)).transcribe(frames)
    assert text == "turn on the lights"
    assert connection.committed and connection.closed
    assert connection.sent == 5 + 19  # stopped sending at the endpoint
    assert next(frames) is not None  # remaining audio is left for the listener


def test_realtime_no_speech_returns_empty_without_commit():
    connection = FakeConnection()
    assert transcriber(FakeClient(connection)).transcribe(iter(silence(40))) == ""
    assert not connection.committed and connection.closed


def test_auto_falls_back_to_batch_and_keeps_audio_already_sent(caplog):
    connection = FakeConnection(fail_on="send")
    client = FakeClient(connection)
    with caplog.at_level(logging.WARNING):
        text = transcriber(client).transcribe(iter(speech(5) + silence(40)))
    assert text == "batch text"
    (call,) = client.batch_calls
    assert call["model_id"] == "scribe_v2"
    name, wav, mime = call["file"]
    assert mime == "audio/wav" and wav[:4] == b"RIFF"
    assert len(wav) == 44 + (5 + 19) * 1280 * 2  # frame 1 was recorded before the failed send; the rest is read in batch mode
    assert "falling back to batch" in caplog.text


def test_batch_mode_skips_realtime():
    client = FakeClient(FakeConnection())
    assert transcriber(client, mode="batch").transcribe(iter(speech(5) + silence(40))) == "batch text"
    assert not hasattr(client, "options")


def test_realtime_mode_does_not_fall_back():
    client = FakeClient(connect_error=OSError("down"))
    with pytest.raises(SttError):
        transcriber(client, mode="realtime").transcribe(iter(speech(5)))
    assert client.batch_calls == []


def test_batch_without_speech_makes_no_request():
    client = FakeClient()
    assert transcriber(client, mode="batch").transcribe(iter(silence(40))) == ""
    assert client.batch_calls == []


def test_api_key_never_appears_in_errors_logs_or_output(caplog, capsys):
    connection = FakeConnection(fail_on="send")
    client = FakeClient(connection, batch_error=RuntimeError(f"401 for key {KEY}"))
    handler = logging.StreamHandler()
    handler.addFilter(RedactingFilter(KEY))
    logging.getLogger("nova.wakeword").addHandler(handler)
    try:
        with caplog.at_level(logging.DEBUG, logger="nova.wakeword"):
            with pytest.raises(SttError) as excinfo:
                transcriber(client).transcribe(iter(speech(5) + silence(40)))
    finally:
        logging.getLogger("nova.wakeword").removeHandler(handler)
    captured = capsys.readouterr()
    assert KEY not in str(excinfo.value)
    assert KEY not in captured.out + captured.err
    assert KEY not in "".join(r.getMessage() for r in caplog.records)
    assert excinfo.value.__cause__ is None and excinfo.value.__suppress_context__


def test_client_factory_failure_does_not_leak_key():
    def factory(key):
        raise RuntimeError(f"bad key {key}")
    with pytest.raises(SttError) as excinfo:
        Transcriber(KEY, client_factory=factory).transcribe(iter([]))
    assert KEY not in str(excinfo.value)


def test_redact():
    assert redact(f"a {KEY} b", KEY) == "a [redacted] b"
    assert redact("plain", "") == "plain"
