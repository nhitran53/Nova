# Voice conversation from Unity

`VoiceSessionClient` starts/stops the spoken conversation (`router/stt.py`: mic -> Scribe -> router -> TTS, repeated) that
`router/voice_server.py` runs on the PC. It only sends HTTP commands and polls status.

## Run the server (on the PC)

```
router/.venv/bin/python -m router.voice_server            # 127.0.0.1:8765; --host / --port / --model / --no-speak / --no-llm
```

It loads the router in the background at start (`router_ready` in the status), so `/voice/start` is quick after that.
The microphone needs `sounddevice` in that venv.

## Unity

Add `VoiceSessionClient` to any GameObject (no scene is edited by this folder). Call `StartVoiceSession()`,
`StopVoiceSession()` or `ToggleVoiceSession()` (e.g. from a button/gesture UnityEvent). Wire `OnStateChanged`,
`OnTranscript`, `OnReply`, `OnError` in the Inspector or via code; `State` holds the current state
(idle | loading | listening | speaking | ended | error). Base URL defaults to `http://127.0.0.1:8765`.
If the server is unreachable it logs one `[Voice]` warning and fires `OnError`; it never throws.

API: `POST /voice/start`, `POST /voice/stop`, `GET /voice/status`
-> `{state, last_transcript, last_reply, route, error, session_id, seq, router_ready}`.

## Limits

- **Python owns the microphone and speaker on the PC.** No audio goes to or from Unity. Quest standalone audio is out of scope;
  with a headset, run the server on the PC, use the PC's LAN address as base URL and start the server with `--host`
  (it has no authentication, so only on a trusted network).
- `ELEVENLABS_API_KEY` is read by Python from its environment (or `router/.env`); Unity never sees or sends it.
- Stop does not cut off a reply that is already being spoken; a new start is refused until that reply ends.
- Tests: `Assets/Tests/EditMode/VoiceSessionClientTests.cs` (status parsing / change detection; Editor only). Python side:
  `router/tests/test_voice_server.py`.
