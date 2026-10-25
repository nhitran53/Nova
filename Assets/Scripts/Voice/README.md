# Voice and shared readings

Python runs the voice conversation on the PC
Unity reads its status and telemetry through `VoiceSessionClient`
Tool selection stays in Python

## Start from the repo folder

Keep PostgreSQL and Ollama running
Put credentials in the root `.env` and keep that file out of Git
Voice needs `sounddevice` and an ElevenLabs key

Run the simulator in one terminal

```cmd
.\router\.venv\Scripts\python.exe simulator.py
```

Run the voice server in another terminal

```sh
cd /home/isaach/Nova-wakeword && /home/isaach/Nova/router/.venv/bin/python -m router.voice_server
```

The default model on this machine is `/home/isaach/models/Nova/router/v1`; `--model` is optional.
Append `--datasource postgres` to enable the shared HUD and wrist telemetry described below.
Server `--device` selects the torch device, not the mic; `--timeout` is seconds of silence that end a conversation.
Restart both processes after code updates

## In Unity

Open `SampleScene` and press Play
Hold the right-hand telephone pose to start voice
Repeat it to stop or press V while testing on desktop
The NOVA orb shows the server state and latest reply
**Show Gesture Diagnostics** adds pose feedback while tuning

The HUD and wrist read the same active mission as NOVA
Keep **Use Server Telemetry** enabled on **Suit Telemetry**
The existing voice client is found automatically
Readings update every half second even without a conversation
Stale or missing data shows **TELEMETRY UNAVAILABLE**
Radiation uses uSv/h and temperature is suit temperature
Database credentials stay on the PC

## Oxygen demo

Open the left wrist computer and poke a button with your right index finger
Use **-10 O2** or **+10 O2** to change oxygen
Use **REFILL** for 100 percent
Pull back before pressing again
The panel has no collision and withdrawal does not count as another click

All three systems read the changed value
The simulator keeps consuming oxygen
Use more than 30 percent for normal conversation or 10 percent for a critical oxygen demo
Turn off **Show Oxygen Demo** on **Wrist Computer** to hide the controls

For an offline HUD demo turn off **Use Server Telemetry** and leave **Simulate** enabled

## Quest Link and standalone

Editor and Quest Link can use `http://127.0.0.1:8765`
Choose the headset microphone and speakers in Windows for Link audio

Standalone telemetry needs the PC LAN address in **Base URL**
Start the server with `--host <PC LAN IP>` on a trusted network
Development builds also need HTTP downloads allowed
Standalone voice audio is not connected yet
Keep the server local and do not expose it to the internet

## Limits

The microphone stays closed while NOVA speaks
Stopping a session lets the current spoken reply finish
Voice does not drive Unity navigation actions yet
Run Python tests in `router/tests`
Unity parsing and gesture tests are in **Test Runner > EditMode**
