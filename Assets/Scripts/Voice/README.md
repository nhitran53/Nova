# Voice and shared readings

Python runs the voice conversation on the PC
Unity reads its status and telemetry through `VoiceSessionClient`
Tool selection stays in Python

Say **NOVA return to start** to return beside the damaged rover wheel
The reset fades briefly and stops auto navigation
Headset floor tracking needs to be ready
Add **thank you** to finish the request then end the conversation

## Live location

Play `SampleScene` and ask **NOVA where am I**
Location comes from a fresh Unity snapshot
Shuttle and rover distances use scene landmarks
Ground distance and route distance stay separate

For older terrain run **Tools > Jezero > Update Georeference** once
New imports already include it
Check the conversion with **Tools > NOVA > Check Spatial Mapping**
CTX and HiRISE share the same coordinate frame
Coordinates use the source Mars sphere and east longitude from 0 to 360
Missing georeferencing returns unavailable

Use **Log Current Location** on `NovaSpatialContextProvider` in Play mode to inspect a snapshot
Test walking and turning and shuttle navigation on Quest

## Start from the repo folder

Keep PostgreSQL running for shared readings
Ollama isn't needed
Put credentials in the root `.env` and keep that file out of Git
Voice needs `sounddevice` and an ElevenLabs key

Run the simulator in one terminal

```cmd
.\router\.venv\Scripts\python.exe simulator.py
```

Run the voice server in another terminal

```powershell
.\router\.venv\Scripts\python.exe -m router.voice_server --model "$HOME\models\Nova\router\ModernBERT-v4.2" --datasource postgres
```

Change the model path if your trained checkpoint is somewhere else
Wait for `router_ready` at `http://127.0.0.1:8765/voice/status` before starting voice
Without `--datasource postgres` the server uses its separate built-in simulation
Server `--device` selects the torch device rather than the mic
`--timeout` sets how many seconds of silence end a conversation
Restart both processes after code updates

## Wake word and Gemini

Start the optional [wake listener](../../../wakeword/README.md#windows-setup) after the voice server is ready
It listens for Hey Nova and releases the mic while the server handles the conversation
Use the same default Windows mic for both processes
Unity notices new sessions through its existing status polling
Lower **Idle Check Seconds** on **Voice Session Client** if that delay is too long

Gemini generates replies with an offline fallback
Add `--no-llm` to the server for offline replies only
More in the [router setup](../../../router/README.md#gemini)

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
Run Python tests in `router/tests`
Unity parsing and gesture tests are in **Test Runner > EditMode**
