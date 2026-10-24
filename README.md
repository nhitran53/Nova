# Nova

Nova is a VR astronaut assistant built in Unity for Meta Quest 3. The project is still in development.

## Getting Started

1. Install **Unity 6000.3.9f1**.
2. Clone the repository and open it in Unity Hub.
3. Open `Assets/Scenes/SampleScene.unity`.
4. Press Play.

## Mars Terrain Setup

The Jezero Crater terrain is generated locally and is not included in the repository.

1. Download `M20_JezeroCrater_CTXDEM_20m.tif` from [USGS](https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic).
2. Place it in `Data/Source/`.
3. Install the required Python packages:

   ```bash
   pip install numpy pillow
   ```

4. Run:

   ```bash
   python Tools/convert_jezero.py
   ```

5. In Unity, select **Tools > Jezero > Import Terrain**.

If the terrain is missing, add `Assets/JezeroTerrain/Generated/Jezero Terrain.prefab` to the scene at position `(0, 0, 0)`, rotation `(0, 0, 0)`, and scale `(1, 1, 1)`.

Terrain data: USGS Astrogeology (Mangold et al., 2021).

## Running in VR

**Quest Link (PC):**
1. Connect your Quest 3 and start Meta Quest Link.
2. Set Meta Horizon Link as the active OpenXR runtime.
3. Select Windows in Unity's Build Profiles.
4. Press Play.

**Standalone Quest 3:** Switch to Android in Build Profiles and select Build and Run.

**Controls:**
- Left joystick: Move
- Right joystick: Snap turn
- Mouse/WASD: Desktop testing

## Helmet HUD

The HUD follows your view and displays oxygen, a compass, an AI placeholder, and critical alerts.

Select **Helmet HUD** in Unity to adjust its position and size.

Press **F8** during Play Mode to test different warning levels. All sensor readings are currently simulated.

## Wrist Computer

The left-wrist computer displays additional suit readings.

**Pinch your right thumb and index finger** to open or close it.

Hand tracking must be enabled. You can adjust the wrist computer's position and size in Unity.

## Development Notes

Reusable prefabs are located in `Assets/HelmetHUD`.

- `SuitTelemetry.SetReadings` — Updates suit telemetry on Unity's main thread.
- `HelmetHUD.SetAssistant` — Updates assistant text.

The AI assistant is not connected yet. Environmental awareness and survival features are planned for future development.