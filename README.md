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

If the terrain is missing, add `Assets/JezeroTerrain/Generated/Jezero Terrain.prefab` to the scene at position `(-25494, 2567, -51220)`, rotation `(0, 0, 0)`, and scale `(1, 1, 1)`.

Terrain data: USGS Astrogeology (Mangold et al., 2021).

### HiRISE Test Area

This adds a **557 × 633 m high-resolution terrain area** around the player's starting location. The rest of Jezero Crater uses the original CTX terrain.

1. Download the following files from USGS:
   - [HiRISE Elevation Model](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_dtm_mosaic)
   - [HiRISE Surface Imagery](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic)

2. Keep their original filenames and place both files in `Data/Source/`.

3. After completing the main CTX terrain setup, run:

   ```bash
   python Tools/convert_hirise.py
   ```

4. In Unity, select **Tools > Jezero > Import HiRISE Test Area**.

The importer automatically adds the detailed terrain to the existing Jezero prefab. Open `SampleScene.unity` to see it. **Do not add another terrain instance.**

The HiRISE area uses 1 m elevation data and 25 cm grayscale imagery, tinted so its average matches the CTX ground color. It's an approximation, not a true-color photo. Its edges blend into the surrounding CTX terrain to create a smoother transition.

All generated files are stored in `Assets/JezeroTerrain/Generated/HiRISE/` and are ignored by Git. Each teammate will need to generate the terrain locally.

**Notes:**
- The terrain is positioned so the player starts near Unity's origin, helping reduce precision-related jitter.
- Both desktop and VR player rigs start inside the detailed area.
- Test the terrain's appearance, collision, and performance in Quest Link and on standalone Quest 3.
- To restore the original CTX-only terrain, select **Tools > Jezero > Import Terrain**.

### Ground Detail and Rocks

Use **Tools > Jezero > Build Surface Detail** to generate or update the ground textures and rocks. This also runs automatically when importing HiRISE terrain.

The system adds dust, gravel, and realistic-looking rocks while preserving the original satellite imagery.

To customize the rocks, select **HiRISE Test Area** and find **Jezero Rock Field** in the Inspector.

- **Radius:** Size of the rocky area.
- **Seed:** Changes the rock layout.
- **Rock Counts:** Controls the number of rocks.
- **Draw Distance:** Controls how far rocks are visible.
- **LOD Distance:** Controls when simpler rock models appear.
- **Max Slope:** Prevents rocks from appearing on steep terrain.

By default, rocks cover an 80 m radius around the starting area. Only large boulders have collision.

After changing settings, run **Build Surface Detail** again.

### Testing

**Quest Link:** Open `SampleScene` using the Windows profile and press Play.

**Standalone Quest 3:** Switch to the Android profile and select Build and Run.

Check ground textures, rock placement, and performance. The target is a stable 72 FPS on Quest 3.

Generated assets are stored in `Assets/JezeroTerrain/Generated/SurfaceDetail/` and are ignored by Git.

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

**Pinch your left thumb and index finger** to open or close it. Release before pinching again.

Hand tracking must be enabled. You can adjust the wrist computer's position and size in Unity.
If it feels too floaty or too shaky, tweak the **Smoothing** settings on Wrist Computer.

## Development Notes

Reusable prefabs are located in `Assets/HelmetHUD`.

- `SuitTelemetry.SetReadings` — Updates suit telemetry on Unity's main thread.
- `HelmetHUD.SetAssistant` — Updates assistant text.

The AI assistant is not connected yet. Environmental awareness and survival features are planned for future development.
