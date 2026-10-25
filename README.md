# Nova

Nova is a Unity astronaut assistant for Meta Quest 3
The project is still in development

## Open the project

1. Use Unity 6000.3.9f1
2. Open the repo in Unity Hub
3. Open `Assets/Scenes/SampleScene.unity`
4. Press Play

For Quest Link connect the headset and use the Windows build profile
For standalone Quest use the Android profile and Build and Run
Standalone voice audio is not connected yet

## Jezero terrain

Terrain files are generated locally and stay out of Git

1. Get `M20_JezeroCrater_CTXDEM_20m.tif` from [USGS](https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic)
2. Put it in `Data/Source/`
3. Install the conversion tools

```sh
pip install numpy pillow
python Tools/convert_jezero.py
```

4. Run **Tools > Jezero > Import Terrain** in Unity

If terrain is missing add `Assets/JezeroTerrain/Generated/Jezero Terrain.prefab`
Use position `(-25494, 2567, -51220)` with zero rotation and unit scale
The player stays near the world origin to reduce tracking jitter
Terrain data comes from USGS Astrogeology and Mangold et al 2021

## HiRISE area

Get the [elevation model](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_dtm_mosaic) and [surface image](https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic) from USGS
Keep their filenames and put both in `Data/Source/`

```sh
python Tools/convert_hirise.py
```

Run **Tools > Jezero > Import HiRISE Test Area**
It updates the existing terrain prefab so do not add another copy
The patch covers about 557 by 633 meters around spawn
It uses 1 meter elevation data and tinted 25 centimeter grayscale imagery
The surrounding landscape keeps the CTX terrain
Run **Import Terrain** to return to CTX only

Use **Tools > Jezero > Build Surface Detail** to rebuild dust and gravel and rocks
Adjust **Jezero Rock Field** on **HiRISE Test Area** then rebuild
The default rock radius is 80 meters
Only boulders have colliders
Textures and rocks are procedural detail rather than mapped Martian features
Generated terrain assets stay in `Assets/JezeroTerrain/Generated/`

## Controls

- Left joystick to move
- Right joystick to snap turn
- WASD and mouse for desktop testing
- Left pinch to open the wrist computer
- Right pinch to open the hand map
- Telephone pose or V to toggle voice

## Shared readings

The HUD and wrist use the same PostgreSQL telemetry as NOVA
Keep `simulator.py` and the voice server running with `--datasource postgres`
Missing or stale readings show **TELEMETRY UNAVAILABLE**
Use the wrist oxygen buttons for demos
Refill sets oxygen to 100 percent and the simulator keeps consuming it

[Voice setup and oxygen demo](Assets/Scripts/Voice/README.md)
[Minimap](Assets/Scripts/Minimap/README.md)
[Navigation](Assets/Scripts/Navigation/README.md)
[Hand gestures](Assets/Scripts/Gestures/README.md)

## Testing

Check readability and tracking in Quest Link
Check terrain edges and boulder collision
Standalone performance still needs testing on Quest 3
The target is 72 FPS

For a local HUD demo turn off **Use Server Telemetry** on **Suit Telemetry**
Leave **Simulate** enabled and use F8 to cycle demo alerts
Prefabs are in `Assets/HelmetHUD`
