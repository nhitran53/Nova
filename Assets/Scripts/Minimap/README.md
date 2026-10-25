# Helmet minimap

The map sits in the bottom left of your helmet
Right pinch opens the hand map and left pinch controls the wrist computer

## Setup (once)

1. Make sure the Jezero terrain is imported (`Tools/convert_jezero.py`, then **Tools > Jezero > Import Terrain**).
   Without it the map still works but shows a dark disc with a grid instead of terrain.
2. `SampleScene` already has the minimap. For another scene use **Tools > Nova > Minimap > Add Minimap To Scene**.
3. Optional **Tools > Nova > Minimap > Create Demo Markers Around Player** adds demo markers near the player.
   Move them onto real features before exporting.
4. Save the scene and press Play.

Keys while testing on desktop: **M** show/hide, **[ ]** zoom, **N** heading-up / north-up.

## Using the hand map

Pinch your right thumb and index finger to open the map above your hand
Release and pinch again to bring it back to the helmet
The map keeps updating in both views
If hand tracking is lost it returns to the helmet

Select `Nova Minimap` to adjust **HUD Slot** and **HUD Diameter**
Use **Palm Offset** and **Rotation Offset** on **Holographic Minimap** to adjust the hand view
**Diameter Meters** changes its size

Keep your palm angled away from the headset when pinching
Quest uses palm facing pinches for its own menu

## Testing

For Quest Link select Windows and connect your headset
Open `SampleScene` and press Play
Check both pinches and make sure movement and the rest of the HUD still work

For standalone Quest 3 select Android and use Build and Run
Hand tracking and readability still need testing in the headset

## What is on the map

- Ground: shaded relief from the real terrain heights, contour lines, and a grid (spacing shown under the map).
- White arrow in the middle: you. With heading-up on, the map turns so forward is up; the red **N** shows north.
- Colored dots: `MapMarker` objects. Cyan habitat, green rover, yellow sample site, white landmark/crater,
  amber caution, red danger. Circles show `radius_m` zones. Far-away markers stick to the rim.
- Bottom line: view size, grid size, and the nearest oxygen refill with distance.

## Keeping the database in sync

`map_locations` seed coordinates are placeholders. After placing markers, use
**Tools > Nova > Minimap > Copy map_locations SQL** and give the result to whoever owns `schema.sql`.
Coordinates: Unity x = `x_m`, Unity z = `z_m`, 1 unit = 1 m.
The map and API use the current Unity scene coordinates
If your database has positions from before the terrain was shifted export them again with the SQL menu

## Files

| File | What it does |
|---|---|
| `MapMarker.cs` | Component for a place on the map (code, name, type, radius, hazard, O2 refill). |
| `MarsMinimap.cs` | Builds and updates the whole minimap at runtime. Lives on the `Nova Minimap` object. |
| `HolographicMinimap.cs` | Shows the same map above your right hand. |
| `MinimapApiLink.cs` | Optional. `GET /map` for extra pins, `POST /position` for zone alerts. Off until `apiBaseUrl` is set. |
| `../../Editor/NovaMinimapTools.cs` | The Tools > Nova > Minimap menu. |

## For teammates (voice / gestures / AI)

Call these on the `MarsMinimap` component: `Show(bool)`, `ZoomIn()`, `ZoomOut()`, `SetHeadingUp(bool)`,
`Highlight(seconds)` (pulses the rim, good for the AI's `minimap` focus command), `SetExternalPins(points)`.
Read `PlayerXZ` and `HeadingDeg` for the astronaut's position and compass heading.
