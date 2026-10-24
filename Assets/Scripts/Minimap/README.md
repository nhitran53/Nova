# Helmet minimap

Round 2D map of Mars pinned to the lower left of the helmet view.

## Setup (once)

1. Make sure the Jezero terrain is imported (`Tools/convert_jezero.py`, then **Tools > Jezero > Import Terrain**).
   Without it the map still works but shows a dark disc with a grid instead of terrain.
2. **Tools > Nova > Minimap > Add Minimap To Main Camera**.
3. **Tools > Nova > Minimap > Create Demo Markers Around Player** (the `schema.sql` seed places, moved to
   where Mars Explorer stands). Drag them onto real features in the Scene view.
4. Save the scene and press Play.

Keys while testing on desktop: **M** show/hide, **[ ]** zoom, **N** heading-up / north-up.

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

## Files

| File | What it does |
|---|---|
| `MapMarker.cs` | Component for a place on the map (code, name, type, radius, hazard, O2 refill). |
| `MarsMinimap.cs` | Builds and updates the whole minimap at runtime. Put it on the Main Camera. |
| `MinimapApiLink.cs` | Optional. `GET /map` for extra pins, `POST /position` for zone alerts. Off until `apiBaseUrl` is set. |
| `../../Editor/NovaMinimapTools.cs` | The Tools > Nova > Minimap menu. |

## For teammates (voice / gestures / AI)

Call these on the `MarsMinimap` component: `Show(bool)`, `ZoomIn()`, `ZoomOut()`, `SetHeadingUp(bool)`,
`Highlight(seconds)` (pulses the rim, good for the AI's `minimap` focus command), `SetExternalPins(points)`.
Read `PlayerXZ` and `HeadingDeg` for the astronaut's position and compass heading.
