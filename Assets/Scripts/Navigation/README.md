# Navigation

A walking route from the spawn to a temporary shuttle (`shuttle_001`) in the HiRISE patch.
The route is worked out ahead of time in the editor and saved as an asset. Nothing searches for paths at runtime

Nothing starts on its own. The shuttle sits on the minimap as a pin. Pick it, press **FIND ROUTE** to preview the route
on the map, then **START NAVIGATION** to get the cyan trail on the ground and the readout in the top-right of the helmet
(distance, ETA, which way to turn, on/off route). **STOP NAVIGATION** hides the trail and readout but keeps the route and the pin

## Setup (once)

1. Open `SampleScene`.
2. **Tools > Nova > Navigation > Add Navigation To Scene**. It adds a `Nova Navigation` object, a `shuttle_001` map marker,
   the destination panel on `Nova Minimap`, and fist walking on the `Quest XR Origin`. Running it again only adds what's missing
3. Save the scene and press Play

## Moving the shuttle / rebuilding the route

Select `Routes/route_shuttle_001.asset`, change **Destination XZ** (scene x/z) or the search settings, then run
**Tools > Nova > Navigation > Rebuild Shuttle Route**. The scene must be open, since it reads the `HiRISE Test Area` terrain and its rocks
Move the `shuttle_001` marker to match (or run Add Navigation To Scene again, which moves it for you)

Route points are scene coordinates. If the terrain gets recentred again, rebuild the route

How the route is made:
- A* over the 513x513 HiRISE heightfield (about 1.1 x 1.2 m cells), 8 neighbours
- each step costs its 3D length x `1 + slopeWeight x (slope / referenceSlope)^2`, so steeper ground costs a lot more
- ground over `maxSlopeDeg`, the edge margin, and anything within boulder radius + `boulderClearance` of the generated boulders is off limits
- the grid zigzag is straightened with shortcuts that don't cross blocked ground or add more than `smoothingTolerance` cost
- the result is resampled every `pointSpacing` meters on the real terrain height

The boulders come from `JezeroRockField` (procedural, seed 2020), not real mapped rocks. Medium rocks and pebbles are ignored.
This is demo guidance, not a verified safe path

## Picking the shuttle on the map

States: Idle (pin only) -> Preview (route on the map, no trail) -> navigating (trail + HUD, then off route / arrived) -> Stopped (route kept)

Headset:
1. Right pinch opens the map over your right hand
2. Poke the shuttle pin with your left index fingertip. A panel shows above the map with the name and roughly how far it is
3. Poke the button: **FIND ROUTE**, then **START NAVIGATION**, then **STOP NAVIGATION** when you're done
4. Poke the pin again to close the panel

The poke only counts if your fingertip comes in from the front (about 2.5 cm in front of the map) and then touches it,
and not in the first half second after the map opens. Pull back before the next press.
While your fingertip is within about 6 cm of the map, both pinches and fist walking are paused, so pressing buttons
never closes the map, opens the wrist computer or starts walking. Tune it on `Minimap Navigation Panel`

Desktop: press Esc to free the cursor, then click the pin and the button (the panel sits to the right of the map).
Clicks on the map don't grab the cursor back

## Tool calls

The map button, the debug keys and the future NOVA orchestrator all go through the same three methods on `RouteNavigator`.
`NavigationService` only looks up precomputed routes

| Call | Method | Does |
|---|---|---|
| `find_navigation_route { destination_id, preference }` | `FindNavigationRoute(request)` | Looks the route up and previews it on the map. Doesn't start anything |
| `start_navigation { route_id }` | `StartNavigation(routeId)` | Shows the trail and HUD and starts tracking. Empty `route_id` = the route found last |
| `stop_navigation { }` | `StopNavigation()` | Hides the trail and HUD, keeps the route and the pin |

`RunToolJson(tool, argumentsJson)` does the same with JSON in and out:

```
RunToolJson("find_navigation_route", "{\"destination_id\":\"shuttle_001\",\"preference\":\"safest\"}")
-> { "success": true, "route_id": "route_shuttle_001", "destination_id": "shuttle_001", "preference": "low_slope",
     "distance_meters": 165.6, "estimated_time_seconds": 138, "route_type": "precomputed",
     "status": "found: no separate 'safest' route; returning the precomputed low_slope route" }
```

There is one route per destination right now (`low_slope`). Asking for another preference returns that route and says so
in `status`. An unknown destination or route id comes back with `success: false`

Desktop debug: **G** = find (the `Test Request` on `Route Navigator`), **H** = start, **J** = stop.
The same three are in the `Route Navigator` context menu under **Tools**. Everything is logged

## Fist walking (Quest)

Close your right hand into a fist and hold it for about 0.4 s. You walk forward where you're looking at 1.2 m/s and
`WALK ACTIVE` shows in the helmet. Open your hand (or lose tracking) and you stop right away

It pushes the existing Continuous Move Provider, so gravity, the CharacterController and the left stick work as before.
While your right fist is closed, the right pinch (hand map) is ignored. While walking, the left pinch (wrist computer) is ignored too.
Walking won't start in the middle of a left pinch, or while your fingertip is at the hand map

Thresholds are on `Fist Locomotion` (curl enter/exit, hold time) if it triggers too easily or not at all

## Files

| File | What it does |
|---|---|
| `NavigationRoute.cs` | Route asset: settings for the rebuild + the saved points and stats |
| `NavigationService.cs` | Tool request/response, looks up precomputed routes |
| `RouteNavigator.cs` | Tool calls, states, trail, progress (nearest segment, remaining, ETA, arrival, off route), debug keys |
| `../Minimap/MinimapNavigationPanel.cs` | Shuttle selection, destination panel and button, poke + mouse input |
| `NavigationHud.cs` | Helmet readout (screen corner on desktop) |
| `Routes/route_shuttle_001.asset` | The shuttle route |
| `../FistLocomotion.cs` | Right fist walking + `WALK ACTIVE` |
| `../../Editor/NovaNavigationTools.cs` | The Tools > Nova > Navigation menu and the route search |
