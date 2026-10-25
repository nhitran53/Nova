# Navigation

A saved walking route leads from spawn to the demo shuttle
The route is built in the Editor and does not search for paths at runtime
It is demo guidance rather than a verified safe path

## Setup

1. Open `SampleScene`
2. Run **Tools > Nova > Navigation > Add Navigation To Scene**
3. Save and press Play

Running setup again only adds missing components and updates the shuttle marker

## Use the map

Open the right-hand map with a right pinch
Poke the shuttle pin with your left index finger
Press **FIND ROUTE** to preview it
Press **START NAVIGATION** to show the ground trail and helmet guidance
Press **STOP NAVIGATION** to hide guidance and keep the preview
Poke the pin again to close its panel
Pull back before each press
Nearby map touches pause pinch and fist input

On desktop press Esc then click the pin and buttons
G finds a route and H starts navigation and J stops it

## Rebuild a route

Select `Routes/route_shuttle_001.asset` and change **Destination XZ**
Keep the scene open and run **Tools > Nova > Navigation > Rebuild Shuttle Route**
Move the shuttle marker to match or run scene setup again
Rebuild after moving the terrain origin because route points use world coordinates

The search uses terrain heights and slope costs
It avoids steep ground and patch edges and generated boulders
Shortcuts stay within **Smoothing Tolerance**
**Point Spacing** controls the saved route points
Small rocks have no collision and do not block the route

## Fist walking

Hold a right fist for about 0.4 seconds to walk forward at 1.2 m/s
Open your hand or lose tracking to stop
The existing move provider still handles the left stick and gravity
Adjust **Curl Enter** and **Curl Exit** and **Hold Time** on **Fist Locomotion**

## Code hooks

`RouteNavigator.FindNavigationRoute` previews a saved route
`StartNavigation` starts it and `StopNavigation` hides guidance
An empty route ID starts the last preview
`RunToolJson` exposes the same actions to a later integration
Voice does not call these Unity methods yet
`NavigationService` returns the available route if a requested preference has no match
