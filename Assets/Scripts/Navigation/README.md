# Navigation

A saved walking route leads from spawn to the demo shuttle
The route is built in the Editor and does not search for paths at runtime
It is demo guidance rather than a verified safe path

## Setup

1. Open `SampleScene`
2. Run **Tools > Nova > Navigation > Add Navigation To Scene**
3. Save and press Play

Running setup again only adds missing components and updates the shuttle marker
It adds the shuttle model to the marker if needed

## Shuttle model

The demo shuttle uses a Viking lander model from Blender
The FBX is in `Assets/Shuttle/Generated`
It uses plain colors without textures
Check its performance on Quest

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

## Voice

Say "NOVA take me to the shuttle" to start demo auto-navigation
The existing BERT and tool pipeline starts the saved shuttle route
"Show me the shuttle on the map" marks its location without starting navigation
Distance questions and insufficient oxygen do not start guidance
The voice server puts the command in `/voice/status` and `RouteNavigator` picks it up
`NavigationService` returns the available route if a requested preference has no match

## Demo auto-navigation

Demo movement for the existing Quest XR Origin
It stays off until a voice request or `start_demo_navigation` call
Map Start and H still give manual guidance
This is not validated autonomous traversal

**Demo Auto Move Speed** defaults to 15 meters per second
**Demo Acceleration** sets the speed ramp and braking
Start near the saved trail and try a slower speed first
Fist walking pauses during auto-movement and returns afterward
Arrival or lost tracking or a blocked route stops movement
Cancel with J or STOP NAVIGATION on the map or the left controller stick

The Route Navigator component menu has **Demo > Auto navigate to shuttle** for Play mode testing
Check stopping and comfort and terrain following on Quest

## Shuttle waypoint

Say "NOVA show me the shuttle on the map"
The map highlights the shuttle with its distance
A cyan beacon and compass arrow mark the same scene location
The hand map opens when the right hand is tracked
Showing a location does not move the astronaut
The waypoint stays until cleared or replaced

Clear it through **Tools > Clear Waypoint** in the Route Navigator component menu
`RunToolJson` also accepts `show_waypoint` with `destination_id` and `clear_waypoint`
**Show Waypoint Beacon** hides the beam without clearing the target
**Waypoint Beacon Height** and **Waypoint Beacon Width** set its size
Check distant visibility and both eyes on Quest
