# Minimap

The map sits at the bottom left of the helmet
Right pinch moves it above your hand
Release and pinch again to return it to the helmet
Lost hand tracking also returns it to the helmet
Left pinch controls the wrist computer

## Setup

`SampleScene` already has the map
For another scene run **Tools > Nova > Minimap > Add Minimap To Scene**
Import the Jezero terrain for a relief map
Without terrain the map shows a grid
Use **Create Demo Markers Around Player** to add test locations
Move those markers onto real features before exporting

## Controls

- M to show or hide
- Brackets to zoom
- N to switch between heading up and north up
- Right pinch to open the hand map

Keep your palm angled away from the headset when pinching
Quest uses palm-facing pinches for its own menu

## What it shows

The center arrow is the player and the red N marks north
Dots show scene markers and faraway locations stay at the rim
The lower line shows map size and the nearest oxygen refill
The cyan line is the navigation route
Poke a destination pin to select it

Adjust **HUD Slot** and **HUD Diameter** on **Nova Minimap**
Adjust **Palm Offset** and **Rotation Offset** and **Diameter Meters** on **Holographic Minimap**

## Coordinates

Unity X maps to `x_m` and Z maps to `z_m`
One unit is one meter
Use **Tools > Nova > Minimap > Copy map_locations SQL** after moving markers
Export again if the terrain origin changes
The SQL seed coordinates are placeholders

`MinimapApiLink` is optional and stays off until its API address is set
It reads extra pins from `GET /map` and sends positions to `POST /position`

[Navigation controls](../Navigation/README.md)
Test both pinches and movement and readability in Quest Link and on the headset
