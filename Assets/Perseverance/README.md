# Perseverance rover

Open SampleScene and select Damaged Perseverance
Toggle Damaged on Rover Damage State to show the repaired wheel
The impact rock sits beside the damaged wheel

Edit Data/Source/Perseverance/Perseverance Damaged.blend in Blender
Export the meshes and pose markers to Assets/Perseverance/Generated/Perseverance.fbx
Use scale 1 with forward -Z and up Y
Keep the textures beside the FBX

The Blender file stays local and the NASA download stays unchanged
The source is missing corrugated_normal.jpg so that normal map stays empty
Quest appearance and performance still need headset testing

## Inspection spawn

Quest starts beside the front-right wheel once floor tracking is ready
Rover Inspection Spawn on Damaged Perseverance controls the placement
Set Distance From Wheel and Fore Aft Offset before Play
Use Inspection Anchor for a custom ground-level spot
Ground Clearance keeps the rig above terrain
Inspection Near Clip lets you look closer

Say NOVA return to start or use Reset To Rover Inspection in Play mode
The reset fades briefly and keeps headset tracking
Check walking and crouching and reset clearance on Quest
