# Telephone gesture

Pose: index, middle and ring fingers curled, thumb and pinky extended. Hold it about 0.35 s to start; it ends 0.15 s after the pose is lost, and ends at once if a fist, walking, a system gesture or a minimap touch takes the hand.

- `Logic/TelephoneGestureLogic.cs` is pure C# (`Nova.Gestures.Logic` asmdef, no engine references). `Tick(dt, sample)` returns `Started`, `Ended` or `None`.
- `TelephoneGestureDetector.cs` is the MonoBehaviour. It reads XR Hands joints, computes each fingertip's distance to the palm as a fraction of hand size, and exposes `OnTelephoneStart`, `OnTelephoneEnd`, `Active`, `Diagnostic`, and the static `TelephoneGestureDetector.AnyActive` / `AnyActiveChanged`.
- Tests: `Assets/Tests/EditMode/TelephoneGestureLogicTests.cs` (Test Runner > EditMode). The test asmdef is Editor-only and constrained to `UNITY_INCLUDE_TESTS`, so it is not in the player build.

## Adding it to the main scene (not done yet)

`Assets/Scenes/SampleScene.unity` has not been edited. To add it:

1. Open `Assets/Scenes/SampleScene.unity`. Select the XR Origin object that has `FistLocomotion` and `PinchToggle` (or an empty child of it).
2. Add Component > `TelephoneGestureDetector`. Set Hand (default Right; use Left if the right hand is used for fist walking and you want the free hand).
3. In On Telephone Start, click +, drag in the object that has the voice client, and pick its `StartVoiceSession()` method (no arguments). Optionally wire On Telephone End to a stop/end method if the client has one.
4. Save the scene.

Scripts can also react without the inspector: `TelephoneGestureDetector.AnyActiveChanged += active => ...;`.

## Tuning (not verified on a headset)

Thresholds are starting values. Check the `Diagnostic` string (`RIGHT / HOLDING / 0`; it also reports `FIST`, `WALKING`, `MAP TOUCH`, `SYSTEM GESTURE`, `NOT TRACKED`) on a Quest and adjust Curl Enter/Exit and Extend Enter/Exit if the pose is hard to hit or a relaxed hand triggers it.
