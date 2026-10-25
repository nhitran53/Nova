# Hand gestures

The telephone pose curls the index and middle and ring fingers
Keep the thumb and pinky extended
Hold for about 0.35 seconds to start voice
Release for about 0.15 seconds to end the pose
The next pose toggles voice off

## Scene setup

`SampleScene` has **Telephone Gesture Detector** on **Quest XR Origin**
Its start event calls `VoiceSessionClient.ToggleVoiceSession` on **Nova Voice**
The end event is unused so the conversation can continue after you release
Pinches pause while the telephone pose is active
Walking and system gestures and map touches take priority

## Tuning

Enable **Show Gesture Diagnostics** on **Nova Assistant Presenter**
Adjust the curl and extend thresholds on the detector
Turn diagnostics off after tuning
Thresholds still need testing in the headset

`Logic/TelephoneGestureLogic.cs` handles pose timing
`Logic/FrontPoke.cs` handles one-way wrist button presses
Run their tests from **Test Runner > EditMode**
[Voice setup](../Voice/README.md)
