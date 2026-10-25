using TMPro;
using Nova.Gestures;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Hands;

[DefaultExecutionOrder(-110)]
public class WristComputer : MonoBehaviour
{
    [SerializeField] SuitTelemetry telemetry;
    [SerializeField] RectTransform panel;
    [SerializeField] TMP_Text readings;
    [Tooltip("In wrist space: -Y is the palm side, -Z is toward the elbow.")]
    [SerializeField] Vector3 wristOffset = new Vector3(0, -0.045f, -0.075f);
    [Tooltip("Faces out of the palm, tilted 20 degrees toward the pinky.")]
    [SerializeField] Vector3 wristRotation = new Vector3(-70, 90, 180);
    [SerializeField, Min(0.0001f)] float panelScale = 0.00035f;
    [SerializeField] bool panelOpen = true;

    [Header("Oxygen demo")]
    [SerializeField] bool showOxygenDemo = true;
    [SerializeField, Min(0.005f)] float demoReleaseDistance = 0.025f;
    [SerializeField, Min(0)] float demoPressDistance = 0.008f;
    [SerializeField, Min(0.1f)] float demoDebounce = 0.35f;
    public static bool FingerNearControls { get; private set; }
    RectTransform demoRoot;
    Button[] demoButtons;
    TMP_Text demoLabel;
    Camera viewCamera;
    XRHandSubsystem demoHands;
    static readonly List<XRHandSubsystem> handSubsystems = new List<XRHandSubsystem>();
    readonly FrontPoke demoPoke = new FrontPoke();
    bool demoWasVisible;
    float demoOpenedAt;

    [Header("Smoothing")]
    [Tooltip("Lower = steadier when still, but laggier.")]
    [SerializeField, Min(0.01f)] float minCutoff = 1.5f;
    [Tooltip("Higher = less lag when moving fast.")]
    [SerializeField, Min(0)] float positionBeta = 8;
    [SerializeField, Min(0)] float rotationBeta = 0.04f;

    Transform trackingSpace;
    XRHandTrackingEvents leftHand;
    Transform wrist;
    float nextSearch;
    bool hasPose;
    float lastSample;
    Vector3 position, rawPosition, positionSpeed;
    Quaternion rotation, rawRotation;
    float rotationSpeed;

    public bool IsOpen => panelOpen;
    public bool WristTracked => leftHand != null && leftHand.handIsTracked;

    void OnEnable()
    {
        var origin = GetComponentInParent<XROrigin>();
        trackingSpace = origin.CameraFloorOffsetObject.transform;
        viewCamera = origin.Camera;
        panel.GetComponent<Canvas>().worldCamera = origin.Camera;
        // Smooth wrist tracking without smoothing walking or snap turns.
        panel.SetParent(trackingSpace, false);
        if (showOxygenDemo && demoRoot == null) BuildDemoControls();
        telemetry.Changed += Refresh;
        Refresh(telemetry.Current);
        panel.gameObject.SetActive(false);
        if (leftHand != null) leftHand.poseUpdated.AddListener(OnWristPose);
    }

    void OnDisable()
    {
        if (telemetry != null) telemetry.Changed -= Refresh;
        if (leftHand != null) leftHand.poseUpdated.RemoveListener(OnWristPose);
        if (panel != null) panel.gameObject.SetActive(false);
        hasPose = false;
        FingerNearControls = demoWasVisible = false;
        demoPoke.Reset();
    }

    void Update()
    {
        if (leftHand == null && Time.unscaledTime >= nextSearch)
        {
            nextSearch = Time.unscaledTime + 0.5f;
            foreach (var skeleton in trackingSpace.GetComponentsInChildren<XRHandSkeletonDriver>(true))
            {
                var events = skeleton.handTrackingEvents;
                if (events == null || events.handedness != Handedness.Left || skeleton.rootTransform == null)
                    continue;
                leftHand = events;
                wrist = skeleton.rootTransform;
                leftHand.poseUpdated.AddListener(OnWristPose);
                break;
            }
        }
        if (!WristTracked) hasPose = false;
        panel.gameObject.SetActive(panelOpen && WristTracked && hasPose);
        UpdateDemoControls();
    }

    void OnWristPose(Pose rootPose)
    {
        var space = wrist.parent;
        var wristPosition = space.TransformPoint(rootPose.position);
        var wristRotationWorld = space.rotation * rootPose.rotation;
        var targetPosition = trackingSpace.InverseTransformPoint(wristPosition + wristRotationWorld * wristOffset);
        var targetRotation = Quaternion.Inverse(trackingSpace.rotation) * wristRotationWorld * Quaternion.Euler(wristRotation);

        // Use elapsed time because this runs twice per frame.
        float now = Time.realtimeSinceStartup;
        float dt = now - lastSample;
        lastSample = now;
        if (!hasPose || dt <= 0 || dt > 0.25f)
        {
            position = targetPosition;
            rotation = targetRotation;
            positionSpeed = Vector3.zero;
            rotationSpeed = 0;
            hasPose = true;
        }
        else
        {
            positionSpeed = Vector3.Lerp(positionSpeed, (targetPosition - rawPosition) / dt, Alpha(1, dt));
            position = Vector3.Lerp(position, targetPosition,
                Alpha(minCutoff + positionBeta * positionSpeed.magnitude, dt));
            rotationSpeed = Mathf.Lerp(rotationSpeed, Quaternion.Angle(rawRotation, targetRotation) / dt, Alpha(1, dt));
            rotation = Quaternion.Slerp(rotation, targetRotation,
                Alpha(minCutoff + rotationBeta * rotationSpeed, dt));
        }
        rawPosition = targetPosition;
        rawRotation = targetRotation;

        panel.localPosition = position;
        panel.localRotation = rotation;
        panel.localScale = Vector3.one * panelScale;
    }

    static float Alpha(float cutoff, float dt)
    {
        float tau = 1 / (2 * Mathf.PI * cutoff);
        return 1 / (1 + tau / dt);
    }

    public void Toggle()
    {
        panelOpen = !panelOpen;
        panel.gameObject.SetActive(panelOpen && WristTracked && hasPose);
    }

    void BuildDemoControls()
    {
        panel.sizeDelta = new Vector2(panel.sizeDelta.x, Mathf.Max(350, panel.sizeDelta.y));
        readings.rectTransform.anchoredPosition += Vector2.up * 50;
        var accent = panel.Find("Accent") as RectTransform;
        if (accent != null) accent.anchoredPosition = new Vector2(0, panel.sizeDelta.y * .5f - 1);
        demoRoot = new GameObject("Oxygen Demo", typeof(RectTransform)).GetComponent<RectTransform>();
        demoRoot.SetParent(panel, false);
        demoRoot.gameObject.layer = panel.gameObject.layer;
        demoRoot.sizeDelta = new Vector2(340, 100);
        demoRoot.anchoredPosition = new Vector2(0, -110);
        demoLabel = DemoText(demoRoot, "DEMO O2", new Vector2(340, 28), new Vector2(0, 30), 24);
        demoButtons = new Button[3];
        string[] labels = { "-10 O2", "+10 O2", "REFILL" };
        for (int i = 0; i < 3; i++)
        {
            int index = i;
            var button = new GameObject(labels[i], typeof(RectTransform), typeof(Image), typeof(Button)).GetComponent<Button>();
            var rect = (RectTransform)button.transform;
            rect.SetParent(demoRoot, false);
            rect.gameObject.layer = panel.gameObject.layer;
            rect.sizeDelta = new Vector2(100, 55);
            rect.anchoredPosition = new Vector2((i - 1) * 112, -20);
            button.image.color = new Color(.05f, .25f, .32f, 1);
            button.targetGraphic = button.image;
            button.image.raycastTarget = false;
            DemoText(rect, labels[i], rect.sizeDelta, Vector2.zero, 22);
            button.onClick.AddListener(() => telemetry.Server?.ChangeDemoOxygen(
                index == 0 ? -10 : index == 1 ? 10 : 100, index != 2));
            demoButtons[i] = button;
        }
    }

    TMP_Text DemoText(RectTransform parent, string value, Vector2 size, Vector2 offset, float fontSize)
    {
        var text = new GameObject("Label", typeof(RectTransform), typeof(TextMeshProUGUI)).GetComponent<TextMeshProUGUI>();
        text.rectTransform.SetParent(parent, false);
        text.gameObject.layer = parent.gameObject.layer;
        text.rectTransform.sizeDelta = size;
        text.rectTransform.anchoredPosition = offset;
        text.font = readings.font;
        text.fontSharedMaterial = readings.fontSharedMaterial;
        text.fontSize = fontSize;
        text.alignment = TextAlignmentOptions.Center;
        text.color = new Color(.65f, .94f, 1);
        text.raycastTarget = false;
        text.text = value;
        return text;
    }

    void UpdateDemoControls()
    {
        FingerNearControls = false;
        if (demoRoot == null) return;
        demoRoot.gameObject.SetActive(showOxygenDemo);
        bool visible = showOxygenDemo && panel.gameObject.activeInHierarchy;
        if (!visible)
        {
            demoWasVisible = false;
            demoPoke.Reset();
            return;
        }
        if (!demoWasVisible)
        {
            demoOpenedAt = Time.unscaledTime;
            demoWasVisible = true;
        }
        var server = telemetry.Server;
        bool ready = telemetry.HasData && server != null && !server.OxygenDemoBusy;
        demoLabel.text = !telemetry.HasData ? "DEMO O2 / OFFLINE" : server != null && server.OxygenDemoError != null
            ? server.OxygenDemoError : $"DEMO O2  {telemetry.Current.oxygenPercent:0}%";
        foreach (var button in demoButtons)
        {
            button.interactable = ready;
            button.image.color = new Color(.05f, .25f, .32f, 1);
        }
        if (demoHands == null || !demoHands.running)
        {
            SubsystemManager.GetSubsystems(handSubsystems);
            demoHands = handSubsystems.Find(s => s.running);
        }
        var aim = MetaAimHand.right;
        bool systemGesture = aim != null && aim.added && ((MetaAimFlags)aim.aimFlags.ReadValue()
            & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) != 0;
        if (systemGesture || demoHands == null || !demoHands.rightHand.isTracked ||
            !demoHands.rightHand.GetJoint(XRHandJointID.IndexTip).TryGetPose(out var pose))
        {
            demoPoke.Reset();
            return;
        }
        var local = demoRoot.InverseTransformPoint(trackingSpace.TransformPoint(pose.position));
        float depth = local.z * demoRoot.lossyScale.z;
        bool over = demoRoot.rect.Contains(new Vector2(local.x, local.y));
        FingerNearControls = over && Mathf.Abs(depth) < .06f;
        bool frontVisible = demoRoot.InverseTransformPoint(viewCamera.transform.position).z < 0;
        if (!ready || !frontVisible || Time.unscaledTime - demoOpenedAt < .5f)
        {
            demoPoke.Reset();
            return;
        }
        var world = trackingSpace.TransformPoint(pose.position);
        int target = -1;
        for (int i = 0; i < demoButtons.Length; i++)
        {
            var rect = (RectTransform)demoButtons[i].transform;
            var point = rect.InverseTransformPoint(world);
            if (rect.rect.Contains(new Vector2(point.x, point.y)))
            {
                target = i;
                break;
            }
        }
        if (target >= 0 && depth < 0 && depth > -.06f)
            demoButtons[target].image.color = new Color(.08f, .48f, .58f, 1);
        if (demoPoke.TryPress(target, depth, Time.unscaledTime, demoPressDistance, demoReleaseDistance, demoDebounce))
            demoButtons[target].onClick.Invoke();
    }

    void Refresh(SuitReadings value)
    {
        if (!telemetry.HasData)
        {
            readings.text = "<color=#FFC04D>TELEMETRY UNAVAILABLE</color>";
            return;
        }
        readings.text = Row("HEART", $"{value.heartRate:0}", "BPM") + "\n" +
            Row("PRESSURE", $"{value.pressureKpa:0.0}", "kPa") + "\n" +
            Row("RADIATION", $"{value.radiationMicrosieverts:0.00}", "uSv/h") + "\n" +
            Row("TEMP", $"{value.temperatureCelsius:0.0}", "°C") + "\n" +
            Row("BATTERY", $"{value.batteryPercent:0}", "%");
    }

    static string Row(string label, string value, string unit) =>
        $"<color=#9DEAF4><size=70%>{label}</size></color><pos=165><b>{value}</b> <size=65%>{unit}</size>";
}
