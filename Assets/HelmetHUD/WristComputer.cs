using TMPro;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.Hands;

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
        panel.GetComponent<Canvas>().worldCamera = origin.Camera;
        // Smooth wrist tracking without smoothing walking or snap turns.
        panel.SetParent(trackingSpace, false);
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

    void Refresh(SuitReadings value)
    {
        readings.text = Row("HEART", $"{value.heartRate:0}", "BPM") + "\n" +
            Row("PRESSURE", $"{value.pressureKpa:0.0}", "kPa") + "\n" +
            Row("RADIATION", $"{value.radiationMicrosieverts:0.00}", "uSv/h") + "\n" +
            Row("TEMP", $"{value.temperatureCelsius:0.0}", "°C") + "\n" +
            Row("BATTERY", $"{value.batteryPercent:0}", "%");
    }

    static string Row(string label, string value, string unit) =>
        $"<color=#9DEAF4><size=70%>{label}</size></color><pos=165><b>{value}</b> <size=65%>{unit}</size>";
}
