using TMPro;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.Hands;

public class WristComputer : MonoBehaviour
{
    [SerializeField] SuitTelemetry telemetry;
    [SerializeField] RectTransform panel;
    [SerializeField] TMP_Text readings;
    [SerializeField] Vector3 wristOffset = new Vector3(0, 0.045f, -0.025f);
    [SerializeField] Vector3 wristRotation = new Vector3(90, 0, 0);
    [SerializeField, Min(0.0001f)] float panelScale = 0.0004f;
    [SerializeField] bool panelOpen = true;

    Transform trackingSpace;
    XRHandTrackingEvents leftHand;
    float nextSearch;

    public bool IsOpen => panelOpen;
    public bool WristTracked => leftHand != null && leftHand.handIsTracked;

    void OnEnable()
    {
        var origin = GetComponentInParent<XROrigin>();
        trackingSpace = origin.CameraFloorOffsetObject.transform;
        panel.GetComponent<Canvas>().worldCamera = origin.Camera;
        telemetry.Changed += Refresh;
        Refresh(telemetry.Current);
        panel.gameObject.SetActive(false);
    }

    void OnDisable()
    {
        if (telemetry != null) telemetry.Changed -= Refresh;
        if (panel != null) panel.gameObject.SetActive(false);
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
                panel.SetParent(skeleton.rootTransform, false);
                ApplyWristPose();
                break;
            }
        }
        panel.gameObject.SetActive(panelOpen && WristTracked);
    }

    void ApplyWristPose()
    {
        panel.localPosition = wristOffset;
        panel.localRotation = Quaternion.Euler(wristRotation);
        panel.localScale = Vector3.one * panelScale;
    }

    void OnValidate()
    {
        if (panel != null && leftHand != null) ApplyWristPose();
    }

    public void Toggle()
    {
        panelOpen = !panelOpen;
        panel.gameObject.SetActive(panelOpen && WristTracked);
    }

    void Refresh(SuitReadings value)
    {
        readings.text = $"<size=80%>SUIT / {(telemetry.IsSimulated ? "SIM" : "EXT")}</size>\n" +
            $"HEART   {value.heartRate:0} BPM\nPRESS   {value.pressureKpa:0.0} kPa\n" +
            $"RAD     {value.radiationMicrosieverts:0.00} uSv/h\nTEMP    {value.temperatureCelsius:0.0}\u00b0C\n" +
            $"POWER   {value.batteryPercent:0}%\nOXYGEN  {value.oxygenPercent:0}%";
    }
}
