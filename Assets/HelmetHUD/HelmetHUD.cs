using TMPro;
using UnityEngine;
using UnityEngine.UI;

[DefaultExecutionOrder(1000)]
public class HelmetHUD : MonoBehaviour
{
    [Header("Head-relative layout")]
    [SerializeField] Camera viewCamera;
    [SerializeField, Min(0.5f)] float distance = 1.6f;
    [SerializeField] float height;
    [SerializeField, Min(0.0001f)] float hudScale = 0.0015f;
    [Tooltip("Horizontal and vertical offsets in canvas units.")]
    [SerializeField] Vector2 edgeOffsets = new Vector2(430, 245);

    [Header("Compass")]
    [Tooltip("World-space north. Jezero uses +Z as north. Y is ignored.")]
    [SerializeField] Vector3 worldNorth = Vector3.forward;
    [SerializeField, Range(45, 160)] float compassSpan = 120;
    [SerializeField, Range(240, 640)] float compassWidth = 520;
    [SerializeField] RectTransform compassViewport;
    [SerializeField] RectTransform[] compassLabels;
    [SerializeField] RectTransform[] compassTicks;
    [SerializeField] TMP_Text headingText;

    [Header("Display")]
    [SerializeField] SuitTelemetry telemetry;
    [SerializeField] RectTransform vitalsRoot, compassRoot, agentRoot;
    [SerializeField] Vector2 agentOffset = new Vector2(350, -15);
    [SerializeField] TMP_Text oxygen;
    [SerializeField] RectTransform warningRoot;
    [SerializeField] TMP_Text alert;
    [SerializeField] Image statusLine;
    [SerializeField] RectTransform orb;
    [SerializeField] TMP_Text assistantStatus, subtitle;

    static readonly Color Cyan = new Color(0.35f, 0.9f, 1);
    static readonly Color Amber = new Color(1, 0.75f, 0.3f);
    static readonly Color Red = new Color(1, 0.35f, 0.35f);
    static readonly string[] MetricNames = { "OXYGEN", "HEART RATE", "SUIT PRESSURE", "BATTERY", "RADIATION", "SUIT TEMP" };
    readonly SuitAlert[] states = new SuitAlert[6];
    Vector3 horizontalForward = Vector3.forward;
    float diagnosticUntil;
    int displayedHeading = -1;
    string assistantMessage = "";
    float orbPulseSpeed = 1.4f, orbPulseAmount = 0.035f;
    Graphic orbGraphic;
    VoiceSessionClient voice;
    string statusAlert = "";

    public float Heading { get; private set; }

    void OnEnable()
    {
        if (viewCamera == null)
        {
            viewCamera = GetComponentInParent<Camera>();
        }
        if (viewCamera == null)
        {
            Debug.LogError("Helmet HUD needs the existing XR camera.", this);
            enabled = false;
            return;
        }
        GetComponent<Canvas>().worldCamera = viewCamera;
        ApplyLayout();
        UpdateCompass();
        // Read the latest head pose just before rendering.
        Application.onBeforeRender += UpdateCompass;
        telemetry.Changed += Refresh;
        Refresh(telemetry.Current);
    }

    void OnDisable()
    {
        Application.onBeforeRender -= UpdateCompass;
        if (telemetry != null) telemetry.Changed -= Refresh;
    }

    void LateUpdate()
    {
        if (voice == null) voice = FindFirstObjectByType<VoiceSessionClient>();
        string nextAlert = voice != null ? voice.Alert : "";
        if (nextAlert != statusAlert)
        {
            statusAlert = nextAlert;
            Refresh(telemetry.Current);
        }
        if (diagnosticUntil > 0 && Time.unscaledTime >= diagnosticUntil)
        {
            // Restore the reply after the gesture message.
            subtitle.text = assistantMessage;
            subtitle.gameObject.SetActive(!string.IsNullOrWhiteSpace(assistantMessage));
            diagnosticUntil = 0;
        }
        float pulse = Mathf.Sin(Time.unscaledTime * orbPulseSpeed);
        orb.localScale = Vector3.one * (1 + pulse * orbPulseAmount);
        orb.anchoredPosition = new Vector2(0, 28 + pulse * 2);
    }

    void ApplyLayout()
    {
        transform.localPosition = new Vector3(0, height, Mathf.Max(0.5f, distance));
        transform.localRotation = Quaternion.identity;
        transform.localScale = Vector3.one * Mathf.Max(0.0001f, hudScale);
        vitalsRoot.anchoredPosition = new Vector2(-edgeOffsets.x, edgeOffsets.y);
        compassRoot.anchoredPosition = new Vector2(0, edgeOffsets.y);
        agentRoot.anchoredPosition = agentOffset;
        warningRoot.anchoredPosition = new Vector2(0, -edgeOffsets.y + 30);
    }

    void OnValidate()
    {
        if (vitalsRoot != null && compassRoot != null && agentRoot != null && warningRoot != null)
            ApplyLayout();
    }

    void UpdateCompass()
    {
        var projected = Vector3.ProjectOnPlane(viewCamera.transform.forward, Vector3.up);
        if (projected.sqrMagnitude > 0.0001f)
            horizontalForward = projected.normalized;
        var north = Vector3.ProjectOnPlane(worldNorth, Vector3.up);
        if (north.sqrMagnitude < 0.0001f) north = Vector3.forward;
        Heading = Mathf.Repeat(Vector3.SignedAngle(north.normalized, horizontalForward, Vector3.up), 360);
        int degrees = Mathf.RoundToInt(Heading) % 360;
        if (degrees != displayedHeading)
        {
            headingText.text = $"{degrees:000}\u00b0";
            displayedHeading = degrees;
        }
        float width = Mathf.Clamp(compassWidth, 240, 640);
        float pixelsPerDegree = width / Mathf.Clamp(compassSpan, 45, 160);
        compassViewport.SetSizeWithCurrentAnchors(RectTransform.Axis.Horizontal, width);
        // Wrap outside the visible ribbon so north scrolls smoothly.
        for (int i = 0; i < compassLabels.Length; i++)
            compassLabels[i].anchoredPosition = new Vector2(Mathf.DeltaAngle(Heading, i * 45) * pixelsPerDegree, 4);
        for (int i = 0; i < compassTicks.Length; i++)
            compassTicks[i].anchoredPosition = new Vector2(Mathf.DeltaAngle(Heading, i * 15) * pixelsPerDegree, -17);
    }

    public void SetAssistant(string status, string message)
    {
        diagnosticUntil = 0;
        assistantMessage = message ?? "";
        assistantStatus.text = status;
        subtitle.text = assistantMessage;
        subtitle.gameObject.SetActive(!string.IsNullOrWhiteSpace(assistantMessage));
    }

    // Update the orb color and pulse for the voice state.
    public void SetAssistantStyle(Color tint, float pulseSpeed = 1.4f, float pulseAmount = 0.035f)
    {
        if (orbGraphic == null) orbGraphic = orb.GetComponent<Graphic>();
        if (orbGraphic != null) orbGraphic.color = tint;
        assistantStatus.color = tint;
        orbPulseSpeed = pulseSpeed;
        orbPulseAmount = pulseAmount;
    }

    public void ReportPinch(string message)
    {
        subtitle.text = message;
        subtitle.gameObject.SetActive(true);
        diagnosticUntil = Time.unscaledTime + 4;
    }

    void Refresh(SuitReadings value)
    {
        if (!telemetry.HasData)
        {
            SetValue(oxygen, "<size=60%>O2</size>  --", SuitAlert.Warning);
            warningRoot.gameObject.SetActive(true);
            alert.text = "TELEMETRY UNAVAILABLE";
            alert.color = statusLine.color = Amber;
            ApplyStatusAlert();
            return;
        }
        statusLine.color = Cyan;
        states[0] = Low(value.oxygenPercent, 30, 15);
        states[1] = Band(value.heartRate, 50, 110, 40, 140);
        states[2] = Band(value.pressureKpa, 28, 35, 25, 40);
        states[3] = Low(value.batteryPercent, 25, 10);
        states[4] = High(value.radiationMicrosieverts, 0.5f, 2);
        states[5] = Band(value.temperatureCelsius, 10, 32, 0, 40);
        SetValue(oxygen, $"<size=60%>O2</size>  {value.oxygenPercent:0}%", states[0]);

        int highest = 0;
        for (int i = 1; i < states.Length; i++)
            if (states[i] > states[highest]) highest = i;
        var state = states[highest];
        warningRoot.gameObject.SetActive(state == SuitAlert.Critical);
        if (state != SuitAlert.Critical)
        {
            ApplyStatusAlert();
            return;
        }
        string detail = highest == 3 ? $"  {value.batteryPercent:0}%" :
            highest == 4 ? $"  {value.radiationMicrosieverts:0.00} uSv/h" : "";
        alert.text = $"{state.ToString().ToUpperInvariant()} / {MetricNames[highest]}{detail}";
        alert.color = statusLine.color = ColorFor(state);
        ApplyStatusAlert();
    }

    // Router alerts take precedence; clearing restores the current telemetry warning.
    void ApplyStatusAlert()
    {
        if (string.IsNullOrWhiteSpace(statusAlert)) return;
        warningRoot.gameObject.SetActive(true);
        alert.text = statusAlert;
        alert.color = statusLine.color = Red;
    }

    static void SetValue(TMP_Text text, string value, SuitAlert state)
    {
        text.text = value;
        text.color = ColorFor(state);
    }

    static SuitAlert Low(float value, float warning, float critical) =>
        value <= critical ? SuitAlert.Critical : value <= warning ? SuitAlert.Warning : SuitAlert.Normal;

    static SuitAlert High(float value, float warning, float critical) =>
        value >= critical ? SuitAlert.Critical : value >= warning ? SuitAlert.Warning : SuitAlert.Normal;

    static SuitAlert Band(float value, float lowWarning, float highWarning, float lowCritical, float highCritical) =>
        value <= lowCritical || value >= highCritical ? SuitAlert.Critical :
        value <= lowWarning || value >= highWarning ? SuitAlert.Warning : SuitAlert.Normal;

    static Color ColorFor(SuitAlert state) => state == SuitAlert.Critical ? Red : state == SuitAlert.Warning ? Amber : Cyan;
}
