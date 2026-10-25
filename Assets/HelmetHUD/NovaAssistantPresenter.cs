using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// Show the voice server state and latest reply on the HUD.
public class NovaAssistantPresenter : MonoBehaviour
{
    // Standby waits for a connection. Ready is idle. Starting waits for the router.
    public enum Look { Standby, Ready, Starting, Listening, Processing, Speaking, Error, Offline }

    [SerializeField] VoiceSessionClient voice;
    [Tooltip("Leave empty: the Helmet HUD on the active camera. The desktop rig has none, so there it only logs.")]
    [SerializeField] HelmetHUD hud;
    [Tooltip("Optional. Its Diagnostic is shown on the HUD so the pose thresholds can be tuned in the headset.")]
    [SerializeField] TelephoneGestureDetector telephone;
    [SerializeField] bool showGestureDiagnostics = true;
    [Tooltip("V = start/stop the voice session (desktop and Editor testing).")]
    [SerializeField] bool keyboardShortcut = true;
    [Tooltip("Keep NOVA's last reply on the HUD this long after the conversation ends.")]
    [SerializeField, Min(0)] float replyHoldSeconds = 8;
    [SerializeField, Min(20)] int maxMessageChars = 160;

    static readonly Color Cyan = new Color(0.35f, 0.9f, 1);
    static readonly Color Amber = new Color(1, 0.75f, 0.3f);
    static readonly Color Green = new Color(0.45f, 1, 0.6f);
    static readonly Color Red = new Color(1, 0.35f, 0.35f);
    static readonly Color Dim = new Color(0.55f, 0.65f, 0.7f);

    // Shared voice display state.
    public Look Current { get; private set; } = Look.Standby;

    Camera boundCamera;
    bool styled;
    string shownStatus, shownMessage, shownDiagnostic;
    float replyUntil;

    public static Look Resolve(string state, bool connectionKnown, bool reachable, bool routerReady)
    {
        if (!connectionKnown) return Look.Standby;
        if (!reachable) return Look.Offline;
        switch (state)
        {
            case VoiceStates.Error: return Look.Error;
            case VoiceStates.Loading: return Look.Starting;
            case VoiceStates.Listening: return Look.Listening;
            case VoiceStates.Processing: return Look.Processing;
            case VoiceStates.Speaking: return Look.Speaking;
            default: return routerReady ? Look.Ready : Look.Starting;   // idle / ended
        }
    }

    void Update()
    {
#if ENABLE_INPUT_SYSTEM
        if (keyboardShortcut && voice != null && Keyboard.current != null && Keyboard.current.vKey.wasPressedThisFrame)
            voice.ToggleVoiceSession();
#endif
    }

    void LateUpdate()
    {
        if (voice == null) return;
        var cam = Camera.main;
        if (cam != boundCamera)
        {
            boundCamera = cam;
            if (cam != null && (hud == null || !hud.isActiveAndEnabled)) hud = cam.GetComponentInChildren<HelmetHUD>(true);
            shownStatus = shownMessage = null;
            styled = false;
        }

        var look = Resolve(voice.State, voice.ConnectionKnown, voice.ServerReachable, voice.RouterReady);
        if (look != Current)
        {
            if (Current == Look.Speaking || Current == Look.Listening) replyUntil = Time.unscaledTime + replyHoldSeconds;
            Current = look;
            styled = false;
            Debug.Log($"[Nova] {StatusText(look)}" + (look == Look.Error ? $": {voice.LastError}" : ""), this);
        }
        if (hud == null || !hud.isActiveAndEnabled) return;

        if (!styled)
        {
            styled = true;
            ApplyStyle(look);
        }
        string vision = look == Look.Processing ? VisionStatus(voice.VisionState) : null;
        string status = vision ?? StatusText(look);
        string message = Clip(vision != null && !string.IsNullOrEmpty(voice.VisionTarget) ? voice.VisionTarget : MessageFor(look));
        if (status != shownStatus || message != shownMessage)
        {
            shownStatus = status;
            shownMessage = message;
            hud.SetAssistant(status, message);
        }

        if (showGestureDiagnostics && telephone != null && telephone.Diagnostic != shownDiagnostic)
        {
            shownDiagnostic = telephone.Diagnostic;
            if (!string.IsNullOrEmpty(shownDiagnostic)) hud.ReportPinch("PHONE " + shownDiagnostic);
        }
    }

    static string StatusText(Look look)
    {
        switch (look)
        {
            case Look.Ready: return "NOVA / READY";
            case Look.Starting: return "NOVA / STARTING";
            case Look.Listening: return "NOVA / LISTENING";
            case Look.Processing: return "NOVA / PROCESSING REQUEST";
            case Look.Speaking: return "NOVA / SPEAKING";
            case Look.Error: return "NOVA / ERROR";
            case Look.Offline: return "NOVA / CONNECTION LOST";
            default: return "NOVA / STANDBY";
        }
    }

    // Keep the normal status when no vision state applies.
    public static string VisionStatus(string visionState)
    {
        switch (visionState)
        {
            case "capturing": return "NOVA / CAPTURING VIEW";
            case "analyzing": return "NOVA / ANALYZING";
            case "identified": return "NOVA / TARGET IDENTIFIED";
            case "error": return "NOVA / VISION ERROR";
            default: return null;
        }
    }

    string MessageFor(Look look)
    {
        switch (look)
        {
            case Look.Processing: return string.IsNullOrEmpty(voice.LastTranscript) ? "" : $"\"{voice.LastTranscript}\"";
            case Look.Speaking:
            case Look.Listening: return voice.LastReply;   // keep the latest reply readable between turns
            case Look.Error: return voice.LastError;
            case Look.Offline: return "Voice server unreachable";
            case Look.Ready:
            case Look.Starting: return Time.unscaledTime < replyUntil ? voice.LastReply : "";
            default: return "";
        }
    }

    void ApplyStyle(Look look)
    {
        switch (look)
        {
            case Look.Ready: hud.SetAssistantStyle(Cyan); break;   // the original idle pulse
            case Look.Starting: hud.SetAssistantStyle(Amber, 2.5f, 0.04f); break;
            case Look.Listening: hud.SetAssistantStyle(Cyan, 3.5f, 0.07f); break;
            case Look.Processing: hud.SetAssistantStyle(Amber, 7f, 0.05f); break;
            case Look.Speaking: hud.SetAssistantStyle(Green, 10f, 0.08f); break;
            case Look.Error: hud.SetAssistantStyle(Red, 0, 0); break;
            case Look.Offline: hud.SetAssistantStyle(Red, 0.6f, 0.015f); break;
            default: hud.SetAssistantStyle(Dim, 0.8f, 0.02f); break;
        }
    }

    string Clip(string text)
    {
        if (string.IsNullOrEmpty(text)) return "";
        return text.Length <= maxMessageChars ? text : text.Substring(0, maxMessageChars - 3) + "...";
    }
}
