using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

// Control the PC voice session and read its status and telemetry.
public class VoiceSessionClient : MonoBehaviour
{
    [Tooltip("voice_server address. localhost only works in the Editor / on the same PC; a headset needs the PC's LAN address.")]
    [SerializeField] string baseUrl = "http://127.0.0.1:8765";
    [SerializeField] float pollIntervalSeconds = 0.5f;
    [Tooltip("Poll interval while the server is unreachable.")]
    [SerializeField] float unreachablePollIntervalSeconds = 3f;
    [SerializeField] float requestTimeoutSeconds = 3f;
    [Tooltip("While no conversation runs, check the server this often so the HUD can show ready / connection lost. 0 = off.")]
    [SerializeField] float idleCheckSeconds = 5f;

    [Header("Telemetry")]
    [SerializeField, Min(0.2f)] float telemetryPollSeconds = 0.5f;
    [SerializeField, Min(1f)] float telemetryTimeoutSeconds = 3f;
    public TelemetryStatus Telemetry { get; private set; } = new TelemetryStatus();
    float telemetryReceivedAt;
    public bool TelemetryAvailable => isActiveAndEnabled && Telemetry.IsFresh(
        Time.realtimeSinceStartup - telemetryReceivedAt, telemetryTimeoutSeconds);
    public bool OxygenDemoBusy { get; private set; }
    public string OxygenDemoError { get; private set; }

    public void ChangeDemoOxygen(float value, bool relative = true)
    {
        if (!TelemetryAvailable || OxygenDemoBusy || float.IsNaN(value) || float.IsInfinity(value)) return;
        StartCoroutine(OxygenDemoRoutine(value, relative));
    }

    [System.Serializable] public class StringEvent : UnityEvent<string> { }

    public StringEvent OnStateChanged = new StringEvent();
    public StringEvent OnTranscript = new StringEvent();
    public StringEvent OnReply = new StringEvent();
    public StringEvent OnError = new StringEvent();
    // Sends the destination when a new navigation command arrives.
    public StringEvent OnNavigation = new StringEvent();

    readonly VoiceStatusTracker tracker = new VoiceStatusTracker();
    bool polling;
    bool commandInFlight;
    bool unreachableReported;

    // One of VoiceStates (idle, loading, listening, speaking, ended, error).
    public string State => tracker.State;
    public bool IsActive => VoiceStates.IsActive(State);
    public bool ServerReachable { get; private set; } = true;
    // False until the server has answered (or failed to answer) at least once.
    public bool ConnectionKnown { get; private set; }
    public bool RouterReady => tracker.RouterReady;
    public string LastTranscript => tracker.LastTranscript;
    public string LastReply => tracker.LastReply;
    public string LastError => tracker.LastError;
    public string LastRoute => tracker.LastRoute;
    public string NavigationAction => tracker.NavigationAction;

    public string BaseUrl
    {
        get => baseUrl;
        set => baseUrl = value ?? "";
    }

    public void StartVoiceSession() => SendCommand("/voice/start");
    public void StopVoiceSession() => SendCommand("/voice/stop");
    public void ToggleVoiceSession()
    {
        if (IsActive) StopVoiceSession();
        else StartVoiceSession();
    }

    void OnEnable()
    {
        // Standalone builds need the PC address.
        if (Application.platform == RuntimePlatform.Android && IsLoopback(baseUrl))
            Report($"Base URL {baseUrl} points at the headset itself. Use the PC's LAN address (server started with --host).");
        if (idleCheckSeconds > 0) StartCoroutine(IdleCheckRoutine());
        StartCoroutine(TelemetryRoutine());
    }

    void OnDisable()
    {
        StopAllCoroutines();
        polling = false;
        commandInFlight = false;
        Telemetry = new TelemetryStatus();
        OxygenDemoBusy = false;
    }

    static bool IsLoopback(string url) =>
        System.Uri.TryCreate(url, System.UriKind.Absolute, out var uri) && (uri.IsLoopback || uri.Host == "localhost");

    // Check the connection while voice is idle.
    IEnumerator IdleCheckRoutine()
    {
        while (true)
        {
            if (!polling && !commandInFlight && !string.IsNullOrWhiteSpace(baseUrl))
            {
                using (var req = UnityWebRequest.Get(Url("/voice/status")))
                {
                    req.timeout = Mathf.Max(1, Mathf.CeilToInt(requestTimeoutSeconds));
                    yield return req.SendWebRequest();
                    if (req.result == UnityWebRequest.Result.Success)
                    {
                        MarkReachable();
                        if (VoiceStatusParser.TryParse(req.downloadHandler.text, out var status)) Apply(status);
                        if (IsActive) EnsurePolling();
                    }
                    else ReportUnreachable("GET /voice/status", req);
                }
            }
            yield return new WaitForSeconds(Mathf.Max(1f, idleCheckSeconds));
        }
    }

    string Url(string path) => baseUrl.TrimEnd('/') + path;

    IEnumerator TelemetryRoutine()
    {
        while (true)
        {
            if (!string.IsNullOrWhiteSpace(baseUrl))
            {
                using (var req = UnityWebRequest.Get(Url("/telemetry")))
                {
                    req.timeout = Mathf.Max(1, Mathf.CeilToInt(requestTimeoutSeconds));
                    yield return req.SendWebRequest();
                    if (req.result == UnityWebRequest.Result.Success &&
                        TelemetryStatusParser.TryParse(req.downloadHandler.text, out var status))
                    {
                        Telemetry = status;
                        telemetryReceivedAt = Time.realtimeSinceStartup;
                    }
                    else Telemetry = new TelemetryStatus();
                }
            }
            else Telemetry = new TelemetryStatus();
            yield return new WaitForSecondsRealtime(Mathf.Max(0.2f, telemetryPollSeconds));
        }
    }

    IEnumerator OxygenDemoRoutine(float value, bool relative)
    {
        OxygenDemoBusy = true;
        OxygenDemoError = null;
        string body = "{\"" + (relative ? "delta" : "value") + "\":"
            + value.ToString(System.Globalization.CultureInfo.InvariantCulture) + "}";
        using (var req = new UnityWebRequest(Url("/demo/oxygen"), UnityWebRequest.kHttpVerbPOST))
        {
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = Mathf.Max(1, Mathf.CeilToInt(requestTimeoutSeconds));
            yield return req.SendWebRequest();
            if (req.result != UnityWebRequest.Result.Success)
                OxygenDemoError = "O2 control unavailable";
        }
        OxygenDemoBusy = false;
    }

    void SendCommand(string path)
    {
        if (!isActiveAndEnabled) return;
        if (string.IsNullOrWhiteSpace(baseUrl))
        {
            Report("Voice server URL is empty.");
            return;
        }
        if (commandInFlight) return;   // a double tap must not send two commands
        StartCoroutine(CommandRoutine(path));
    }

    IEnumerator CommandRoutine(string path)
    {
        commandInFlight = true;
        using (var req = new UnityWebRequest(Url(path), UnityWebRequest.kHttpVerbPOST))
        {
            req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes("{}"));
            req.downloadHandler = new DownloadHandlerBuffer();
            req.SetRequestHeader("Content-Type", "application/json");
            req.timeout = Mathf.Max(1, Mathf.CeilToInt(requestTimeoutSeconds));
            yield return req.SendWebRequest();
            if (req.result == UnityWebRequest.Result.Success)
            {
                MarkReachable();
                if (VoiceStatusParser.TryParse(req.downloadHandler.text, out var status))
                {
                    Apply(status);
                    if (path == "/voice/start" && !status.started)
                        Debug.Log("[Voice] /voice/start ignored: a conversation is already running or still shutting down.");
                }
                else Debug.LogWarning($"[Voice] POST {path}: unreadable answer.");
                if (IsActive) EnsurePolling();
            }
            else ReportUnreachable($"POST {path}", req);
        }
        commandInFlight = false;
    }

    void EnsurePolling()
    {
        if (polling || !isActiveAndEnabled) return;
        polling = true;
        StartCoroutine(PollRoutine());
    }

    // Poll during voice sessions and retry more slowly while offline.
    IEnumerator PollRoutine()
    {
        while (true)
        {
            using (var req = UnityWebRequest.Get(Url("/voice/status")))
            {
                req.timeout = Mathf.Max(1, Mathf.CeilToInt(requestTimeoutSeconds));
                yield return req.SendWebRequest();
                if (req.result == UnityWebRequest.Result.Success)
                {
                    MarkReachable();
                    if (VoiceStatusParser.TryParse(req.downloadHandler.text, out var status)) Apply(status);
                    if (!IsActive) break;
                }
                else
                {
                    ReportUnreachable("GET /voice/status", req);
                    yield return new WaitForSeconds(Mathf.Max(0.5f, unreachablePollIntervalSeconds));
                    continue;
                }
            }
            yield return new WaitForSeconds(Mathf.Max(0.1f, pollIntervalSeconds));
        }
        polling = false;
    }

    void Apply(VoiceStatus status)
    {
        var changes = tracker.Apply(status);
        if (changes.StateChanged) OnStateChanged.Invoke(tracker.State);
        if (changes.TranscriptChanged) OnTranscript.Invoke(tracker.LastTranscript);
        if (changes.ReplyChanged) OnReply.Invoke(tracker.LastReply);
        if (changes.ErrorChanged) OnError.Invoke(tracker.LastError);
        if (changes.NavigationRequested) OnNavigation.Invoke(tracker.NavigationDestination);
    }

    void MarkReachable()
    {
        ServerReachable = ConnectionKnown = true;
        unreachableReported = false;
    }

    void ReportUnreachable(string what, UnityWebRequest req)
    {
        ServerReachable = false;
        ConnectionKnown = true;
        if (unreachableReported) return;   // one diagnostic per outage, not one per poll
        unreachableReported = true;
        Report($"{what} failed ({req.responseCode} {req.error}). Is `python -m router.voice_server` running at {baseUrl}? "
               + "A headset cannot reach the PC's localhost; use the PC's LAN address and start the server with --host.");
    }

    void Report(string message)
    {
        Debug.LogWarning($"[Voice] {message}");
        OnError.Invoke(message);
    }
}
