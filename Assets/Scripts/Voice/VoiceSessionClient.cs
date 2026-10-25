using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.Networking;

// Starts and stops the spoken conversation run by router/voice_server.py and reports its progress.
// Python owns the microphone and speaker on the PC; this component only sends commands and reads status.
// See README.md in this folder. If the server is not running it logs one warning and keeps trying quietly.
public class VoiceSessionClient : MonoBehaviour
{
    [Tooltip("voice_server address. localhost only works in the Editor / on the same PC; a headset needs the PC's LAN address.")]
    [SerializeField] string baseUrl = "http://127.0.0.1:8765";
    [SerializeField] float pollIntervalSeconds = 0.5f;
    [Tooltip("Poll interval while the server is unreachable.")]
    [SerializeField] float unreachablePollIntervalSeconds = 3f;
    [SerializeField] float requestTimeoutSeconds = 3f;

    [System.Serializable] public class StringEvent : UnityEvent<string> { }

    public StringEvent OnStateChanged = new StringEvent();
    public StringEvent OnTranscript = new StringEvent();
    public StringEvent OnReply = new StringEvent();
    public StringEvent OnError = new StringEvent();

    readonly VoiceStatusTracker tracker = new VoiceStatusTracker();
    bool polling;
    bool commandInFlight;
    bool unreachableReported;

    // One of VoiceStates (idle, loading, listening, speaking, ended, error).
    public string State => tracker.State;
    public bool IsActive => VoiceStates.IsActive(State);
    public bool ServerReachable { get; private set; } = true;

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

    void OnDisable()
    {
        StopAllCoroutines();
        polling = false;
        commandInFlight = false;
    }

    string Url(string path) => baseUrl.TrimEnd('/') + path;

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

    // Polls while a conversation is active (retrying more slowly when the server cannot be reached); stops once the state is terminal.
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
    }

    void MarkReachable()
    {
        ServerReachable = true;
        unreachableReported = false;
    }

    void ReportUnreachable(string what, UnityWebRequest req)
    {
        ServerReachable = false;
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
