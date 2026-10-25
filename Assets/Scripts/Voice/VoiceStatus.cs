using System;
using UnityEngine;

[Serializable]
public class TelemetryStatus
{
    public int schema_version;
    public bool available;
    public string source;
    public int mission_id;
    public long sample_id;
    public float age_seconds = float.NaN;
    public float max_age_seconds = float.NaN;
    public float o2_percent = float.NaN;
    public float heart_rate = float.NaN;
    public float suit_pressure_kpa = float.NaN;
    public float battery_percent = float.NaN;
    public float radiation_usv_h = float.NaN;
    public float suit_temp_c = float.NaN;

    public bool IsFresh(float elapsed, float timeout) => available && elapsed >= 0 && elapsed <= timeout
        && Mathf.Abs(age_seconds + elapsed) <= max_age_seconds;
}

public static class TelemetryStatusParser
{
    public static bool TryParse(string json, out TelemetryStatus status)
    {
        status = null;
        if (string.IsNullOrWhiteSpace(json)) return false;
        TelemetryStatus parsed;
        try { parsed = JsonUtility.FromJson<TelemetryStatus>(json); }
        catch (ArgumentException) { return false; }
        if (parsed == null || parsed.schema_version != 1) return false;
        if (parsed.available)
        {
            if (parsed.source != "postgres" || parsed.mission_id <= 0 || parsed.sample_id <= 0) return false;
            foreach (var value in new[] { parsed.age_seconds, parsed.max_age_seconds, parsed.o2_percent,
                parsed.heart_rate, parsed.suit_pressure_kpa, parsed.battery_percent, parsed.radiation_usv_h, parsed.suit_temp_c })
                if (float.IsNaN(value) || float.IsInfinity(value)) return false;
            if (parsed.max_age_seconds <= 0 || !parsed.IsFresh(0, 1) || parsed.o2_percent < 0 || parsed.o2_percent > 100
                || parsed.battery_percent < 0 || parsed.battery_percent > 100 || parsed.heart_rate < 0
                || parsed.heart_rate > 250 || parsed.suit_pressure_kpa < 0 || parsed.radiation_usv_h < 0) return false;
        }
        status = parsed;
        return true;
    }
}

// States reported by GET /voice/status (router/voice_server.py).
public static class VoiceStates
{
    public const string Idle = "idle";
    public const string Loading = "loading";
    public const string Listening = "listening";
    // Processing lasts until the reply is ready.
    public const string Processing = "processing";
    public const string Speaking = "speaking";
    public const string Ended = "ended";
    public const string Error = "error";

    // States with an active conversation.
    public static bool IsActive(string state) =>
        state == Loading || state == Listening || state == Processing || state == Speaking;
}

// Fields match the voice server response.
[Serializable]
public class VoiceStatus
{
    public string state = "";
    public string last_transcript = "";
    public string last_reply = "";
    public string route = "";
    public string error = "";
    public int session_id;
    public int seq;
    public bool router_ready;
    public bool started;   // only present in the /voice/start answer: false = ignored (already running)
}

public static class VoiceStatusParser
{
    // Reject missing state and normalize null text.
    public static bool TryParse(string json, out VoiceStatus status)
    {
        status = null;
        if (string.IsNullOrWhiteSpace(json)) return false;
        VoiceStatus parsed;
        try { parsed = JsonUtility.FromJson<VoiceStatus>(json); }
        catch (ArgumentException) { return false; }
        if (parsed == null || string.IsNullOrEmpty(parsed.state)) return false;
        parsed.last_transcript = parsed.last_transcript ?? "";
        parsed.last_reply = parsed.last_reply ?? "";
        parsed.route = parsed.route ?? "";
        parsed.error = parsed.error ?? "";
        status = parsed;
        return true;
    }
}

// Report new turns even when their words repeat.
public sealed class VoiceStatusTracker
{
    public struct Changes
    {
        public bool StateChanged, TranscriptChanged, ReplyChanged, ErrorChanged;
    }

    public string State { get; private set; } = VoiceStates.Idle;
    public string LastTranscript { get; private set; } = "";
    public string LastReply { get; private set; } = "";
    public string LastError { get; private set; } = "";
    // Last reply route. Python still owns tool selection.
    public string LastRoute { get; private set; } = "";
    public bool RouterReady { get; private set; }
    int sessionId;

    public Changes Apply(VoiceStatus status)
    {
        var changes = new Changes();
        if (status.session_id != sessionId)   // a new conversation: forget the old texts
        {
            sessionId = status.session_id;
            LastTranscript = LastReply = LastError = LastRoute = "";
        }
        changes.StateChanged = status.state != State;
        // Use state changes so a repeated reply still counts as a new turn.
        bool newUtterance = changes.StateChanged && (status.state == VoiceStates.Processing ||
            (status.state == VoiceStates.Speaking && State != VoiceStates.Processing));
        bool newReply = changes.StateChanged && status.state == VoiceStates.Speaking;
        changes.TranscriptChanged = status.last_transcript.Length > 0 && (status.last_transcript != LastTranscript || newUtterance);
        changes.ReplyChanged = status.last_reply.Length > 0 && (status.last_reply != LastReply || newReply);
        changes.ErrorChanged = status.error.Length > 0 && status.error != LastError;
        State = status.state;
        LastTranscript = status.last_transcript;
        LastReply = status.last_reply;
        LastError = status.error;
        LastRoute = status.route;
        RouterReady = status.router_ready;
        return changes;
    }
}
