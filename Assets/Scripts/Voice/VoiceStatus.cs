using System;
using UnityEngine;

// States reported by GET /voice/status (router/voice_server.py).
public static class VoiceStates
{
    public const string Idle = "idle";
    public const string Loading = "loading";
    public const string Listening = "listening";
    public const string Speaking = "speaking";
    public const string Ended = "ended";
    public const string Error = "error";

    // A conversation is running on the server while loading, listening or speaking.
    public static bool IsActive(string state) => state == Loading || state == Listening || state == Speaking;
}

// JSON body of GET /voice/status (and of the POST /voice/start and /voice/stop answers). Field names match the server.
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
    // False for malformed JSON or a body without "state". JSON null strings become "".
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

// Turns successive status snapshots into "what is new", so the same words in two turns still raise two events.
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
    int sessionId;

    public Changes Apply(VoiceStatus status)
    {
        var changes = new Changes();
        if (status.session_id != sessionId)   // a new conversation: forget the old texts
        {
            sessionId = status.session_id;
            LastTranscript = LastReply = LastError = "";
        }
        changes.StateChanged = status.state != State;
        // A turn arrives as the move to "speaking"; compare that too so a repeated "yes" still counts as new.
        bool newTurn = changes.StateChanged && status.state == VoiceStates.Speaking;
        changes.TranscriptChanged = status.last_transcript.Length > 0 && (status.last_transcript != LastTranscript || newTurn);
        changes.ReplyChanged = status.last_reply.Length > 0 && (status.last_reply != LastReply || newTurn);
        changes.ErrorChanged = status.error.Length > 0 && status.error != LastError;
        State = status.state;
        LastTranscript = status.last_transcript;
        LastReply = status.last_reply;
        LastError = status.error;
        return changes;
    }
}
