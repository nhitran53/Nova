using NUnit.Framework;

// Pure parsing / change-detection tests; no scene, no network.
public class VoiceSessionClientTests
{
    const string Idle = "{\"state\": \"idle\", \"last_transcript\": null, \"last_reply\": null, \"route\": null, \"error\": null, \"session_id\": 0, \"seq\": 0, \"router_ready\": true}";

    static VoiceStatus Status(string state, string transcript = "", string reply = "", string error = "", int session = 1)
        => new VoiceStatus { state = state, last_transcript = transcript, last_reply = reply, error = error, session_id = session };

    [Test]
    public void Parse_ServerIdleBodyWithNulls()
    {
        Assert.IsTrue(VoiceStatusParser.TryParse(Idle, out var status));
        Assert.AreEqual("idle", status.state);
        Assert.AreEqual("", status.last_transcript);
        Assert.AreEqual("", status.error);
        Assert.IsTrue(status.router_ready);
    }

    [Test]
    public void Parse_SpeakingBodyAndStartedFlag()
    {
        const string json = "{\"state\": \"speaking\", \"last_transcript\": \"check oxygen\", \"last_reply\": \"Oxygen is 84 percent.\", \"route\": \"tool\", \"error\": null, \"session_id\": 3, \"seq\": 7, \"router_ready\": true, \"started\": false}";
        Assert.IsTrue(VoiceStatusParser.TryParse(json, out var status));
        Assert.AreEqual("check oxygen", status.last_transcript);
        Assert.AreEqual("Oxygen is 84 percent.", status.last_reply);
        Assert.AreEqual("tool", status.route);
        Assert.AreEqual(3, status.session_id);
        Assert.AreEqual(7, status.seq);
        Assert.IsFalse(status.started);
    }

    [TestCase("")]
    [TestCase("   ")]
    [TestCase("not json")]
    [TestCase("{}")]
    [TestCase("{\"error\": \"nope\"}")]
    public void Parse_RejectsMalformedOrStatelessBodies(string json)
    {
        Assert.IsFalse(VoiceStatusParser.TryParse(json, out var status));
        Assert.IsNull(status);
    }

    [Test]
    public void IsActive_OnlyWhileLoadingListeningOrSpeaking()
    {
        Assert.IsTrue(VoiceStates.IsActive("loading"));
        Assert.IsTrue(VoiceStates.IsActive("listening"));
        Assert.IsTrue(VoiceStates.IsActive("speaking"));
        Assert.IsFalse(VoiceStates.IsActive("idle"));
        Assert.IsFalse(VoiceStates.IsActive("ended"));
        Assert.IsFalse(VoiceStates.IsActive("error"));
    }

    [Test]
    public void Tracker_ReportsStateChangesOnlyOnce()
    {
        var tracker = new VoiceStatusTracker();
        Assert.IsTrue(tracker.Apply(Status("listening")).StateChanged);
        Assert.IsFalse(tracker.Apply(Status("listening")).StateChanged);
        Assert.AreEqual("listening", tracker.State);
    }

    [Test]
    public void Tracker_RepeatedTranscriptInASecondTurnIsStillNew()
    {
        var tracker = new VoiceStatusTracker();
        tracker.Apply(Status("listening"));
        var first = tracker.Apply(Status("speaking", "yes", "Okay."));
        Assert.IsTrue(first.TranscriptChanged);
        Assert.IsTrue(first.ReplyChanged);
        Assert.IsFalse(tracker.Apply(Status("speaking", "yes", "Okay.")).TranscriptChanged);   // same poll again
        tracker.Apply(Status("listening", "yes", "Okay."));
        var second = tracker.Apply(Status("speaking", "yes", "Okay."));
        Assert.IsTrue(second.TranscriptChanged);
        Assert.IsTrue(second.ReplyChanged);
    }

    [Test]
    public void Tracker_ErrorFiresOncePerMessage()
    {
        var tracker = new VoiceStatusTracker();
        Assert.IsTrue(tracker.Apply(Status("error", error: "SttError: no key")).ErrorChanged);
        Assert.IsFalse(tracker.Apply(Status("error", error: "SttError: no key")).ErrorChanged);
    }

    [Test]
    public void Tracker_NewSessionForgetsOldTexts()
    {
        var tracker = new VoiceStatusTracker();
        tracker.Apply(Status("speaking", "hello", "Hi.", session: 1));
        tracker.Apply(Status("ended", "hello", "Hi.", session: 1));
        var next = tracker.Apply(Status("loading", session: 2));
        Assert.IsTrue(next.StateChanged);
        Assert.IsFalse(next.TranscriptChanged);
        Assert.AreEqual("", tracker.LastTranscript);
        Assert.IsTrue(tracker.Apply(Status("speaking", "hello", "Hi.", session: 2)).TranscriptChanged);
    }
}
