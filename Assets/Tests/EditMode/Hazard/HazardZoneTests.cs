using NUnit.Framework;
using UnityEngine;

public class HazardZoneTests
{
    [Test]
    public void VoiceStatus_ParsesMeteorAndAlert()
    {
        const string json = "{\"state\":\"idle\",\"alert\":\"METEOR INCOMING\",\"hazard_seq\":12,"
            + "\"hazard\":{\"active\":true,\"kind\":\"meteor\",\"center_x_m\":10.5,\"center_z_m\":-20,"
            + "\"radius_m\":30,\"eta_s\":45,\"inside\":true,\"edge_distance_m\":-5.25}}";
        Assert.IsTrue(VoiceStatusParser.TryParse(json, out var status));
        Assert.AreEqual("METEOR INCOMING", status.alert);
        Assert.AreEqual(12, status.hazard_seq);
        Assert.IsTrue(status.hazard.active);
        Assert.AreEqual("meteor", status.hazard.kind);
        Assert.AreEqual(10.5f, status.hazard.center_x_m);
        Assert.AreEqual(-20f, status.hazard.center_z_m);
        Assert.AreEqual(30f, status.hazard.radius_m);
        Assert.AreEqual(45, status.hazard.eta_s);
        Assert.IsTrue(status.hazard.inside);
        Assert.AreEqual(-5.25f, status.hazard.edge_distance_m);
    }

    [TestCase("{\"state\":\"idle\"}")]
    [TestCase("{\"state\":\"idle\",\"alert\":null,\"hazard\":null}")]
    [TestCase("{\"state\":\"idle\",\"hazard\":{\"active\":false}}")]
    public void VoiceStatus_MissingOrClearedHazardDefaultsToInactive(string json)
    {
        Assert.IsTrue(VoiceStatusParser.TryParse(json, out var status));
        Assert.IsNotNull(status.hazard);
        Assert.IsFalse(status.hazard.active);
        Assert.AreEqual("", status.alert);
        Assert.AreEqual(0, status.hazard_seq);
    }

    [Test]
    public void VoiceStatusTracker_HazardChangesOnlyWithSequenceIncludingClearAndReset()
    {
        var tracker = new VoiceStatusTracker();
        var status = new VoiceStatus { state = "idle", hazard_seq = 7,
            hazard = new HazardZoneData { active = true, kind = "meteor", radius_m = 10 } };
        Assert.IsTrue(tracker.Apply(status).HazardChanged); // Initial snapshot must render on connect.
        Assert.IsTrue(tracker.Hazard.active);
        Assert.IsFalse(tracker.Apply(status).HazardChanged);
        status.session_id = 2;
        Assert.IsFalse(tracker.Apply(status).HazardChanged);
        status.hazard = new HazardZoneData();
        status.hazard_seq = 8;
        Assert.IsTrue(tracker.Apply(status).HazardChanged);
        Assert.IsFalse(tracker.Hazard.active);
        Assert.IsFalse(tracker.Apply(status).HazardChanged);
        status.hazard_seq = 0; // Router restart is a change, not only increasing sequences.
        Assert.IsTrue(tracker.Apply(status).HazardChanged);
    }

    [Test]
    public void VoiceSessionClient_EmitsHazardEventOncePerSequence()
    {
        var root = new GameObject("Hazard event test");
        root.SetActive(false); // Never start network coroutines in EditMode.
        try
        {
            var client = root.AddComponent<VoiceSessionClient>();
            int calls = 0;
            HazardZoneData received = null;
            client.OnHazardChanged += value => { calls++; received = value; };
            var apply = typeof(VoiceSessionClient).GetMethod("Apply",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
            var status = new VoiceStatus { state = "idle", hazard_seq = 1,
                hazard = new HazardZoneData { active = true } };
            apply.Invoke(client, new object[] { status });
            apply.Invoke(client, new object[] { status });
            Assert.AreEqual(1, calls);
            Assert.IsTrue(received.active);
            status.hazard_seq = 2;
            status.hazard = new HazardZoneData();
            apply.Invoke(client, new object[] { status });
            Assert.AreEqual(2, calls);
            Assert.IsFalse(received.active);
        }
        finally { Object.DestroyImmediate(root); }
    }

    [TestCase(10, 20, -5, true)]
    [TestCase(13, 24, 0, true)]
    [TestCase(16, 28, 5, false)]
    public void Geometry_UsesSignedXZClearanceRegardlessOfHeight(float x, float z, float edge, bool inside)
    {
        foreach (float height in new[] { -100f, 0f, 100f })
        {
            var position = new Vector3(x, height, z);
            Assert.AreEqual(edge, HazardZoneGeometry.EdgeDistance(position, 10, 20, 5), 0.0001f);
            Assert.AreEqual(inside, HazardZoneGeometry.IsInside(position, 10, 20, 5));
        }
    }
}
