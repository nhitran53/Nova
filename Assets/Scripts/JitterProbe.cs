using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR;

// Temporary: logs how much the rig, camera, HUD and wrist panel move each frame.
// Stand still in the headset and read the [Jitter] lines in the Console. Delete when done.
public class JitterProbe : MonoBehaviour
{
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Spawn() => DontDestroyOnLoad(new GameObject("Jitter Probe").AddComponent<JitterProbe>());

    class Track
    {
        public string name;
        public bool has;
        public Vector3 pos;
        public Quaternion rot;
        public float sumMove, maxMove, sumTurn, maxTurn;

        public void Sample(Vector3 p, Quaternion r)
        {
            if (has)
            {
                float move = (p - pos).magnitude * 1000;
                float turn = Quaternion.Angle(rot, r);
                sumMove += move; maxMove = Mathf.Max(maxMove, move);
                sumTurn += turn; maxTurn = Mathf.Max(maxTurn, turn);
            }
            pos = p; rot = r; has = true;
        }

        public string Report(int frames)
        {
            string text = $"{name} {sumMove / frames:0.00}/{maxMove:0.00}mm {sumTurn / frames:0.000}/{maxTurn:0.000}deg";
            sumMove = maxMove = sumTurn = maxTurn = 0;
            return text;
        }
    }

    readonly Track origin = new Track { name = "origin" };
    readonly Track head = new Track { name = "head(local)" };
    readonly Track hud = new Track { name = "hud-vs-head" };
    readonly Track wrist = new Track { name = "wrist(local)" };

    readonly System.Collections.Generic.List<XRDisplaySubsystem> displays = new();
    XROrigin xrOrigin;
    Transform helmet, panel;
    float refresh = 72;
    int frames, slowFrames;
    float windowStart, lastFrame, maxFrameMs, nextFind;

    void OnEnable() => Application.onBeforeRender += Sample;
    void OnDisable() => Application.onBeforeRender -= Sample;

    void Sample()
    {
        float now = Time.realtimeSinceStartup;
        if (now >= nextFind && (xrOrigin == null || !xrOrigin.isActiveAndEnabled || helmet == null || panel == null))
        {
            nextFind = now + 1;
            Find();
        }
        if (xrOrigin == null || !xrOrigin.isActiveAndEnabled)
            return;

        var camera = xrOrigin.Camera.transform;
        origin.Sample(xrOrigin.transform.position, xrOrigin.transform.rotation);
        head.Sample(camera.localPosition, camera.localRotation);
        if (helmet != null)
            hud.Sample(camera.InverseTransformPoint(helmet.position), Quaternion.Inverse(camera.rotation) * helmet.rotation);
        if (panel != null && panel.gameObject.activeInHierarchy)
            wrist.Sample(panel.localPosition, panel.localRotation);
        else
            wrist.has = false;

        float frameMs = lastFrame > 0 ? (now - lastFrame) * 1000 : 0;
        lastFrame = now;
        if (frameMs > 1500 / refresh) slowFrames++;
        maxFrameMs = Mathf.Max(maxFrameMs, frameMs);
        frames++;

        if (now - windowStart < 1)
            return;
        Debug.Log($"[Jitter] {frames / (now - windowStart):0} fps (headset {refresh:0} Hz), slow frames {slowFrames}, worst {maxFrameMs:0.0}ms | avg/max per frame: " +
            $"{origin.Report(frames)} | {head.Report(frames)} | {hud.Report(frames)} | {wrist.Report(frames)}");
        frames = slowFrames = 0;
        maxFrameMs = 0;
        windowStart = now;
    }

    void Find()
    {
        xrOrigin = FindFirstObjectByType<XROrigin>();
        var hudComponent = FindFirstObjectByType<HelmetHUD>();
        helmet = hudComponent != null ? hudComponent.transform : null;
        var display = GameObject.Find("Wrist Display");
        if (display != null) panel = display.transform;
        SubsystemManager.GetSubsystems(displays);
        if (displays.Count > 0 && displays[0].TryGetDisplayRefreshRate(out float rate) && rate > 0)
            refresh = rate;
    }
}
