using System;
using System.IO;
using System.Text;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// Send requested captures and scene target facts to the voice server.
// F7 saves a local capture for desktop testing.
public class NovaVision : MonoBehaviour
{
    [SerializeField, Min(0)] float highlightSeconds = 3;
    [SerializeField] bool debugKey = true;

    VoiceSessionClient voice;
    NovaViewCapture capture;
    GameObject highlight;
    float highlightUntil;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Install()
    {
        var client = FindAnyObjectByType<VoiceSessionClient>(FindObjectsInactive.Include);
        if (client != null && client.GetComponent<NovaVision>() == null) client.gameObject.AddComponent<NovaVision>();
    }

    void Awake()
    {
        voice = GetComponent<VoiceSessionClient>();
        capture = GetComponent<NovaViewCapture>() ?? gameObject.AddComponent<NovaViewCapture>();
    }

    void OnEnable()
    {
        if (voice != null) voice.OnVisionRequest.AddListener(OnVisionRequest);
    }

    void OnDisable()
    {
        if (voice != null) voice.OnVisionRequest.RemoveListener(OnVisionRequest);
    }

    void Update()
    {
#if ENABLE_INPUT_SYSTEM
        if (debugKey && Keyboard.current != null && Keyboard.current.f7Key.wasPressedThisFrame) DebugCapture();
#endif
        if (highlight != null && Time.unscaledTime > highlightUntil) Destroy(highlight);
    }

    void OnVisionRequest(string mode)
    {
        int seq = voice.VisionSeq;
        var pointing = Resolve(mode);
        var focus = Focus(mode, pointing, out float focusFov);
        capture.Capture(result =>
        {
            if (result.error != null)
            {
                Debug.LogWarning($"[Vision] capture failed: {result.error}");
                voice.SendVisionFrame($"{{\"seq\":{seq},\"error\":\"{Escape(result.error)}\"}}");
                return;
            }
            voice.SendVisionFrame(FrameJson(seq, mode, result, pointing));
            Debug.Log($"[Vision] sent {result.width}x{result.height} JPEG ({result.jpeg.Length / 1024} KB)" +
                      (result.focusJpeg != null ? $" + {result.focusSize}px focus at {result.focusFov:0} deg ({result.focusJpeg.Length / 1024} KB)" : "") +
                      $", mode {mode}, target " + (pointing.target != null ? pointing.target.objectId : pointing.reason ?? "none"));
        }, focus, focusFov);
        ShowHighlight(pointing);   // after the render, so it is never in the image
    }

    // Use the hand ray for pointing and the view centre for looking.
    static NovaPointingTarget.Pointing Resolve(string mode)
    {
        var head = Camera.main;
        if (mode == "pointing") return NovaPointingTarget.Resolve(head);
        var gaze = new NovaPointingTarget.Pointing { hand = "gaze" };
        if (head != null) gaze.target = NovaPointingTarget.ResolveRay(new Ray(head.transform.position, head.transform.forward), head.transform.position);
        return gaze;
    }

    // Frame about 2.5 meters around the selected target.
    static Vector3? Focus(string mode, NovaPointingTarget.Pointing pointing, out float fov)
    {
        fov = 35;
        var head = Camera.main;
        if (head == null || pointing.target == null) return null;
        float distance = Mathf.Max(0.3f, Vector3.Distance(head.transform.position, pointing.target.point));
        fov = Mathf.Clamp(2 * Mathf.Atan(1.25f / distance) * Mathf.Rad2Deg, 12, 35);
        return mode == "pointing" ? pointing.target.point : head.transform.position + head.transform.forward * distance;
    }

    string FrameJson(int seq, string mode, NovaViewCapture.Result result, NovaPointingTarget.Pointing pointing)
    {
        var sb = new StringBuilder(result.jpeg.Length * 4 / 3 + 512);
        sb.Append("{\"seq\":").Append(seq).Append(",\"mode\":\"").Append(Escape(mode)).Append('"');
        sb.Append(",\"width\":").Append(result.width).Append(",\"height\":").Append(result.height);
        sb.Append(",\"image_jpeg_base64\":\"").Append(Convert.ToBase64String(result.jpeg)).Append('"');
        if (result.focusJpeg != null)
            sb.Append(",\"focus_jpeg_base64\":\"").Append(Convert.ToBase64String(result.focusJpeg)).Append('"')
              .Append(",\"focus_size\":").Append(result.focusSize)
              .Append(",\"focus_fov\":").Append(result.focusFov.ToString("0.#", System.Globalization.CultureInfo.InvariantCulture));
        var target = pointing.target;
        sb.Append(",\"has_target\":").Append(target != null ? "true" : "false");
        if (target != null)
            sb.Append(",\"target\":").Append(NovaPointingTarget.ToJson(target, pointing.hand, capture.ViewportPoint(target.point)));
        sb.Append(",\"pointing\":{");
        if (pointing.hand != null) sb.Append("\"hand\":\"").Append(Escape(pointing.hand)).Append("\",");
        sb.Append("\"reason\":\"").Append(Escape(pointing.reason ?? "")).Append("\"}}");
        return sb.ToString();
    }

    void ShowHighlight(NovaPointingTarget.Pointing pointing)
    {
        if (highlight != null) Destroy(highlight);
        if (pointing.target == null || highlightSeconds <= 0) return;
        var shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) return;
        highlight = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        highlight.name = "Nova Vision Target";
        highlight.layer = 5;   // UI: never raycast against or captured
        Destroy(highlight.GetComponent<Collider>());
        var material = new Material(shader);
        material.SetColor("_BaseColor", new Color(0.35f, 0.9f, 1));
        highlight.GetComponent<Renderer>().sharedMaterial = material;
        highlight.transform.position = pointing.target.point;
        highlight.transform.localScale = Vector3.one * Mathf.Clamp(pointing.target.distance * 0.012f, 0.04f, 1.5f);
        highlightUntil = Time.unscaledTime + highlightSeconds;
    }

    void DebugCapture()
    {
        var pointing = Resolve("pointing");
        var focus = Focus("pointing", pointing, out float focusFov);
        capture.Capture(result =>
        {
            if (result.error != null) { Debug.LogWarning($"[Vision] debug capture failed: {result.error}"); return; }
            string path = Path.Combine(Application.persistentDataPath, $"nova_view_{DateTime.Now:yyyyMMdd_HHmmss}.jpg");
            File.WriteAllBytes(path, result.jpeg);
            if (result.focusJpeg != null) File.WriteAllBytes(Path.ChangeExtension(path, null) + "_focus.jpg", result.focusJpeg);
            string target = pointing.target != null
                ? NovaPointingTarget.ToJson(pointing.target, pointing.hand, capture.ViewportPoint(pointing.target.point))
                : pointing.reason ?? "none";
            Debug.Log($"[Vision] debug capture {result.width}x{result.height} ({result.jpeg.Length / 1024} KB)" +
                      (result.focusJpeg != null ? $" + focus {result.focusSize}px at {result.focusFov:0} deg" : "") + $" saved to {path}; target {target}");
        }, focus, focusFov);
        ShowHighlight(pointing);
    }

    void OnDestroy()
    {
        if (highlight != null) Destroy(highlight);
    }

    static string Escape(string text) => (text ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", " ").Replace("\r", " ");
}
