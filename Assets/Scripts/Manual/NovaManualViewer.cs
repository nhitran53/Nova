using System.Collections.Generic;
using TMPro;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Hands;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// Shows Resources/NovaManual pages requested by the voice server.
// Pinch the HUD preview to open. Drag to scroll and use two fingers to zoom.
// On desktop use L to open or close and Ctrl + mouse wheel to zoom.
public class NovaManualViewer : MonoBehaviour
{
    [Tooltip("HUD position of the mini view, left of the vitals and minimap column.")]
    [SerializeField] Vector2 miniSlot = new Vector2(-650, 30);
    [SerializeField] float miniWidth = 170;
    [SerializeField, Min(0.2f)] float openDistance = 0.45f;
    [SerializeField, Min(0.0001f)] float pageScale = 0.0006f;   // metres per canvas unit (800 units = 0.48 m)
    [SerializeField, Min(1)] float startZoom = 1.3f;
    [SerializeField, Min(1)] float maxZoom = 4;
    [SerializeField] bool keyboardShortcut = true;

    [Header("Touch")]
    [Tooltip("Fingertip within this many metres of the page counts as touching it (drag, zoom).")]
    [SerializeField, Min(0.005f)] float touchDepth = 0.02f;
    [Tooltip("Press depth in metres for a button poke.")]
    [SerializeField, Min(0)] float pressDistance = 0.008f;
    [Tooltip("Pullback distance in metres before another poke.")]
    [SerializeField, Min(0.005f)] float releaseDistance = 0.025f;

    const int UiLayer = 5;
    const float NearDepth = 0.1f, PinchClosed = 0.025f, PinchOpen = 0.04f;
    static readonly Vector2 PageSize = new Vector2(800, 680), ViewSize = new Vector2(760, 540), Button = new Vector2(70, 60);
    static readonly Color Cyan = new Color(0.35f, 0.9f, 1), Panel = new Color(0.02f, 0.06f, 0.08f, 0.85f),
        ButtonColor = new Color(0.08f, 0.2f, 0.25f, 0.95f), Hover = new Color(0.15f, 0.45f, 0.55f, 0.95f), CloseColor = new Color(0.35f, 0.1f, 0.1f, 0.95f);
    static readonly List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();
    static XROrigin origin;

    // Stops map and wrist pinches while a hand is on the manual.
    public static bool HandOnManual { get; private set; }

    VoiceSessionClient voice;
    Texture2D page;
    string title;
    HelmetHUD hud;
    RectTransform mini, large, view, content, titleBar, zoomIn, zoomOut, close;
    Material pageMaterial; // Draws the open page over the helmet HUD.
    float zoom;
    readonly bool[] wasPinching = new bool[2], touching = new bool[2], armed = new bool[2];
    readonly Vector2[] lastTouch = new Vector2[2];
    int mover = -1;            // hand moving the page by its title bar
    Vector3 moveOffset;
    float lastSpread = -1;     // two-finger zoom

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Install()
    {
        var client = FindAnyObjectByType<VoiceSessionClient>(FindObjectsInactive.Include);
        if (client != null && client.GetComponent<NovaManualViewer>() == null) client.gameObject.AddComponent<NovaManualViewer>();
    }

    void Awake() => voice = GetComponent<VoiceSessionClient>();

    void OnEnable()
    {
        if (voice != null) voice.OnDisplay.AddListener(Show);
    }

    void OnDisable()
    {
        if (voice != null) voice.OnDisplay.RemoveListener(Show);
        HandOnManual = false;
    }

    // Load the requested page and rebuild its preview.
    public void Show(string asset)
    {
        var texture = Resources.Load<Texture2D>("NovaManual/" + asset);
        if (texture == null) { Debug.LogWarning($"[Manual] no page Resources/NovaManual/{asset}", this); return; }
        Dismiss();
        page = texture;
        title = voice != null && !string.IsNullOrEmpty(voice.Display.title) ? voice.Display.title
            : System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(asset.Replace('_', ' '));
        BuildMini();
        Debug.Log($"[Manual] showing {title} ({page.width}x{page.height})", this);
    }

    // Hide both views until NOVA requests another page.
    public void Dismiss()
    {
        if (large != null) Destroy(large.gameObject);
        if (mini != null) Destroy(mini.gameObject);
        large = view = content = titleBar = zoomIn = zoomOut = close = mini = null;
        page = null;
        mover = -1;
        lastSpread = -1;
        for (int i = 0; i < 2; i++) touching[i] = armed[i] = false;
    }

    void Update()
    {
        HandOnManual = false;
        if (page == null) return;
        if (large == null && (mini == null || hud == null)) BuildMini();   // the active camera (and its HUD) can change
#if ENABLE_INPUT_SYSTEM
        if (keyboardShortcut && Keyboard.current != null && Keyboard.current.lKey.wasPressedThisFrame)
        {
            if (large != null) Dismiss(); else OpenLarge();
            if (page == null) return;
        }
        if (large != null && Mouse.current != null)
        {
            float wheel = Mouse.current.scroll.ReadValue().y;
            bool ctrl = Keyboard.current != null && Keyboard.current.ctrlKey.isPressed;
            if (wheel != 0) { if (ctrl) Zoom(wheel > 0 ? 1.1f : 1 / 1.1f); else Pan(new Vector2(0, -wheel * 0.5f)); }
        }
#endif
        var hands = Hands();
        if (large == null)
        {
            for (int i = 0; i < 2; i++) Grab(i, Hand(hands, i));
            return;
        }
        Move(hands);
        if (large == null) return;
        TouchPage(hands);
    }

    static XRHand? Hand(XRHandSubsystem hands, int i) => hands == null ? (XRHand?)null : i == 0 ? hands.rightHand : hands.leftHand;

    void Grab(int i, XRHand? hand)
    {
        bool over = false, pinching = Pinching(hand, out var point);
        if (mini != null && point.HasValue) over = OverMini(point.Value);
        if (over) HandOnManual = true;
        if (over && pinching && !wasPinching[i]) OpenLarge();
        wasPinching[i] = pinching;
    }

    bool OverMini(Vector3 point)
    {
        var cam = Camera.main;
        if (cam == null) return false;
        var plane = new Plane(mini.forward, mini.position);
        var ray = new Ray(cam.transform.position, point - cam.transform.position);
        if (!plane.Raycast(ray, out float enter)) return false;
        Vector2 local = mini.InverseTransformPoint(ray.GetPoint(enter));
        var rect = mini.rect;
        rect.min -= Vector2.one * 20;
        rect.max += Vector2.one * 20;
        return rect.Contains(local);
    }

    // Pinch the title bar to pick the page up; it follows the pinch and keeps facing you.
    void Move(XRHandSubsystem hands)
    {
        for (int i = 0; i < 2; i++)
        {
            bool pinching = Pinching(Hand(hands, i), out var point);
            if (mover == i)
            {
                HandOnManual = true;
                if (!pinching || !point.HasValue) { mover = -1; continue; }
                large.position = point.Value + moveOffset;
                FaceHead();
            }
            else if (mover < 0 && pinching && !wasPinching[i] && point.HasValue && Near(point.Value, titleBar, 0.06f))
            {
                mover = i;
                moveOffset = large.position - point.Value;
            }
            wasPinching[i] = pinching;
        }
    }

    // One finger scrolls. Two fingers zoom. Pokes press the buttons.
    void TouchPage(XRHandSubsystem hands)
    {
        int touches = 0;
        Vector2 first = default, second = default;
        for (int i = 0; i < 2; i++)
        {
            if (i == mover || !Tip(Hand(hands, i), out var tip)) { touching[i] = armed[i] = false; continue; }
            Vector3 local = large.InverseTransformPoint(tip);
            float depth = -local.z * pageScale;   // metres in front of the page (+) or behind it (-)
            bool inside = large.rect.Contains((Vector2)local);
            if (inside && depth < NearDepth) HandOnManual = true;
            if (Poke(i, tip, depth, inside)) return;
            bool onPage = inside && depth < touchDepth && depth > -2 * touchDepth && view.rect.Contains((Vector2)view.InverseTransformPoint(tip));
            if (onPage)
            {
                if (touches++ == 0) first = local; else second = local;
                if (touching[i] && lastSpread < 0) Pan((Vector2)local - lastTouch[i]);
            }
            touching[i] = onPage;
            lastTouch[i] = local;
        }
        if (touches == 2)
        {
            float spread = Vector2.Distance(first, second);
            if (lastSpread > 0) Zoom(spread / lastSpread);
            lastSpread = spread;
        }
        else lastSpread = -1;
        Highlight(zoomIn, false); Highlight(zoomOut, false); Highlight(close, false, true);
        for (int i = 0; i < 2; i++)
            if (Tip(Hand(hands, i), out var t))
                foreach (var b in new[] { zoomIn, zoomOut, close })
                    if (Near(t, b, 0.06f)) Highlight(b, true, b == close);
    }

    // Approach from the front and pull back before pressing again.
    bool Poke(int i, Vector3 tip, float depth, bool inside)
    {
        if (!inside || depth > releaseDistance) { armed[i] = inside && depth > releaseDistance; return false; }
        if (!armed[i] || depth > pressDistance) return false;
        armed[i] = false;
        if (Near(tip, close, 0.03f)) { Dismiss(); return true; }
        if (Near(tip, zoomIn, 0.03f)) { Zoom(1.25f); return true; }
        if (Near(tip, zoomOut, 0.03f)) { Zoom(0.8f); return true; }
        return false;
    }

    bool Near(Vector3 world, RectTransform rect, float depthMetres)
    {
        if (rect == null) return false;
        Vector3 local = rect.InverseTransformPoint(world);
        return rect.rect.Contains((Vector2)local) && Mathf.Abs(local.z * pageScale) < depthMetres;
    }

    void Pan(Vector2 delta)
    {
        if (content == null || delta == Vector2.zero) return;
        content.anchoredPosition = Clamp(content.anchoredPosition + delta);
    }

    void Zoom(float factor)
    {
        if (content == null || page == null) return;
        float next = Mathf.Clamp(zoom * factor, 1, maxZoom);
        if (Mathf.Approximately(next, zoom)) return;
        float ratio = next / zoom;
        zoom = next;
        float width = ViewSize.x * zoom;
        content.sizeDelta = new Vector2(width, width * page.height / page.width);
        content.anchoredPosition = Clamp(content.anchoredPosition * ratio);   // zoom about the centre of the view
    }

    Vector2 Clamp(Vector2 p)
    {
        var overflow = Vector2.Max(Vector2.zero, (content.sizeDelta - ViewSize) / 2);
        return new Vector2(Mathf.Clamp(p.x, -overflow.x, overflow.x), Mathf.Clamp(p.y, -overflow.y, overflow.y));
    }

    void Highlight(RectTransform button, bool hover, bool isClose = false)
    {
        if (button == null) return;
        button.GetComponent<Image>().color = hover ? Hover : isClose ? CloseColor : ButtonColor;
    }

    void FaceHead()
    {
        var cam = Camera.main;
        if (cam == null) return;
        var away = Vector3.ProjectOnPlane(large.position - cam.transform.position, Vector3.up);
        if (away.sqrMagnitude > 0.0001f) large.rotation = Quaternion.LookRotation(away.normalized, Vector3.up);
    }

    void BuildMini()
    {
        var cam = Camera.main;
        hud = cam != null ? cam.GetComponentInChildren<HelmetHUD>(true) : null;
        if (hud == null || page == null) return;   // the desktop rig has no HUD: use L there
        if (mini != null) Destroy(mini.gameObject);
        float height = miniWidth * page.height / page.width;
        mini = Rect("Nova Manual Mini", hud.transform, miniSlot, new Vector2(miniWidth + 16, height + 56));
        Fill(mini, Panel);
        var image = Rect("Page", mini, new Vector2(0, -12), new Vector2(miniWidth, height)).gameObject.AddComponent<RawImage>();
        image.texture = page;
        image.raycastTarget = false;
        CopyMaterial(image);
        var label = Text(mini, $"MANUAL / {title.ToUpperInvariant()}\nPINCH TO OPEN", new Vector2(0, height / 2 + 8), new Vector2(miniWidth + 16, 44), 15);
        label.color = Cyan;
        label.enableAutoSizing = true;   // long titles shrink instead of spilling out of the panel
        label.fontSizeMin = 9;
        label.fontSizeMax = 15;
    }

    void OpenLarge()
    {
        var cam = Camera.main;
        if (cam == null || page == null || large != null) return;
        var head = cam.transform;
        var forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        if (forward.sqrMagnitude < 0.01f) forward = head.up;
        forward.Normalize();
        var go = new GameObject("Nova Manual", typeof(RectTransform), typeof(Canvas));
        go.layer = UiLayer;
        var canvas = go.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.worldCamera = cam;
        canvas.overrideSorting = true;
        canvas.sortingOrder = 32000;
        large = go.GetComponent<RectTransform>();
        large.sizeDelta = PageSize;
        large.localScale = Vector3.one * pageScale;
        large.position = head.position + forward * openDistance - Vector3.up * 0.1f;
        large.rotation = Quaternion.LookRotation(forward, Vector3.up);
        Fill(large, Panel);
        float top = PageSize.y / 2 - Button.y / 2 - 8;
        titleBar = Rect("Title Bar", large, new Vector2(-120, top), new Vector2(PageSize.x - 280, Button.y));
        Fill(titleBar, new Color(0.05f, 0.14f, 0.17f, 0.9f));
        Text(titleBar, title + "   <size=60%>(pinch here to move)</size>", Vector2.zero, titleBar.sizeDelta, 28).color = Cyan;
        zoomOut = PokeButton("Zoom Out", "-", new Vector2(PageSize.x / 2 - 3 * Button.x - 3 * 10 + Button.x / 2, top), ButtonColor);
        zoomIn = PokeButton("Zoom In", "+", new Vector2(PageSize.x / 2 - 2 * Button.x - 2 * 10 + Button.x / 2, top), ButtonColor);
        close = PokeButton("Close", "X", new Vector2(PageSize.x / 2 - Button.x - 10 + Button.x / 2, top), CloseColor);
        view = Rect("Viewport", large, new Vector2(0, -20), ViewSize);
        view.gameObject.AddComponent<RectMask2D>();
        zoom = 1;
        content = Rect("Page", view, Vector2.zero, new Vector2(ViewSize.x, ViewSize.x * page.height / page.width));
        var image = content.gameObject.AddComponent<RawImage>();
        image.texture = page;
        image.raycastTarget = false;
        Zoom(startZoom);
        Text(large, "Drag one finger to scroll  /  two fingers to zoom  /  pinch the title to move",
             new Vector2(0, -PageSize.y / 2 + 22), new Vector2(PageSize.x, 30), 18).color = Cyan;
        DrawOverHud(large);
        if (mini != null) mini.gameObject.SetActive(false);
    }

    RectTransform PokeButton(string name, string label, Vector2 position, Color color)
    {
        var button = Rect(name, large, position, Button);
        Fill(button, color);
        Text(button, label, Vector2.zero, Button, 40);
        return button;
    }

    // Draw the page and its text after the helmet HUD.
    void DrawOverHud(RectTransform root)
    {
        var shader = Shader.Find("Nova/Visor UI");
        if (shader != null && pageMaterial == null) pageMaterial = new Material(shader) { renderQueue = 4001 };
        foreach (var graphic in root.GetComponentsInChildren<Graphic>(true))
        {
            if (graphic is TMP_Text text) text.fontMaterial.renderQueue = 4002;   // a per-text instance
            else if (pageMaterial != null) graphic.material = pageMaterial;
        }
    }

    static RectTransform Rect(string name, Transform parent, Vector2 position, Vector2 size)
    {
        var rt = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        rt.gameObject.layer = UiLayer;
        rt.SetParent(parent, false);
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = position;
        rt.sizeDelta = size;
        return rt;
    }

    void Fill(RectTransform rt, Color color)
    {
        var image = rt.gameObject.AddComponent<Image>();
        image.color = color;
        image.raycastTarget = false;
        CopyMaterial(image);
    }

    // Use the existing visor material for the HUD preview.
    void CopyMaterial(Graphic graphic)
    {
        if (hud == null || !graphic.transform.IsChildOf(hud.transform)) return;
        var style = hud.GetComponentInChildren<Image>(true);
        if (style != null && style != graphic) graphic.material = style.material;
    }

    TMP_Text Text(RectTransform parent, string text, Vector2 position, Vector2 size, float fontSize)
    {
        var tmp = Rect("Text", parent, position, size).gameObject.AddComponent<TextMeshProUGUI>();
        var style = hud != null ? hud.GetComponentInChildren<TMP_Text>(true) : null;
        if (style != null)
        {
            tmp.font = style.font;
            if (parent.IsChildOf(hud.transform)) tmp.fontSharedMaterial = style.fontSharedMaterial;
        }
        tmp.text = text;
        tmp.fontSize = fontSize;
        tmp.alignment = TextAlignmentOptions.Center;
        tmp.raycastTarget = false;
        return tmp;
    }

    static XRHandSubsystem Hands()
    {
        SubsystemManager.GetSubsystems(subsystems);
        foreach (var s in subsystems) if (s.running) return s;
        return null;
    }

    static bool Tip(XRHand? hand, out Vector3 world)
    {
        world = default;
        return hand.HasValue && TryJoint(hand.Value, XRHandJointID.IndexTip, out world);
    }

    // Separate open and closed thresholds keep the pinch steady.
    bool Pinching(XRHand? hand, out Vector3? point)
    {
        point = null;
        if (!hand.HasValue || !TryJoint(hand.Value, XRHandJointID.IndexTip, out var index) ||
            !TryJoint(hand.Value, XRHandJointID.ThumbTip, out var thumb)) return false;
        point = (index + thumb) / 2;
        float gap = Vector3.Distance(index, thumb);
        int i = hand.Value.handedness == Handedness.Right ? 0 : 1;
        return wasPinching[i] ? gap < PinchOpen : gap < PinchClosed;
    }

    static bool TryJoint(XRHand hand, XRHandJointID id, out Vector3 world)
    {
        world = default;
        if (!hand.isTracked || !hand.GetJoint(id).TryGetPose(out Pose pose)) return false;
        if (origin == null || !origin.isActiveAndEnabled) origin = FindAnyObjectByType<XROrigin>();
        var space = origin != null && origin.CameraFloorOffsetObject != null ? origin.CameraFloorOffsetObject.transform : null;
        world = space != null ? space.TransformPoint(pose.position) : pose.position;
        return true;
    }

    void OnDestroy()
    {
        HandOnManual = false;
        Dismiss();
        if (pageMaterial != null) Destroy(pageMaterial);
    }
}
