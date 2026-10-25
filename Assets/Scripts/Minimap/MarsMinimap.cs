using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.UI;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif
using Debug = UnityEngine.Debug;

// Add with Tools > Nova > Minimap > Add Minimap To Scene.
// Uses the helmet HUD on Quest and a screen overlay on desktop.
// Terrain heights provide the relief map. Markers show locations and distant pins stay at the rim.
// The center arrow shows the player. The map can follow the view or keep north up.
public class MarsMinimap : MonoBehaviour
{
    [Header("Astronaut")]
    [Tooltip("The eye/head. Leave empty: it follows whichever camera is active (desktop or Quest rig).")]
    public Transform head;

    [Header("Docking into the Helmet HUD (Quest rig)")]
    [Tooltip("If the active camera has a Helmet HUD, put the minimap inside it.")]
    public bool dockIntoHelmetHud = true;
    [Tooltip("Center of the minimap in Helmet HUD canvas units. Keep clear of the center alerts and AI orb.")]
    public Vector2 hudSlot = new Vector2(430f, -245f);
    [Tooltip("Diameter of the minimap in Helmet HUD canvas units.")]
    public float hudDiameter = 200f;

    [Header("Desktop rig (no Helmet HUD): drawn flat on the screen")]
    [Tooltip("Map diameter in pixels for a 1080p-tall window (scales with the window).")]
    [Min(50f)] public float screenDiameter = 360f;
    [Tooltip("Gap from the bottom-left corner of the screen, in pixels for a 1080p-tall window.")]
    public Vector2 screenMargin = new Vector2(40f, 60f);

    [Header("Map")]
    [Tooltip("How many meters the map shows edge to edge at each zoom step.")]
    public float[] zoomLevelsMeters = { 300f, 1000f, 4000f };
    public int zoomIndex = 1;
    [Tooltip("On: the map turns with you (forward = up). Off: north is always up.")]
    public bool headingUp = true;
    [Range(0.2f, 1f)] public float opacity = 0.92f;

    [Header("Terrain picture")]
    [Tooltip("Pixels per side of the relief texture. It covers twice the visible width so it can scroll.")]
    public int textureSize = 512;
    [Tooltip("Makes hills look steeper in the shading so small bumps are visible.")]
    public float verticalExaggeration = 4f;
    public bool contourLines = true;
    public bool gridLines = true;
    [Tooltip("The Jezero importer stores missing DEM data as holes at height 0. Draw those as 'no data'.")]
    public bool zeroHeightIsNoData = true;
    [Tooltip("Max milliseconds per frame spent drawing the relief, so the headset never stutters.")]
    public float buildBudgetMs = 2.5f;

    [Header("Desktop testing keys")]
    [Tooltip("M = show/hide, [ and ] = zoom, N = heading-up / north-up.")]
    public bool keyboardShortcuts = true;

    // Values read by other scripts
    public Vector2 PlayerXZ { get; private set; }
    public float HeadingDeg { get; private set; }   // 0..360, 0 = north (+z), 90 = east (+x)
    public float ViewMeters => zoomLevelsMeters[Mathf.Clamp(zoomIndex, 0, zoomLevelsMeters.Length - 1)];
    public bool Visible => canvasGO != null && canvasGO.activeSelf;
    public bool DockedInHelmetHud { get; private set; }
    public bool Holographic { get; private set; }
    // Used by MinimapNavigationPanel: the picked destination, and where things are drawn.
    public string SelectedCode { get; set; }
    public RectTransform Content => content;
    public Material UiMaterial => uiMaterial;
    public RectTransform MarkerRect(string code) => views.TryGetValue(code, out var v) ? v.root : null;

    const float PX = 512f;      // the map circle is 512 units wide inside `content`, then scaled to fit
    const float IconPx = 26f;

    // UI
    int uiLayer;
    Material uiMaterial;   // null = Unity's default UI material; the visor material when docked
    Transform attachedTo;
    Font font;
    Sprite discSprite, ringSprite, badgeSprite, arrowSprite;
    GameObject canvasGO;
    RectTransform content;   // holds everything; this is what gets placed and scaled
    CanvasScaler scaler;
    CanvasGroup presentationGroup;
    Coroutine transition;
    Vector3 presentationScale = Vector3.one;
    RectTransform rotator, zoneLayer, routeLayer, markerLayer, arrow, north;
    RawImage terrainImage;
    Image frame;
    Text infoText;
    float highlightUntil;
    float shownOpacity = -1f;
    float nextInfo;

    // Terrain
    struct TerrainInfo { public TerrainData data; public float x, y, z, sx, sy, sz; }
    TerrainInfo[] terrains = Array.Empty<TerrainInfo>();
    float minY, maxY;
    float nextTerrainScan;
    bool terrainsChanged;
    Texture2D front, back;
    Vector2 frontCenter;
    float frontCover;
    bool hasFront;
    Coroutine building;
    Color32[] pixels;
    float[] heights;
    bool[] valid;

    // Markers
    class MarkerView
    {
        public RectTransform root;
        public Image icon, zone;
        public Text label;
        public bool used;
    }
    readonly Dictionary<string, MarkerView> views = new Dictionary<string, MarkerView>();
    readonly Dictionary<string, MapPoint> externalPins = new Dictionary<string, MapPoint>();
    readonly List<MapPoint> points = new List<MapPoint>();
    readonly HashSet<string> seenCodes = new HashSet<string>();
    readonly List<string> unusedCodes = new List<string>();

    // Active navigation route (same points as the trail on the ground)
    NavigationRoute drawnRoute;
    readonly List<Vector2> routeXZ = new List<Vector2>();
    readonly List<Image> routeSegments = new List<Image>();
    readonly List<float> routeAlong = new List<float>();
    static readonly Color RouteColor = new Color(0.25f, 0.9f, 1f, 0.95f);
    static readonly Color RoutePreviewColor = new Color(0.25f, 0.9f, 1f, 0.5f);
    static readonly Color RouteWalkedColor = new Color(0.7f, 0.7f, 0.7f, 0.35f);

    static readonly Color32 NoDataColor = new Color32(22, 20, 24, 255);
    static readonly Color[] Ramp = {
        new Color(0.20f, 0.09f, 0.06f),   // low: dark rust
        new Color(0.45f, 0.20f, 0.10f),
        new Color(0.72f, 0.40f, 0.20f),
        new Color(0.88f, 0.62f, 0.38f),
        new Color(0.97f, 0.86f, 0.70f),   // high: pale butterscotch
    };
    static readonly Vector3 SunDir = new Vector3(-0.5f, 0.5f, 0.7071f);   // from the north-west, 45 degrees up

    // Map controls for other scripts
    public void Show(bool on) { if (canvasGO != null) canvasGO.SetActive(on); }
    public void ZoomIn() => SetZoomIndex(zoomIndex - 1);
    public void ZoomOut() => SetZoomIndex(zoomIndex + 1);
    public void SetZoomIndex(int index) => zoomIndex = Mathf.Clamp(index, 0, zoomLevelsMeters.Length - 1);
    public void SetHeadingUp(bool on) => headingUp = on;

    // Briefly pulse the map rim.
    public void Highlight(float seconds = 4f)
    {
        Show(true);
        highlightUntil = Time.unscaledTime + seconds;
    }

    // Add pins from external data.
    // A scene MapMarker with the same code always wins.
    public void SetExternalPins(IEnumerable<MapPoint> pins)
    {
        externalPins.Clear();
        if (pins == null) return;
        foreach (var p in pins)
            if (p != null && !string.IsNullOrEmpty(p.code))
                externalPins[p.code] = p;
    }

    // Unity lifecycle
    void Start()
    {
        if (zoomLevelsMeters == null || zoomLevelsMeters.Length == 0) zoomLevelsMeters = new[] { 1000f };
        SetZoomIndex(zoomIndex);
        textureSize = Mathf.Clamp(textureSize, 64, 2048);
        BuildHud();
        ScanTerrains();
    }

    void OnEnable()
    {
        if (canvasGO != null) canvasGO.SetActive(true);
        FinishTransition();
    }

    void OnDisable()
    {
        if (canvasGO != null) canvasGO.SetActive(false);
        StopAllCoroutines();
        building = null;
        FinishTransition();
    }

    void OnDestroy()
    {
        if (canvasGO != null) Destroy(canvasGO);
        if (front != null) Destroy(front);
        if (back != null) Destroy(back);
    }

    void LateUpdate()   // after Mars Explorer has moved this frame
    {
        if (canvasGO == null) return;

        // Follow the camera selected by ExplorerRigs.
        if (head == null || !head.gameObject.activeInHierarchy)
            head = Camera.main != null ? Camera.main.transform : null;
        if (head == null) return;
        if (attachedTo != head) Attach();
        if (!DockedInHelmetHud && !Holographic) ApplyScreenLayout();
        HandleKeys();

        // Update the player position even when hidden because the API reads it.
        Vector3 p = head.position;
        PlayerXZ = new Vector2(p.x, p.z);
        HeadingDeg = ComputeHeading(head);

        if (!canvasGO.activeSelf) return;

        if (terrains.Length == 0 && Time.unscaledTime >= nextTerrainScan) ScanTerrains();
        UpdateTerrainTexture();

        float rot = headingUp ? HeadingDeg : 0f;
        rotator.localRotation = Quaternion.Euler(0f, 0f, rot);
        arrow.localRotation = Quaternion.Euler(0f, 0f, headingUp ? 0f : -HeadingDeg);
        float a = rot * Mathf.Deg2Rad;
        north.anchoredPosition = new Vector2(-Mathf.Sin(a), Mathf.Cos(a)) * (PX * 0.5f + 24f);

        UpdateRoute();
        UpdateMarkers(rot);
        UpdateFrame();
        if (Time.unscaledTime >= nextInfo)
        {
            nextInfo = Time.unscaledTime + 0.25f;
            UpdateInfoText();
        }
    }

    // Dock into the Helmet HUD if this camera has one, otherwise float in front of the eye.
    void Attach()
    {
        Holographic = false;
        attachedTo = head;
        var root = (RectTransform)canvasGO.transform;
        var canvas = canvasGO.GetComponent<Canvas>();
        canvas.worldCamera = head.GetComponent<Camera>();
        HelmetHUD hud = dockIntoHelmetHud ? head.GetComponentInChildren<HelmetHUD>(true) : null;
        if (hud != null)
        {
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.overrideSorting = false;
            // Use the HUD world-space canvas when docked.
            root.SetParent(hud.transform, false);       // HelmetHUD sits on the HUD's Canvas
            root.anchorMin = Vector2.zero;
            root.anchorMax = Vector2.one;
            root.offsetMin = root.offsetMax = Vector2.zero;
            root.anchoredPosition3D = Vector3.zero;
            root.localRotation = Quaternion.identity;
            root.localScale = Vector3.one;
            content.anchorMin = content.anchorMax = new Vector2(0.5f, 0.5f);
            content.anchoredPosition = hudSlot;
            content.localScale = Vector3.one * (hudDiameter / PX);
            SetMaterial(FindVisorMaterial(hud));
            DockedInHelmetHud = true;
        }
        else
        {
            // Use a desktop screen overlay to keep the round map from stretching at the view edge.
            root.SetParent(transform, false);
            root.localScale = Vector3.one;
            root.localRotation = Quaternion.identity;
            canvasGO.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
            SetMaterial(null);
            DockedInHelmetHud = false;
            ApplyScreenLayout();
        }
        presentationScale = content.localScale;
        FinishTransition();
        Debug.Log(DockedInHelmetHud ? $"[Minimap] Docked into Helmet HUD on '{head.name}'." : $"[Minimap] Drawn on screen (desktop rig, camera '{head.name}').");
    }

    public bool PresentOnHand(Transform anchor, float diameterMeters)
    {
        if (canvasGO == null || !DockedInHelmetHud || anchor == null || !isActiveAndEnabled) return false;
        FinishTransition();
        var root = (RectTransform)canvasGO.transform;
        root.SetParent(anchor, false);
        var canvas = canvasGO.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.WorldSpace;
        canvas.overrideSorting = false;
        root.anchorMin = root.anchorMax = root.pivot = new Vector2(.5f, .5f);
        root.sizeDelta = content.sizeDelta;
        root.anchoredPosition3D = Vector3.zero;
        root.localRotation = Quaternion.identity;
        root.localScale = Vector3.one * (Mathf.Clamp(diameterMeters, .18f, .5f) / PX);
        content.anchorMin = content.anchorMax = new Vector2(.5f, .5f);
        content.anchoredPosition3D = Vector3.zero;
        presentationScale = Vector3.one;
        content.localScale = presentationScale;
        SetMaterial(null); // World-space depth-tested UI; critical visor alerts render above it.
        DockedInHelmetHud = false;
        Holographic = true;
        Show(true);
        transition = StartCoroutine(AnimatePresentation());
        return true;
    }

    public void ReturnToHud()
    {
        if (!Holographic || canvasGO == null) return;
        if (head == null)
        {
            canvasGO.transform.SetParent(transform, false);
            Holographic = DockedInHelmetHud = false;
            attachedTo = null;
            return;
        }
        Attach();
        if (isActiveAndEnabled) transition = StartCoroutine(AnimatePresentation());
    }

    IEnumerator AnimatePresentation()
    {
        float start = Time.unscaledTime;
        while (Time.unscaledTime - start < .16f)
        {
            float t = Mathf.SmoothStep(0, 1, (Time.unscaledTime - start) / .16f);
            presentationGroup.alpha = Mathf.Lerp(.35f, 1, t);
            content.localScale = presentationScale * Mathf.Lerp(.85f, 1, t);
            yield return null;
        }
        presentationGroup.alpha = 1;
        content.localScale = presentationScale;
        transition = null;
    }

    void FinishTransition()
    {
        if (transition != null) StopCoroutine(transition);
        transition = null;
        if (presentationGroup != null) presentationGroup.alpha = 1;
        if (content != null) content.localScale = presentationScale;
    }

    // Reuse the visor material to draw over terrain and support the round mask.
    static Material FindVisorMaterial(HelmetHUD hud)
    {
        foreach (var g in hud.GetComponentsInChildren<Graphic>(true))
        {
            var m = g.material;
            if (m != null && m.shader != null && m.shader.name == "Nova/Visor UI") return m;
        }
        return null;
    }

    void SetMaterial(Material m)
    {
        uiMaterial = m;
        foreach (var g in canvasGO.GetComponentsInChildren<Graphic>(true)) g.material = m;
    }

    // Update desktop placement each frame so Inspector changes appear immediately.
    // Scale through CanvasScaler to keep text sharp.
    void ApplyScreenLayout()
    {
        float s = screenDiameter / PX;   // 1 canvas unit = s pixels on a 1080p-tall window
        scaler.referenceResolution = new Vector2(1920f / s, 1080f / s);
        content.anchorMin = content.anchorMax = Vector2.zero;
        content.localScale = Vector3.one;
        content.anchoredPosition = screenMargin / s + content.sizeDelta * 0.5f;
    }

    // Read heading from the horizontal viewing direction.
    static float ComputeHeading(Transform t)
    {
        Vector3 f = t.forward;
        if (f.x * f.x + f.z * f.z < 0.01f)     // looking straight down/up: use the top of the head instead
            f = f.y < 0f ? t.up : -t.up;
        float h = Mathf.Atan2(f.x, f.z) * Mathf.Rad2Deg;
        if (h < 0f) h += 360f;
        if (h >= 360f) h = 0f;                 // database CHECK is heading_deg < 360
        return h;
    }

    void HandleKeys()
    {
#if ENABLE_INPUT_SYSTEM
        if (!keyboardShortcuts) return;
        var kb = Keyboard.current;
        if (kb == null) return;
        if (kb.mKey.wasPressedThisFrame) Show(!Visible);
        if (kb.nKey.wasPressedThisFrame) headingUp = !headingUp;
        if (kb.rightBracketKey.wasPressedThisFrame || kb.equalsKey.wasPressedThisFrame) ZoomIn();
        if (kb.leftBracketKey.wasPressedThisFrame || kb.minusKey.wasPressedThisFrame) ZoomOut();
#endif
    }

    // Build the HUD at runtime
    void BuildHud()
    {
        uiLayer = LayerMask.NameToLayer("UI");
        if (uiLayer < 0) uiLayer = 0;
        font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        discSprite = MinimapSprites.Disc(256);
        ringSprite = MinimapSprites.Ring(256, 0.05f);
        badgeSprite = MinimapSprites.Badge(64);
        arrowSprite = MinimapSprites.Arrow(64);

        canvasGO = new GameObject("Minimap Canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler));
        canvasGO.layer = uiLayer;
        var canvas = canvasGO.GetComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 10;
        // Scale with the window on desktop. The helmet HUD handles its own scale.
        scaler = canvasGO.GetComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920f, 1080f);
        scaler.matchWidthOrHeight = 1f;
        canvasGO.transform.SetParent(transform, false);
        canvasGO.SetActive(enabled);

        var root = NewRect("Content", canvasGO.transform, new Vector2(PX + 80f, PX + 120f));
        content = root;
        presentationGroup = root.gameObject.AddComponent<CanvasGroup>();
        presentationGroup.interactable = presentationGroup.blocksRaycasts = false;

        frame = NewImage("Rim", root, ringSprite, PX + 18f);

        // Use a dark circle as the background and clip the map to it.
        var mask = NewImage("Mask", root, discSprite, PX);
        mask.color = new Color(0.05f, 0.04f, 0.04f, 0.85f);
        mask.gameObject.AddComponent<Mask>().showMaskGraphic = true;

        rotator = NewRect("Rotator", mask.rectTransform, new Vector2(PX, PX));

        var terrainRt = NewRect("Terrain", rotator, new Vector2(PX, PX));
        terrainImage = terrainRt.gameObject.AddComponent<RawImage>();
        terrainImage.raycastTarget = false;
        terrainImage.material = uiMaterial;
        terrainImage.enabled = false;

        zoneLayer = NewRect("Zones", rotator, new Vector2(PX, PX));
        routeLayer = NewRect("Route", rotator, new Vector2(PX, PX));
        markerLayer = NewRect("Markers", rotator, new Vector2(PX, PX));

        var arrowImg = NewImage("You", root, arrowSprite, 38f);
        var arrowOutline = arrowImg.gameObject.AddComponent<Outline>();
        arrowOutline.effectColor = new Color(0f, 0f, 0f, 0.9f);
        arrowOutline.effectDistance = new Vector2(2f, -2f);
        arrow = arrowImg.rectTransform;

        var northText = NewText("North", root, 36, new Vector2(40f, 40f));
        northText.text = "N";
        northText.fontStyle = FontStyle.Bold;
        northText.color = new Color(1f, 0.45f, 0.35f);
        north = northText.rectTransform;

        infoText = NewText("Info", root, 30, new Vector2(PX + 80f, 36f));
        infoText.rectTransform.anchoredPosition = new Vector2(0f, -(PX * 0.5f + 46f));
    }

    RectTransform NewRect(string name, Transform parent, Vector2 size)
    {
        var go = new GameObject(name, typeof(RectTransform));
        go.layer = uiLayer;
        var rt = (RectTransform)go.transform;
        rt.SetParent(parent, false);
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);
        rt.sizeDelta = size;
        rt.anchoredPosition = Vector2.zero;
        return rt;
    }

    Image NewImage(string name, Transform parent, Sprite sprite, float size)
    {
        var img = NewRect(name, parent, new Vector2(size, size)).gameObject.AddComponent<Image>();
        img.sprite = sprite;
        img.raycastTarget = false;
        img.material = uiMaterial;
        return img;
    }

    Text NewText(string name, Transform parent, int fontSize, Vector2 box)
    {
        var t = NewRect(name, parent, box).gameObject.AddComponent<Text>();
        t.font = font;
        t.fontSize = fontSize;
        t.alignment = TextAnchor.MiddleCenter;
        t.horizontalOverflow = HorizontalWrapMode.Overflow;
        t.verticalOverflow = VerticalWrapMode.Overflow;
        t.color = Color.white;
        t.raycastTarget = false;
        t.material = uiMaterial;
        var outline = t.gameObject.AddComponent<Outline>();
        outline.effectColor = new Color(0f, 0f, 0f, 0.85f);
        outline.effectDistance = new Vector2(1.5f, -1.5f);
        return t;
    }

    void UpdateFrame()
    {
        Color normal = new Color(0.35f, 0.9f, 1f, 0.75f);
        if (Time.unscaledTime < highlightUntil)
        {
            float pulse = 0.5f + 0.5f * Mathf.Sin(Time.unscaledTime * 8f);
            frame.color = Color.Lerp(normal, new Color(0.3f, 0.95f, 1f, 1f), pulse);
        }
        else if (frame.color != normal)
        {
            frame.color = normal;
        }

        if (!Mathf.Approximately(shownOpacity, opacity))
        {
            shownOpacity = opacity;
            terrainImage.color = new Color(1f, 1f, 1f, opacity);
        }
    }

    void UpdateInfoText()
    {
        float view = ViewMeters;
        string text = $"{FormatDistance(view)} view  |  grid {FormatDistance(NiceStep(view / 5f))}";

        // Show the distance to the nearest oxygen refill.
        MapPoint best = null;
        float bestDist = float.MaxValue;
        foreach (var pt in points)
        {
            if (!pt.o2_refill) continue;
            float d = Vector2.Distance(PlayerXZ, new Vector2(pt.x_m, pt.z_m));
            if (d < bestDist) { bestDist = d; best = pt; }
        }
        if (best != null) text += $"  |  O2: {best.name} {FormatDistance(bestDist)}";
        infoText.text = text;
    }

    static string FormatDistance(float meters)
    {
        if (meters < 1000f) return $"{Mathf.RoundToInt(meters)} m";
        return $"{meters / 1000f:0.#} km";
    }

    // Markers
    void UpdateMarkers(float rot)
    {
        points.Clear();
        seenCodes.Clear();
        foreach (var m in MapMarker.All)
        {
            if (m == null || string.IsNullOrEmpty(m.code) || !seenCodes.Add(m.code)) continue;
            points.Add(m.ToPoint());
        }
        foreach (var kv in externalPins)
            if (seenCodes.Add(kv.Key)) points.Add(kv.Value);
        // Show navigation destinations even without a scene marker.
        var nav = RouteNavigator.Current;
        if (nav != null)
            foreach (var destination in nav.Destinations)
                if (seenCodes.Add(destination.code)) points.Add(destination);
        string active = nav != null && nav.MapRoute != null ? nav.MapRoute.destinationId : null;
        float pulse = 1f + 0.15f * Mathf.Sin(Time.unscaledTime * 6f);

        foreach (var v in views.Values) v.used = false;

        float pxPerMeter = PX / ViewMeters;
        float edge = PX * 0.5f - IconPx * 0.5f - 4f;
        Quaternion upright = Quaternion.Euler(0f, 0f, -rot);   // keep icons and labels readable while the map turns

        foreach (var pt in points)
        {
            MarkerView v = GetView(pt.code);
            v.used = true;

            Vector2 local = (new Vector2(pt.x_m, pt.z_m) - PlayerXZ) * pxPerMeter;   // +x east = right, +z north = up
            float dist = local.magnitude;
            bool onRim = dist > edge;
            if (onRim) local *= edge / dist;

            Color c = MinimapStyle.ColorFor(pt.location_type, pt.hazard_level);
            v.root.anchoredPosition = local;
            v.root.localRotation = upright;
            // The selected destination and the routed one get a bigger, pulsing marker.
            bool highlighted = pt.code == SelectedCode || pt.code == active;
            v.root.localScale = Vector3.one * (onRim ? 0.75f : 1f) * (highlighted ? 1.4f * pulse : 1f);
            v.icon.color = highlighted ? Color.Lerp(c, Color.white, 0.35f) : c;
            string label = onRim ? pt.code : (string.IsNullOrEmpty(pt.name) ? pt.code : pt.name);
            if (v.label.text != label) v.label.text = label;
            v.label.color = c;

            // Draw zone circles at their real size.
            float zonePx = Mathf.Max(0f, pt.radius_m) * 2f * pxPerMeter;
            bool showZone = !onRim && zonePx > IconPx * 1.2f;
            if (v.zone.enabled != showZone) v.zone.enabled = showZone;
            if (showZone)
            {
                v.zone.rectTransform.anchoredPosition = local;
                v.zone.rectTransform.sizeDelta = new Vector2(zonePx, zonePx);
                bool risky = pt.hazard_level == "danger" || pt.hazard_level == "caution";
                v.zone.color = new Color(c.r, c.g, c.b, risky ? 0.28f : 0.14f);
            }
        }

        unusedCodes.Clear();
        foreach (var kv in views)
            if (!kv.Value.used) unusedCodes.Add(kv.Key);
        foreach (var code in unusedCodes)
        {
            Destroy(views[code].root.gameObject);
            Destroy(views[code].zone.gameObject);
            views.Remove(code);
        }
    }

    // Rebuild line segments when the route changes. Only reposition them each frame.
    void UpdateRoute()
    {
        // Preview and stopped routes are drawn faint; while navigating the walked part goes grey.
        var nav = RouteNavigator.Current;
        var route = nav != null ? nav.MapRoute : null;
        if (route != drawnRoute)
        {
            drawnRoute = route;
            routeXZ.Clear();
            routeAlong.Clear();
            if (route != null && route.IsBuilt)
            {
                // Sample every few meters to keep the map line lightweight.
                foreach (var p in route.points)
                {
                    var xz = new Vector2(p.x, p.z);
                    if (routeXZ.Count == 0 || Vector2.Distance(routeXZ[routeXZ.Count - 1], xz) >= 4f) routeXZ.Add(xz);
                }
                var end = route.points[route.points.Length - 1];
                routeXZ[routeXZ.Count - 1] = new Vector2(end.x, end.z);
                routeAlong.Add(0f);
                for (int i = 1; i < routeXZ.Count; i++) routeAlong.Add(routeAlong[i - 1] + Vector2.Distance(routeXZ[i - 1], routeXZ[i]));
            }
            while (routeSegments.Count < routeXZ.Count - 1)
            {
                var segment = NewImage("Route Segment", routeLayer, null, 1f);
                segment.color = RouteColor;
                routeSegments.Add(segment);
            }
            for (int i = 0; i < routeSegments.Count; i++)
                routeSegments[i].enabled = i < routeXZ.Count - 1;
        }
        if (routeXZ.Count < 2) return;

        bool navigating = nav.Navigating;
        float walked = navigating ? nav.TraveledMeters : -1f;
        float pxPerMeter = PX / ViewMeters;
        Vector2 a = (routeXZ[0] - PlayerXZ) * pxPerMeter;
        for (int i = 1; i < routeXZ.Count; i++)
        {
            Vector2 b = (routeXZ[i] - PlayerXZ) * pxPerMeter;
            Vector2 d = b - a;
            var image = routeSegments[i - 1];
            Color color = !navigating ? RoutePreviewColor : routeAlong[i] <= walked ? RouteWalkedColor : RouteColor;
            if (image.color != color) image.color = color;
            var rt = image.rectTransform;
            rt.anchoredPosition = (a + b) * 0.5f;
            rt.sizeDelta = new Vector2(d.magnitude + 3f, 6f);   // a little overlap hides the joints
            rt.localRotation = Quaternion.Euler(0f, 0f, Mathf.Atan2(d.y, d.x) * Mathf.Rad2Deg);
            a = b;
        }
    }

    MarkerView GetView(string code)
    {
        if (views.TryGetValue(code, out var v)) return v;
        v = new MarkerView();
        v.zone = NewImage("Zone " + code, zoneLayer, discSprite, 10f);
        v.zone.enabled = false;
        v.icon = NewImage("Marker " + code, markerLayer, badgeSprite, IconPx);
        v.root = v.icon.rectTransform;
        v.label = NewText("Label", v.root, 28, new Vector2(240f, 34f));
        v.label.rectTransform.anchoredPosition = new Vector2(0f, IconPx * 0.5f + 17f);
        views[code] = v;
        return v;
    }

    // Terrain relief picture
    void ScanTerrains()
    {
        nextTerrainScan = Time.unscaledTime + 2f;
        var found = Terrain.activeTerrains;
        var list = new List<TerrainInfo>();
        minY = float.MaxValue;
        maxY = float.MinValue;
        foreach (var t in found)
        {
            if (t == null || t.terrainData == null) continue;
            Vector3 pos = t.GetPosition();
            Vector3 size = t.terrainData.size;
            list.Add(new TerrainInfo { data = t.terrainData, x = pos.x, y = pos.y, z = pos.z, sx = size.x, sy = size.y, sz = size.z });
            minY = Mathf.Min(minY, pos.y);
            maxY = Mathf.Max(maxY, pos.y + size.y);
        }
        if (list.Count == 0) { minY = 0f; maxY = 1f; }
        else if (terrains.Length == 0) Debug.Log($"[Minimap] Found {list.Count} terrain tiles, elevation {minY:0} to {maxY:0} m.");
        terrainsChanged = list.Count != terrains.Length;
        // Use the finer terrain where HiRISE and CTX overlap.
        list.Sort((a, b) => (a.sx * a.sz / (a.data.heightmapResolution * a.data.heightmapResolution)).CompareTo(
            b.sx * b.sz / (b.data.heightmapResolution * b.data.heightmapResolution)));
        terrains = list.ToArray();
    }

    void UpdateTerrainTexture()
    {
        float view = ViewMeters;
        float cover = view * 2f;   // texture is twice the view, so it can scroll before it needs redrawing

        if (building == null)
        {
            Vector2 off = PlayerXZ - frontCenter;
            bool need = !hasFront || terrainsChanged || !Mathf.Approximately(frontCover, cover)
                        || Mathf.Max(Mathf.Abs(off.x), Mathf.Abs(off.y)) > view * 0.25f;
            if (need) building = StartCoroutine(BuildTexture(PlayerXZ, cover));
        }

        if (hasFront)
        {
            // Keep showing the current texture while the next one is built.
            float w = view / frontCover;
            Vector2 min = (PlayerXZ - frontCenter) / frontCover + new Vector2(0.5f - w * 0.5f, 0.5f - w * 0.5f);
            terrainImage.uvRect = new Rect(min.x, min.y, w, w);
        }
        if (terrainImage.enabled != hasFront) terrainImage.enabled = hasFront;
    }

    // Build relief rows within buildBudgetMs each frame then swap textures.
    IEnumerator BuildTexture(Vector2 center, float cover)
    {
        terrainsChanged = false;
        int n = textureSize;
        if (pixels == null || pixels.Length != n * n)
        {
            pixels = new Color32[n * n];
            heights = new float[n * n];
            valid = new bool[n * n];
        }
        float step = cover / n;                       // meters per pixel
        float x0 = center.x - cover * 0.5f + step * 0.5f;
        float z0 = center.y - cover * 0.5f + step * 0.5f;
        var clock = Stopwatch.StartNew();
        int lastTile = -1;

        // Pass 1: sample heights. Row 0 is the south edge (Unity textures start at the bottom).
        for (int j = 0; j < n; j++)
        {
            float z = z0 + j * step;
            int row = j * n;
            for (int i = 0; i < n; i++)
                heights[row + i] = SampleHeight(x0 + i * step, z, ref lastTile, out valid[row + i]);
            if (clock.Elapsed.TotalMilliseconds > buildBudgetMs) { yield return null; clock.Restart(); }
        }

        // Pass 2: color by elevation, shade by slope, add contour and grid lines.
        float range = Mathf.Max(1f, maxY - minY);
        float contour = NiceStep(cover * 0.5f / 50f);
        float grid = NiceStep(cover * 0.5f / 5f);
        float flat = SunDir.z;
        for (int j = 0; j < n; j++)
        {
            int row = j * n;
            float z = z0 + j * step;
            bool gridRow = gridLines && Mathf.FloorToInt(z / grid) != Mathf.FloorToInt((z + step) / grid);
            for (int i = 0; i < n; i++)
            {
                int k = row + i;
                float x = x0 + i * step;
                bool gridHere = gridRow || (gridLines && Mathf.FloorToInt(x / grid) != Mathf.FloorToInt((x + step) / grid));
                if (!valid[k])
                {
                    pixels[k] = gridHere ? new Color32(40, 38, 44, 255) : NoDataColor;
                    continue;
                }
                float hc = heights[k];
                float hl = i > 0 && valid[k - 1] ? heights[k - 1] : hc;
                float hr = i < n - 1 && valid[k + 1] ? heights[k + 1] : hc;
                float hd = j > 0 && valid[k - n] ? heights[k - n] : hc;
                float hu = j < n - 1 && valid[k + n] ? heights[k + n] : hc;

                float dzdx = (hr - hl) / (2f * step) * verticalExaggeration;
                float dzdz = (hu - hd) / (2f * step) * verticalExaggeration;
                Vector3 normal = new Vector3(-dzdx, -dzdz, 1f).normalized;   // (east, north, up)
                float shade = Mathf.Max(0f, Vector3.Dot(normal, SunDir)) / flat;
                Color col = RampColor((hc - minY) / range) * Mathf.Clamp(0.3f + 0.7f * shade, 0.2f, 1.25f);

                if (contourLines)
                {
                    int c0 = Mathf.FloorToInt(hc / contour);
                    if (Mathf.FloorToInt(hr / contour) != c0 || Mathf.FloorToInt(hu / contour) != c0) col *= 0.72f;
                }
                if (gridHere) col = Color.Lerp(col, Color.white, 0.25f);
                col.a = 1f;
                pixels[k] = col;
            }
            if (clock.Elapsed.TotalMilliseconds > buildBudgetMs) { yield return null; clock.Restart(); }
        }

        // Upload to the spare texture before displaying it.
        if (back == null || back.width != n)
        {
            if (back != null) Destroy(back);
            back = new Texture2D(n, n, TextureFormat.RGBA32, true)
            {
                name = "Minimap Relief",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
        }
        back.SetPixels32(pixels);
        back.Apply(true);
        (front, back) = (back, front);
        frontCenter = center;
        frontCover = cover;
        hasFront = true;
        terrainImage.texture = front;
        building = null;
    }

    float SampleHeight(float x, float z, ref int tile, out bool ok)
    {
        tile = -1;
        for (int t = 0; t < terrains.Length; t++)
        {
            var candidate = terrains[t];
            if (!Inside(candidate, x, z)) continue;
            int holes = candidate.data.holesResolution;
            int hx = Mathf.Clamp((int)((x - candidate.x) / candidate.sx * holes), 0, holes - 1);
            int hz = Mathf.Clamp((int)((z - candidate.z) / candidate.sz * holes), 0, holes - 1);
            if (candidate.data.IsHole(hx, hz)) continue;
            tile = t;
            break;
        }
        if (tile < 0) { ok = false; return 0f; }
        TerrainInfo info = terrains[tile];
        float h = info.data.GetInterpolatedHeight((x - info.x) / info.sx, (z - info.z) / info.sz);
        ok = !(zeroHeightIsNoData && h <= info.sy * 1e-5f);
        return info.y + h;
    }

    static bool Inside(in TerrainInfo t, float x, float z) =>
        x >= t.x && x <= t.x + t.sx && z >= t.z && z <= t.z + t.sz;

    static Color RampColor(float t)
    {
        t = Mathf.Clamp01(t) * (Ramp.Length - 1);
        int i = Mathf.Min((int)t, Ramp.Length - 2);
        return Color.Lerp(Ramp[i], Ramp[i + 1], t - i);
    }

    // Round up to a 1, 2 or 5 step for readable scale labels.
    static float NiceStep(float raw)
    {
        float p = Mathf.Pow(10f, Mathf.Floor(Mathf.Log10(Mathf.Max(raw, 0.01f))));
        float m = raw / p;
        return (m <= 1f ? 1f : m <= 2f ? 2f : m <= 5f ? 5f : 10f) * p;
    }
}

// Generate map sprites without separate image assets.
static class MinimapSprites
{
    public static Sprite Disc(int size) =>
        Make(size, (x, y, s) => Edge(1f - Len(x, y), s));

    public static Sprite Ring(int size, float thickness) =>
        Make(size, (x, y, s) =>
        {
            float d = Len(x, y);
            return Edge(1f - d, s) * Edge(d - (1f - thickness), s);
        });

    // White dot with a dark rim. Tinted per marker; the rim stays dark.
    public static Sprite Badge(int size)
    {
        var px = new Color32[size * size];
        float s = size * 0.5f;
        for (int j = 0; j < size; j++)
            for (int i = 0; i < size; i++)
            {
                float x = (i + 0.5f) / s - 1f, y = (j + 0.5f) / s - 1f, d = Len(x, y);
                float inner = Edge(0.68f - d, s);
                Color c = Color.Lerp(new Color(0.08f, 0.08f, 0.08f), Color.white, inner);
                c.a = Edge(1f - d, s);
                px[j * size + i] = c;
            }
        return ToSprite(px, size);
    }

    // Arrowhead pointing up (the astronaut).
    public static Sprite Arrow(int size)
    {
        Vector2[] poly = { new Vector2(0f, 0.95f), new Vector2(0.72f, -0.8f), new Vector2(0f, -0.38f), new Vector2(-0.72f, -0.8f) };
        var px = new Color32[size * size];
        const int ss = 4;   // 4x4 supersampling for smooth edges
        for (int j = 0; j < size; j++)
            for (int i = 0; i < size; i++)
            {
                int hits = 0;
                for (int sy = 0; sy < ss; sy++)
                    for (int sx = 0; sx < ss; sx++)
                    {
                        float x = (i + (sx + 0.5f) / ss) / size * 2f - 1f;
                        float y = (j + (sy + 0.5f) / ss) / size * 2f - 1f;
                        if (InPolygon(poly, x, y)) hits++;
                    }
                px[j * size + i] = new Color(1f, 1f, 1f, hits / (float)(ss * ss));
            }
        return ToSprite(px, size);
    }

    // White sprite whose alpha comes from alpha(x, y, pixelsPerUnit), with x and y in -1..1.
    static Sprite Make(int size, Func<float, float, float, float> alpha)
    {
        var px = new Color32[size * size];
        float s = size * 0.5f;
        for (int j = 0; j < size; j++)
            for (int i = 0; i < size; i++)
            {
                float x = (i + 0.5f) / s - 1f, y = (j + 0.5f) / s - 1f;
                px[j * size + i] = new Color(1f, 1f, 1f, alpha(x, y, s));
            }
        return ToSprite(px, size);
    }

    static Sprite ToSprite(Color32[] px, int size)
    {
        var tex = new Texture2D(size, size, TextureFormat.RGBA32, true)
        {
            wrapMode = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
        };
        tex.SetPixels32(px);
        tex.Apply(true);
        return Sprite.Create(tex, new Rect(0, 0, size, size), new Vector2(0.5f, 0.5f), 100f);
    }

    static float Len(float x, float y) => Mathf.Sqrt(x * x + y * y);

    // 0 outside, 1 inside, with about one pixel of soft edge.
    static float Edge(float signedDist, float pixelsPerUnit) => Mathf.Clamp01(signedDist * pixelsPerUnit + 0.5f);

    static bool InPolygon(Vector2[] poly, float x, float y)
    {
        bool inside = false;
        for (int a = 0, b = poly.Length - 1; a < poly.Length; b = a++)
            if ((poly[a].y > y) != (poly[b].y > y) &&
                x < (poly[b].x - poly[a].x) * (y - poly[a].y) / (poly[b].y - poly[a].y) + poly[a].x)
                inside = !inside;
        return inside;
    }
}
