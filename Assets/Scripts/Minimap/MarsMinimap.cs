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

// NOVA helmet minimap (Stage 4).
//
// Add this to the Main Camera (Tools > Nova > Minimap > Add Minimap To Main Camera) and press Play.
// It builds its own small round map, pinned to the side of the view like a helmet display:
//   - the ground is a shaded-relief picture drawn from the real Jezero terrain heights
//   - every MapMarker in the scene (and any pins from the API) is drawn as a colored dot
//   - the white arrow in the middle is you; the map turns so "up" is where you are looking
//   - markers that are too far away stick to the rim, so you can still see which way home is
//
// No prefabs or scene wiring needed, so it won't conflict with teammates editing the scene.
// Works the same on desktop (Mars Explorer) and later on Quest (the XR camera is still Camera.main).
public class MarsMinimap : MonoBehaviour
{
    [Header("Astronaut")]
    [Tooltip("The eye/head. Empty = this object if it is a camera, otherwise Camera.main.")]
    public Transform head;

    [Header("Placement in the helmet (measured from the eye)")]
    [Min(0.2f)] public float distance = 0.6f;
    [Tooltip("Negative = left of center.")]
    [Range(-60f, 60f)] public float yawDeg = -28f;
    [Tooltip("Negative = below center.")]
    [Range(-60f, 60f)] public float pitchDeg = -18f;
    [Tooltip("Physical width of the round map in meters.")]
    [Min(0.03f)] public float sizeMeters = 0.18f;

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

    // ---------- read by other scripts (MinimapApiLink, the AI hooks) ----------
    public Vector2 PlayerXZ { get; private set; }
    public float HeadingDeg { get; private set; }   // 0..360, 0 = north (+z), 90 = east (+x)
    public float ViewMeters => zoomLevelsMeters[Mathf.Clamp(zoomIndex, 0, zoomLevelsMeters.Length - 1)];
    public bool Visible => canvasGO != null && canvasGO.activeSelf;

    const float PX = 512f;      // the map circle is 512 canvas units wide (then scaled down to sizeMeters)
    const float IconPx = 26f;

    // UI
    int uiLayer;
    Font font;
    Sprite discSprite, ringSprite, badgeSprite, arrowSprite;
    GameObject canvasGO;
    RectTransform rotator, zoneLayer, markerLayer, arrow, north;
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

    static readonly Color32 NoDataColor = new Color32(22, 20, 24, 255);
    static readonly Color[] Ramp = {
        new Color(0.20f, 0.09f, 0.06f),   // low: dark rust
        new Color(0.45f, 0.20f, 0.10f),
        new Color(0.72f, 0.40f, 0.20f),
        new Color(0.88f, 0.62f, 0.38f),
        new Color(0.97f, 0.86f, 0.70f),   // high: pale butterscotch
    };
    static readonly Vector3 SunDir = new Vector3(-0.5f, 0.5f, 0.7071f);   // from the north-west, 45 degrees up

    // =====================================================================
    // Public controls (for gestures, voice, or the AI's "show me the minimap")
    // =====================================================================
    public void Show(bool on) { if (canvasGO != null) canvasGO.SetActive(on); }
    public void ZoomIn() => SetZoomIndex(zoomIndex - 1);
    public void ZoomOut() => SetZoomIndex(zoomIndex + 1);
    public void SetZoomIndex(int index) => zoomIndex = Mathf.Clamp(index, 0, zoomLevelsMeters.Length - 1);
    public void SetHeadingUp(bool on) => headingUp = on;

    // Pulse the rim for a few seconds (e.g. when the AI says "here is your map").
    public void Highlight(float seconds = 4f)
    {
        Show(true);
        highlightUntil = Time.unscaledTime + seconds;
    }

    // Extra pins that are not scene objects (from GET /map, vision detections...).
    // A scene MapMarker with the same code always wins.
    public void SetExternalPins(IEnumerable<MapPoint> pins)
    {
        externalPins.Clear();
        if (pins == null) return;
        foreach (var p in pins)
            if (p != null && !string.IsNullOrEmpty(p.code))
                externalPins[p.code] = p;
    }

    // =====================================================================
    // Unity lifecycle
    // =====================================================================
    void Start()
    {
        if (zoomLevelsMeters == null || zoomLevelsMeters.Length == 0) zoomLevelsMeters = new[] { 1000f };
        SetZoomIndex(zoomIndex);
        textureSize = Mathf.Clamp(textureSize, 64, 2048);
        if (head == null)
            head = GetComponent<Camera>() != null ? transform : (Camera.main != null ? Camera.main.transform : null);

        BuildHud();
        ScanTerrains();
    }

    void OnEnable() { if (canvasGO != null) canvasGO.SetActive(true); }

    void OnDisable()
    {
        if (canvasGO != null) canvasGO.SetActive(false);
        StopAllCoroutines();
        building = null;
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
        if (head == null)
        {
            if (Camera.main == null) return;
            head = Camera.main.transform;
        }
        if (canvasGO.transform.parent != head) canvasGO.transform.SetParent(head, false);
        ApplyPlacement();
        HandleKeys();

        // Always track the astronaut, even while the map is hidden (the API link reads these).
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

        UpdateMarkers(rot);
        UpdateFrame();
        if (Time.unscaledTime >= nextInfo)
        {
            nextInfo = Time.unscaledTime + 0.25f;
            UpdateInfoText();
        }
    }

    void ApplyPlacement()
    {
        Transform t = canvasGO.transform;
        Quaternion dir = Quaternion.Euler(-pitchDeg, yawDeg, 0f);
        t.localPosition = dir * Vector3.forward * distance;
        t.localRotation = dir;   // faces the eye
        t.localScale = Vector3.one * (sizeMeters / PX);
    }

    // Compass heading of where the astronaut is looking, flattened onto the ground.
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

    // =====================================================================
    // Building the HUD (done in code so there is nothing to wire up)
    // =====================================================================
    void BuildHud()
    {
        uiLayer = LayerMask.NameToLayer("UI");
        if (uiLayer < 0) uiLayer = 0;
        font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        discSprite = MinimapSprites.Disc(256);
        ringSprite = MinimapSprites.Ring(256, 0.05f);
        badgeSprite = MinimapSprites.Badge(64);
        arrowSprite = MinimapSprites.Arrow(64);

        canvasGO = new GameObject("Nova Minimap", typeof(RectTransform), typeof(Canvas));
        canvasGO.layer = uiLayer;
        canvasGO.GetComponent<Canvas>().renderMode = RenderMode.WorldSpace;
        var root = (RectTransform)canvasGO.transform;
        root.sizeDelta = new Vector2(PX + 80f, PX + 120f);
        if (head != null) root.SetParent(head, false);

        frame = NewImage("Rim", root, ringSprite, PX + 18f);

        // The mask is a dark disc: it is the background and cuts everything inside into a circle.
        var mask = NewImage("Mask", root, discSprite, PX);
        mask.color = new Color(0.05f, 0.04f, 0.04f, 0.85f);
        mask.gameObject.AddComponent<Mask>().showMaskGraphic = true;

        rotator = NewRect("Rotator", mask.rectTransform, new Vector2(PX, PX));

        var terrainRt = NewRect("Terrain", rotator, new Vector2(PX, PX));
        terrainImage = terrainRt.gameObject.AddComponent<RawImage>();
        terrainImage.raycastTarget = false;
        terrainImage.enabled = false;

        zoneLayer = NewRect("Zones", rotator, new Vector2(PX, PX));
        markerLayer = NewRect("Markers", rotator, new Vector2(PX, PX));

        var arrowImg = NewImage("You", root, arrowSprite, 38f);
        var arrowOutline = arrowImg.gameObject.AddComponent<Outline>();
        arrowOutline.effectColor = new Color(0f, 0f, 0f, 0.9f);
        arrowOutline.effectDistance = new Vector2(2f, -2f);
        arrow = arrowImg.rectTransform;

        var northText = NewText("North", root, 30, new Vector2(40f, 40f));
        northText.text = "N";
        northText.fontStyle = FontStyle.Bold;
        northText.color = new Color(1f, 0.45f, 0.35f);
        north = northText.rectTransform;

        infoText = NewText("Info", root, 22, new Vector2(PX + 80f, 36f));
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
        var outline = t.gameObject.AddComponent<Outline>();
        outline.effectColor = new Color(0f, 0f, 0f, 0.85f);
        outline.effectDistance = new Vector2(1.5f, -1.5f);
        return t;
    }

    void UpdateFrame()
    {
        Color normal = new Color(0.85f, 0.92f, 1f, 0.6f);
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

        // Nearest oxygen refill: the one number an astronaut always wants on the map.
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

    // =====================================================================
    // Markers
    // =====================================================================
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
            v.root.localScale = Vector3.one * (onRim ? 0.75f : 1f);
            v.icon.color = c;
            string label = onRim ? pt.code : (string.IsNullOrEmpty(pt.name) ? pt.code : pt.name);
            if (v.label.text != label) v.label.text = label;
            v.label.color = c;

            // Zone circle (hazard areas, habitat footprint...) in true scale.
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

    MarkerView GetView(string code)
    {
        if (views.TryGetValue(code, out var v)) return v;
        v = new MarkerView();
        v.zone = NewImage("Zone " + code, zoneLayer, discSprite, 10f);
        v.zone.enabled = false;
        v.icon = NewImage("Marker " + code, markerLayer, badgeSprite, IconPx);
        v.root = v.icon.rectTransform;
        v.label = NewText("Label", v.root, 20, new Vector2(200f, 26f));
        v.label.rectTransform.anchoredPosition = new Vector2(0f, IconPx * 0.5f + 13f);
        views[code] = v;
        return v;
    }

    // =====================================================================
    // Terrain relief picture
    // =====================================================================
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
            // Show the part of the texture around the player. Works even while a new one is being drawn.
            float w = view / frontCover;
            Vector2 min = (PlayerXZ - frontCenter) / frontCover + new Vector2(0.5f - w * 0.5f, 0.5f - w * 0.5f);
            terrainImage.uvRect = new Rect(min.x, min.y, w, w);
        }
        if (terrainImage.enabled != hasFront) terrainImage.enabled = hasFront;
    }

    // Draws the relief a few rows per frame (time-boxed by buildBudgetMs), then swaps it in.
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

        // Upload into the hidden texture, then swap it in.
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
        if (tile < 0 || !Inside(terrains[tile], x, z))
        {
            tile = -1;
            for (int t = 0; t < terrains.Length; t++)
                if (Inside(terrains[t], x, z)) { tile = t; break; }
            if (tile < 0) { ok = false; return 0f; }
        }
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

    // 1, 2, 5, 10, 20, 50... the first "round" number at or above raw.
    static float NiceStep(float raw)
    {
        float p = Mathf.Pow(10f, Mathf.Floor(Mathf.Log10(Mathf.Max(raw, 0.01f))));
        float m = raw / p;
        return (m <= 1f ? 1f : m <= 2f ? 2f : m <= 5f ? 5f : 10f) * p;
    }
}

// Small sprites drawn in code, so the minimap needs no image assets.
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
