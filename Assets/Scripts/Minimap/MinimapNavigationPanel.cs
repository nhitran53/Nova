using System.Collections.Generic;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.Hands;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// Poke with the left index finger on Quest. Press Esc and click on desktop.
[RequireComponent(typeof(MarsMinimap))]
[DefaultExecutionOrder(-90)]   // Read map input before gestures and movement.
[DisallowMultipleComponent]
public class MinimapNavigationPanel : MonoBehaviour
{
    [Header("Poke (left index fingertip on the hand map)")]
    [Tooltip("Distance in meters that pauses pinch and fist input near the map.")]
    [SerializeField, Min(0.01f)] float hoverDistance = 0.06f;
    [Tooltip("Press when the fingertip gets this close to the map surface.")]
    [SerializeField, Min(0)] float pressDistance = 0.008f;
    [Tooltip("Pullback distance in meters before another press.")]
    [SerializeField, Min(0.005f)] float releaseDistance = 0.025f;
    [Tooltip("Seconds to ignore presses after opening.")]
    [SerializeField, Min(0)] float openGrace = 0.5f;
    [SerializeField, Min(0)] float debounce = 0.35f;
    [Tooltip("Marker hit radius in map units. The map diameter is 512 units.")]
    [SerializeField, Min(10)] float markerHitRadius = 44f;

    public static bool FingerNearMap { get; private set; }
    static MinimapNavigationPanel current;

    const float PX = 512f;
    static readonly Color PanelColor = new Color(0.04f, 0.05f, 0.07f, 0.85f);
    static readonly Color ButtonColor = new Color(0.12f, 0.45f, 0.55f, 0.95f);
    static readonly Color ButtonHover = new Color(0.25f, 0.75f, 0.9f, 1f);
    static readonly Color StopColor = new Color(0.6f, 0.25f, 0.2f, 0.95f);

    MarsMinimap map;
    RectTransform panel, button;
    Image buttonImage;
    Text title, info, buttonLabel;
    string selected;
    NavigationRoute lastMapRoute;
    float holographicSince = -1, nextPress, nextRefresh;
    bool armed, pointerIsWorld;
    Vector3 pointerWorld;
    Vector2 pointerScreen;
    Transform boundHead, trackingSpace;
    XRHandSubsystem hands;
    static readonly List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();

    void Awake() => map = GetComponent<MarsMinimap>();
    void OnEnable() => current = this;

    void OnDisable()
    {
        if (current == this) current = null;
        FingerNearMap = false;
        if (panel != null) panel.gameObject.SetActive(false);
    }

    public static bool MouseOverControls()
    {
#if ENABLE_INPUT_SYSTEM
        if (current == null || Mouse.current == null || !current.MouseMode) return false;
        current.SetPointer(Mouse.current.position.ReadValue());
        return current.HitAnything(out _);
#else
        return false;
#endif
    }

    public void SelectDestination(string code)
    {
        if (map == null) map = GetComponent<MarsMinimap>();
        selected = map.SelectedCode = code;
        lastMapRoute = RouteNavigator.Current != null ? RouteNavigator.Current.MapRoute : null;
        nextRefresh = 0;
    }

    bool MouseMode => map.Visible && !map.Holographic && !map.DockedInHelmetHud;

    void Update()
    {
        var nav = RouteNavigator.Current;
        if (panel == null && map.Content != null) Build();
        if (panel == null || nav == null) { FingerNearMap = false; return; }

        if (nav.MapRoute != lastMapRoute)
        {
            lastMapRoute = nav.MapRoute;
            if (lastMapRoute != null) selected = lastMapRoute.destinationId;
        }
        if (selected != null && !nav.HasDestination(selected)) selected = null;
        map.SelectedCode = selected;

        UpdatePoke();
#if ENABLE_INPUT_SYSTEM
        var mouse = Mouse.current;
        if (MouseMode && mouse != null && Cursor.lockState != CursorLockMode.Locked)
        {
            SetPointer(mouse.position.ReadValue());
            if (mouse.leftButton.wasPressedThisFrame) Press(nav);
        }
#endif
        Layout();
        bool show = map.Visible && !map.DockedInHelmetHud && selected != null;
        if (panel.gameObject.activeSelf != show) panel.gameObject.SetActive(show);
        if (show) Refresh(nav);
    }

    void UpdatePoke()
    {
        FingerNearMap = false;
        float now = Time.unscaledTime;
        if (!map.Holographic) { holographicSince = -1; armed = false; return; }
        if (holographicSince < 0) holographicSince = now;
        if (!ReadFingertip(out Vector3 tip)) { armed = false; return; }

        // Negative distance faces the viewer.
        Vector3 local = map.Content.InverseTransformPoint(tip);
        float viewerSide = map.head != null && map.Content.InverseTransformPoint(map.head.position).z > 0 ? -1 : 1;
        float depth = local.z * map.Content.lossyScale.z * viewerSide;
        pointerIsWorld = true;
        pointerWorld = tip;
        bool over = new Vector2(local.x, local.y).magnitude < PX * 0.5f + 24f || HitRect(panel);
        FingerNearMap = over && depth > -hoverDistance && depth < releaseDistance;

        // Require a front approach before pressing.
        if (over && depth < -releaseDistance && now - holographicSince >= openGrace) armed = true;
        if (!over || depth > releaseDistance) armed = false;
        if (armed && depth >= -pressDistance && now >= nextPress)
        {
            armed = false;
            nextPress = now + debounce;
            Press(RouteNavigator.Current);
        }
    }

    bool ReadFingertip(out Vector3 tip)
    {
        tip = default;
        if (boundHead != map.head)
        {
            boundHead = map.head;
            var origin = boundHead != null ? boundHead.GetComponentInParent<XROrigin>() : null;
            trackingSpace = origin != null ? origin.CameraFloorOffsetObject.transform : null;
        }
        if (trackingSpace == null) return false;
        var aim = MetaAimHand.left;
        if (aim != null && aim.added &&
            ((MetaAimFlags)aim.aimFlags.ReadValue() & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) != 0)
            return false;
        if (hands == null || !hands.running)
        {
            hands = null;
            SubsystemManager.GetSubsystems(subsystems);
            foreach (var s in subsystems) if (s.running) { hands = s; break; }
        }
        if (hands == null || !hands.leftHand.isTracked ||
            !hands.leftHand.GetJoint(XRHandJointID.IndexTip).TryGetPose(out Pose pose)) return false;
        tip = trackingSpace.TransformPoint(pose.position);
        return true;
    }

    void SetPointer(Vector2 screen)
    {
        pointerIsWorld = false;
        pointerScreen = screen;
    }

    bool ToLocal(RectTransform rt, out Vector2 local)
    {
        if (pointerIsWorld)
        {
            local = rt.InverseTransformPoint(pointerWorld);
            return true;
        }
        return RectTransformUtility.ScreenPointToLocalPointInRectangle(rt, pointerScreen, null, out local);
    }

    bool HitRect(RectTransform rt) =>
        rt != null && rt.gameObject.activeInHierarchy && ToLocal(rt, out var local) && rt.rect.Contains(local);

    // Test in marker space to handle map rotation and scale.
    bool HitMarker(RectTransform marker) =>
        marker != null && ToLocal((RectTransform)marker.parent, out var local) &&
        Vector2.Distance(local, marker.anchoredPosition) < markerHitRadius;

    bool HitAnything(out string marker)
    {
        marker = null;
        if (HitRect(button) || HitRect(panel)) return true;
        var nav = RouteNavigator.Current;
        if (nav == null) return false;
        foreach (var d in nav.Destinations)
            if (HitMarker(map.MarkerRect(d.code))) { marker = d.code; return true; }
        return false;
    }

    void Press(RouteNavigator nav)
    {
        if (nav == null) return;
        if (HitRect(button)) { RunAction(nav); return; }
        if (HitRect(panel)) return;
        if (HitAnything(out string marker) && marker != null)
            selected = selected == marker ? null : marker;
    }

    enum Action { Find, Start, Stop }

    Action NextAction(RouteNavigator nav)
    {
        if (nav.Navigating) return Action.Stop;
        return nav.MapRoute != null && nav.MapRoute.destinationId == selected ? Action.Start : Action.Find;
    }

    void RunAction(RouteNavigator nav)
    {
        switch (NextAction(nav))
        {
            case Action.Find: nav.FindNavigationRoute(new NavigationRequest { destination_id = selected }); break;
            case Action.Start: nav.StartNavigation(nav.MapRoute.routeId); break;
            case Action.Stop: nav.StopNavigation(); break;
        }
        nextRefresh = 0;
    }

    void Refresh(RouteNavigator nav)
    {
        var action = NextAction(nav);
        bool hover = HitRect(button) && (!pointerIsWorld || FingerNearMap);
        buttonImage.color = hover ? ButtonHover : action == Action.Stop ? StopColor : ButtonColor;
        if (Time.unscaledTime < nextRefresh) return;
        nextRefresh = Time.unscaledTime + 0.2f;

        MapPoint point = null;
        foreach (var marker in MapMarker.All)
            if (marker != null && marker.code == selected) { point = marker.ToPoint(); break; }
        if (point == null)
            foreach (var d in nav.Destinations) if (d.code == selected) point = d;
        if (point == null) return;
        title.text = $"{point.name.ToUpperInvariant()}   <size=20>{point.code}</size>";
        buttonLabel.text = action == Action.Find ? "FIND ROUTE" : action == Action.Start ? "START NAVIGATION" : "STOP NAVIGATION";

        float away = Vector2.Distance(map.PlayerXZ, new Vector2(point.x_m, point.z_m));
        var route = nav.MapRoute != null && nav.MapRoute.destinationId == selected ? nav.MapRoute : null;
        if (route == null) info.text = $"About {Mathf.RoundToInt(away)} m away";
        else if (nav.Navigating)
            info.text = nav.Status == NavigationStatus.Arrived ? "Arrived"
                : $"{Mathf.RoundToInt(nav.RemainingMeters)} m left  ·  ETA {Clock(nav.EtaSeconds)}" +
                  (nav.Status == NavigationStatus.OffRoute ? "\nOFF ROUTE" : "");
        else
            info.text = $"Route {Mathf.RoundToInt(route.lengthMeters)} m  ·  about {Clock(route.lengthMeters / Mathf.Max(0.1f, nav.walkingSpeed))} walk" +
                (nav.Status == NavigationStatus.Stopped ? "\nNavigation stopped" : "");
    }

    static string Clock(float seconds)
    {
        int s = Mathf.CeilToInt(seconds);
        return $"{s / 60}:{s % 60:00}";
    }

    // Above the hand map or beside the desktop map.
    void Layout()
    {
        panel.anchoredPosition = map.Holographic ? new Vector2(0, PX * 0.5f + 150) : new Vector2(PX * 0.5f + 250, 0);
    }

    void Build()
    {
        var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
        panel = NewRect("Destination Panel", map.Content, new Vector2(420, 190));
        var background = panel.gameObject.AddComponent<Image>();
        background.color = PanelColor;
        background.raycastTarget = false;
        background.material = map.UiMaterial;

        title = NewText("Title", panel, font, 32, new Vector2(400, 40), new Vector2(0, 62));
        title.fontStyle = FontStyle.Bold;
        info = NewText("Info", panel, font, 24, new Vector2(400, 60), new Vector2(0, 12));
        button = NewRect("Button", panel, new Vector2(320, 62));
        button.anchoredPosition = new Vector2(0, -56);
        buttonImage = button.gameObject.AddComponent<Image>();
        buttonImage.raycastTarget = false;
        buttonImage.material = map.UiMaterial;
        buttonLabel = NewText("Label", button, font, 28, new Vector2(320, 62), Vector2.zero);
        buttonLabel.fontStyle = FontStyle.Bold;
        panel.gameObject.SetActive(false);
    }

    static RectTransform NewRect(string name, Transform parent, Vector2 size)
    {
        var rt = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        rt.gameObject.layer = parent.gameObject.layer;
        rt.SetParent(parent, false);
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);
        rt.sizeDelta = size;
        return rt;
    }

    Text NewText(string name, Transform parent, Font font, int size, Vector2 box, Vector2 position)
    {
        var t = NewRect(name, parent, box).gameObject.AddComponent<Text>();
        t.rectTransform.anchoredPosition = position;
        t.font = font;
        t.fontSize = size;
        t.alignment = TextAnchor.MiddleCenter;
        t.horizontalOverflow = HorizontalWrapMode.Overflow;
        t.verticalOverflow = VerticalWrapMode.Overflow;
        t.supportRichText = true;
        t.color = Color.white;
        t.raycastTarget = false;
        t.material = map.UiMaterial;
        return t;
    }
}
