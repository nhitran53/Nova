using System.Collections.Generic;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public enum NavigationStatus { Idle, Preview, OnRoute, OffRoute, Arrived, Stopped }

// Build the trail once. Share progress through Current.
public class RouteNavigator : MonoBehaviour
{
    public static RouteNavigator Current { get; private set; }

    [Tooltip("Saved routes available for navigation.")]
    [SerializeField] NavigationRoute[] routes;
    [Tooltip("Leave empty to follow the active desktop or Quest camera.")]
    public Transform head;
    [SerializeField] NavigationRequest testRequest = new NavigationRequest { destination_id = "shuttle_001" };

    [Header("Progress")]
    [Min(0.1f)] public float walkingSpeed = 1.2f;
    [Tooltip("Further than this from the trail counts as off route.")]
    [SerializeField, Min(1)] float offRouteMeters = 8;
    [Tooltip("Return distance in meters. Keep below Off Route Meters.")]
    [SerializeField, Min(0.5f)] float backOnRouteMeters = 5;
    [SerializeField, Min(1)] float arrivalMeters = 5;
    [Tooltip("The 'next waypoint' is this far ahead along the trail.")]
    [SerializeField, Min(1)] float lookAheadMeters = 8;
    [Tooltip("Required distance gain before switching trail segments.")]
    [SerializeField, Min(0)] float segmentSwitchMargin = 1.5f;

    [Header("Trail")]
    [SerializeField] Material trailMaterial;
    [SerializeField] Color trailColor = new Color(0.25f, 0.9f, 1f);
    [SerializeField, Min(0.01f)] float trailWidth = 0.12f;
    [SerializeField, Min(0)] float trailLift = 0.2f;

    [Header("Desktop testing")]
    [Tooltip("G = find_navigation_route (test request), H = start_navigation, J = stop_navigation.")]
    [SerializeField] bool keyboardShortcut = true;

    public NavigationService Service { get; private set; }
    public NavigationRoute ActiveRoute { get; private set; }
    // Keep the preview after stopping.
    public NavigationRoute PreviewRoute { get; private set; }
    public NavigationRoute MapRoute => ActiveRoute != null ? ActiveRoute : PreviewRoute;
    public bool Navigating => ActiveRoute != null;
    public NavigationStatus Status { get; private set; }
    public float RemainingMeters { get; private set; }
    public float TraveledMeters { get; private set; }
    public float EtaSeconds { get; private set; }
    public float DistanceFromTrail { get; private set; }
    public Vector3 NextWaypoint { get; private set; }
    public IReadOnlyList<MapPoint> Destinations => destinations;
    public NavigationResponse LastResponse { get; private set; }

    readonly List<MapPoint> destinations = new List<MapPoint>();
    float[] along;   // Distance along the route
    int segment;
    LineRenderer trail;
    Material ownMaterial;
    VoiceSessionClient voice;

    void Awake()
    {
        Current = this;
        Service = new NavigationService(routes ?? new NavigationRoute[0], () => walkingSpeed);
        foreach (var route in Service.Routes)
        {
            if (route == null || !route.IsBuilt || HasDestination(route.destinationId)) continue;
            Vector3 end = route.Destination;
            destinations.Add(new MapPoint
            {
                code = route.destinationId, name = route.destinationName, location_type = MapLocationType.landmark.ToString(),
                x_m = end.x, z_m = end.z, radius_m = arrivalMeters, hazard_level = MapHazardLevel.safe.ToString(), source = "route",
            });
        }
    }

    void OnEnable()
    {
        voice = FindAnyObjectByType<VoiceSessionClient>(FindObjectsInactive.Include);
        if (voice != null) voice.OnNavigation.AddListener(OnVoiceNavigation);
    }

    void OnDisable()
    {
        if (voice != null) voice.OnNavigation.RemoveListener(OnVoiceNavigation);
    }

    // Find previews the route. Start also enables guidance.
    void OnVoiceNavigation(string destinationId)
    {
        var response = FindNavigationRoute(new NavigationRequest { destination_id = destinationId });
        if (response.success && voice.NavigationAction == "start") StartNavigation(response.route_id);
    }

    void OnDestroy()
    {
        if (Current == this) Current = null;
        if (ownMaterial != null) Destroy(ownMaterial);
    }

    public bool HasDestination(string destinationId)
    {
        foreach (var d in destinations) if (d.code == destinationId) return true;
        return false;
    }

    public NavigationResponse FindNavigationRoute(NavigationRequest request)
    {
        var response = Service.FindNavigationRoute(request, out var route);
        if (route != null)
        {
            if (route == ActiveRoute) response.status = "active";
            else
            {
                PreviewRoute = route;
                if (!Navigating) Status = NavigationStatus.Preview;
            }
        }
        return Log("find_navigation_route", request, response);
    }

    // An empty ID starts the last preview.
    public NavigationResponse StartNavigation(string routeId)
    {
        NavigationResponse response;
        NavigationRoute route;
        if (string.IsNullOrWhiteSpace(routeId))
        {
            route = PreviewRoute;
            response = route != null ? Service.Describe(route, "")
                : Service.Describe(null, "error: no route found yet. Call find_navigation_route first");
        }
        else response = Service.GetRoute(routeId, out route);
        if (response.success)
        {
            if (route != ActiveRoute) BeginNavigation(route);
            response.status = "navigating";
        }
        return Log("start_navigation", new NavigationRequest { route_id = routeId }, response);
    }

    public NavigationResponse StopNavigation()
    {
        var route = ActiveRoute;
        NavigationResponse response;
        if (route == null) response = Service.Describe(PreviewRoute, "not navigating");
        else
        {
            ActiveRoute = null;
            PreviewRoute = route;
            Status = NavigationStatus.Stopped;
            if (trail != null) trail.enabled = false;
            response = Service.Describe(route, "stopped");
        }
        response.success = true;
        return Log("stop_navigation", new NavigationRequest(), response);
    }

    public string RunToolJson(string tool, string argumentsJson)
    {
        var args = string.IsNullOrWhiteSpace(argumentsJson) ? new NavigationRequest() : JsonUtility.FromJson<NavigationRequest>(argumentsJson);
        NavigationResponse response;
        switch (tool)
        {
            case "find_navigation_route": response = FindNavigationRoute(args); break;
            case "start_navigation": response = StartNavigation(args.route_id); break;
            case "stop_navigation": response = StopNavigation(); break;
            default: response = Service.Describe(null, $"error: unknown tool '{tool}'"); break;
        }
        return JsonUtility.ToJson(response);
    }

    [ContextMenu("Tools/find_navigation_route (test request)")]
    public void TestFind() => FindNavigationRoute(testRequest);

    [ContextMenu("Tools/start_navigation")]
    public void TestStart() => StartNavigation(null);

    [ContextMenu("Tools/stop_navigation")]
    public void TestStop() => StopNavigation();

    NavigationResponse Log(string tool, NavigationRequest request, NavigationResponse response)
    {
        LastResponse = response;
        string text = $"[Navigation] {tool} {JsonUtility.ToJson(request)}\n-> {JsonUtility.ToJson(response)}";
        if (response.success) Debug.Log(text, this);
        else Debug.LogWarning(text, this);
        return response;
    }

    void BeginNavigation(NavigationRoute route)
    {
        ActiveRoute = route;
        PreviewRoute = route;
        var points = route.points;
        along = new float[points.Length];
        for (int i = 1; i < points.Length; i++) along[i] = along[i - 1] + Vector3.Distance(points[i - 1], points[i]);
        segment = 0;
        Status = NavigationStatus.OnRoute;
        BuildTrail(points);
        UpdateProgress(true);
    }

    void BuildTrail(Vector3[] points)
    {
        if (trail == null)
        {
            var go = new GameObject("Route Trail");
            go.transform.SetParent(transform, false);
            trail = go.AddComponent<LineRenderer>();
            trail.useWorldSpace = true;
            trail.alignment = LineAlignment.View;
            trail.numCornerVertices = 2;
            trail.numCapVertices = 2;
            trail.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            trail.receiveShadows = false;
            trail.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
            trail.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
            trail.allowOcclusionWhenDynamic = false;
            if (trailMaterial == null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Sprites/Default");
                ownMaterial = new Material(shader) { color = trailColor };
                if (ownMaterial.HasProperty("_BaseColor")) ownMaterial.SetColor("_BaseColor", trailColor);
            }
            trail.sharedMaterial = trailMaterial != null ? trailMaterial : ownMaterial;
        }
        var lifted = new Vector3[points.Length];
        for (int i = 0; i < points.Length; i++) lifted[i] = points[i] + Vector3.up * trailLift;
        trail.startColor = trail.endColor = trailColor;
        trail.widthMultiplier = trailWidth;
        trail.positionCount = lifted.Length;
        trail.SetPositions(lifted);
        trail.enabled = true;
    }

    void Update()
    {
#if ENABLE_INPUT_SYSTEM
        var kb = Keyboard.current;
        if (keyboardShortcut && kb != null)
        {
            if (kb.gKey.wasPressedThisFrame) TestFind();
            if (kb.hKey.wasPressedThisFrame) TestStart();
            if (kb.jKey.wasPressedThisFrame) TestStop();
        }
#endif
        if (head == null || !head.gameObject.activeInHierarchy)
            head = Camera.main != null ? Camera.main.transform : null;
        if (ActiveRoute != null && head != null) UpdateProgress(false);
    }

    void UpdateProgress(bool snap)
    {
        if (head == null) return;
        var points = ActiveRoute.points;
        Vector2 p = new Vector2(head.position.x, head.position.z);

        // Require a closer match before jumping over 10 m along the trail.
        // Avoid progress jumps where the trail folds back.
        int best = segment;
        float bestScore = float.MaxValue;
        for (int i = 0; i < points.Length - 1; i++)
        {
            float score = SegmentDistance(p, points, i, out _);
            if (!snap && Mathf.Abs(along[i] - along[segment]) > 10) score += segmentSwitchMargin;
            if (score < bestScore) { best = i; bestScore = score; }
        }
        segment = best;
        float lateral = SegmentDistance(p, points, segment, out float t0);
        float s = Mathf.Lerp(along[segment], along[segment + 1], t0);
        float total = along[along.Length - 1];
        DistanceFromTrail = lateral;
        TraveledMeters = s;

        Vector3 end = points[points.Length - 1];
        float toEnd = Vector2.Distance(p, new Vector2(end.x, end.z));
        bool arrived = toEnd <= arrivalMeters || (Status == NavigationStatus.Arrived && toEnd < arrivalMeters * 2);
        if (arrived) Status = NavigationStatus.Arrived;
        else if (lateral > offRouteMeters) Status = NavigationStatus.OffRoute;
        else if (Status != NavigationStatus.OffRoute || lateral < backOnRouteMeters) Status = NavigationStatus.OnRoute;

        if (Status == NavigationStatus.Arrived)
        {
            RemainingMeters = 0;
            NextWaypoint = end;
        }
        else if (Status == NavigationStatus.OffRoute)
        {
            RemainingMeters = total - s + lateral;
            NextWaypoint = Vector3.Lerp(points[segment], points[segment + 1], t0);
        }
        else
        {
            RemainingMeters = total - s;
            NextWaypoint = PointAt(points, Mathf.Min(total, s + lookAheadMeters));
        }
        EtaSeconds = RemainingMeters / Mathf.Max(0.1f, walkingSpeed);
    }

    static float SegmentDistance(Vector2 p, Vector3[] points, int i, out float t)
    {
        Vector2 a = new Vector2(points[i].x, points[i].z), b = new Vector2(points[i + 1].x, points[i + 1].z);
        Vector2 ab = b - a;
        float len = ab.sqrMagnitude;
        t = len > 1e-6f ? Mathf.Clamp01(Vector2.Dot(p - a, ab) / len) : 0;
        return Vector2.Distance(p, a + ab * t);
    }

    Vector3 PointAt(Vector3[] points, float distance)
    {
        for (int i = segment; i < points.Length - 1; i++)
            if (along[i + 1] >= distance)
            {
                float span = along[i + 1] - along[i];
                return Vector3.Lerp(points[i], points[i + 1], span > 1e-4f ? (distance - along[i]) / span : 0);
            }
        return points[points.Length - 1];
    }
}
