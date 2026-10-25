using System.Collections.Generic;
using UnityEngine;
using Unity.XR.CoreUtils;
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit.Inputs.Readers;
using UnityEngine.XR.Interaction.Toolkit.Locomotion.Movement;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public enum NavigationStatus { Idle, Preview, OnRoute, OffRoute, Arrived, Stopped }

// Build the trail once. Share progress through Current.
[DefaultExecutionOrder(-150)]
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

    [Header("Demo auto movement")]
    [Tooltip("Demo movement only. This is not validated autonomous traversal.")]
    [Min(0.1f)] public float demoAutoMoveSpeed = 15;
    [Min(0.1f)] public float demoAcceleration = 8;
    [Tooltip("Demo movement can start this far from the trail; it walks straight onto the trail first.")]
    [SerializeField, Min(1)] float demoJoinMeters = 60;

    [Header("Location waypoint")]
    public bool showWaypointBeacon = true;
    [SerializeField, Min(5)] float waypointBeaconHeight = 120;
    [SerializeField, Min(.05f)] float waypointBeaconWidth = .5f;

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
    public bool AutoNavigating { get; private set; }
    public MapMarker Waypoint { get; private set; }
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
    XROrigin autoOrigin;
    ContinuousMoveProvider autoMove;
    FistLocomotion autoFist;
    XRInputValueReader.InputSourceMode leftMode, rightMode;
    Vector2 leftValue, rightValue;
    Transform savedForward;
    float savedSpeed, savedAirControl;
    bool savedStrafe, resumeFist, requireTracking;
    int autoWaypoint;
    float autoSpeed, stalledSeconds;
    bool joiningTrail;
    Vector3 lastAutoPosition;
    Vector3[] autoPoints;
    GameObject beacon;
    LineRenderer beaconCore, beaconGlow;
    Material beaconMaterial;
    static readonly Color WaypointCyan = new Color(.2f, .9f, 1);

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
        EndAutoMovement();
        if (beacon != null) beacon.SetActive(false);
    }

    // Map requests mark a location. Start requests move along the route.
    void OnVoiceNavigation(string destinationId)
    {
        if (voice.NavigationAction == "return_to_start")
        {
            StopNavigation();
            var spawn = FindAnyObjectByType<RoverInspectionSpawn>();
            if (spawn != null) spawn.ResetInspectionPosition();
            else Debug.LogWarning("The rover inspection spawn is missing. Cannot return to start.", this);
        }
        else if (voice.NavigationAction == "start") StartDemoNavigation(destinationId);
        else if (voice.NavigationAction == "find") ShowWaypoint(destinationId);
    }

    void OnDestroy()
    {
        if (Current == this) Current = null;
        if (ownMaterial != null) Destroy(ownMaterial);
        if (beaconMaterial != null) Destroy(beaconMaterial);
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
        EndAutoMovement();
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
            case "start_demo_navigation": response = StartDemoNavigation(args.destination_id); break;
            case "stop_navigation": response = StopNavigation(); break;
            case "show_waypoint": response = ShowWaypoint(args.destination_id); break;
            case "clear_waypoint": response = ClearWaypoint(); break;
            default: response = Service.Describe(null, $"error: unknown tool '{tool}'"); break;
        }
        return JsonUtility.ToJson(response);
    }

    [ContextMenu("Tools/find_navigation_route (test request)")]
    public void TestFind() => FindNavigationRoute(testRequest);

    [ContextMenu("Tools/start_navigation")]
    public void TestStart() => StartNavigation(null);

    [ContextMenu("Demo/Auto navigate to shuttle")]
    public void TestAutoStart() => StartDemoNavigation("shuttle_001");

    [ContextMenu("Tools/stop_navigation")]
    public void TestStop() => StopNavigation();

    public NavigationResponse ShowWaypoint(string destinationId)
    {
        MapMarker target = null;
        foreach (var marker in MapMarker.All)
            if (marker != null && marker.code == destinationId) { target = marker; break; }
        if (target == null)
            return new NavigationResponse { status = "Location marker not found", destination_id = destinationId };
        Waypoint = target;
        var map = FindAnyObjectByType<MarsMinimap>();
        if (map != null)
        {
            var panel = map.GetComponent<MinimapNavigationPanel>();
            if (panel != null) panel.SelectDestination(destinationId);
            else map.SelectedCode = destinationId;
            map.FocusLocation(target.transform.position);
            var hologram = map.GetComponent<HolographicMinimap>();
            if (hologram != null && hologram.RightHandTracked && !map.Holographic) hologram.Toggle();
        }
        UpdateBeacon();
        return new NavigationResponse { success = true, destination_id = destinationId, route_type = "location", status = "waypoint_shown" };
    }

    public NavigationResponse ClearWaypoint()
    {
        string code = Waypoint != null ? Waypoint.code : null;
        Waypoint = null;
        if (beacon != null) beacon.SetActive(false);
        var map = FindAnyObjectByType<MarsMinimap>();
        if (map != null && map.SelectedCode == code)
        {
            var panel = map.GetComponent<MinimapNavigationPanel>();
            if (panel != null) panel.SelectDestination(null);
            else map.SelectedCode = null;
        }
        return new NavigationResponse { success = true, route_type = "location", status = "waypoint_cleared" };
    }

    [ContextMenu("Tools/Clear Waypoint")]
    public void TestClearWaypoint() => ClearWaypoint();

    void LateUpdate() => UpdateBeacon();

    void UpdateBeacon()
    {
        bool visible = Waypoint != null && showWaypointBeacon;
        if (!visible)
        {
            if (beacon != null) beacon.SetActive(false);
            return;
        }
        if (beacon == null)
        {
            var shader = Shader.Find("Nova/Visor UI");
            if (shader == null) return;
            beaconMaterial = new Material(shader) { renderQueue = 3100 };
            beacon = new GameObject("Location Beacon");
            beacon.transform.SetParent(transform, false);
            beaconGlow = CreateBeam("Glow");
            beaconCore = CreateBeam("Core");
        }
        beacon.SetActive(true);
        var bottom = Waypoint.transform.position + Vector3.up * .1f;
        var top = bottom + Vector3.up * Mathf.Max(5, waypointBeaconHeight);
        float pulse = .85f + .15f * Mathf.Sin(Time.unscaledTime * 2);
        SetBeam(beaconCore, bottom, top, waypointBeaconWidth, .9f * pulse);
        SetBeam(beaconGlow, bottom, top, waypointBeaconWidth * 5, .12f * pulse);
    }

    LineRenderer CreateBeam(string name)
    {
        var go = new GameObject(name);
        go.transform.SetParent(beacon.transform, false);
        var line = go.AddComponent<LineRenderer>();
        line.sharedMaterial = beaconMaterial;
        line.useWorldSpace = true;
        line.positionCount = 2;
        line.alignment = LineAlignment.View;
        line.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        line.receiveShadows = false;
        line.lightProbeUsage = UnityEngine.Rendering.LightProbeUsage.Off;
        line.reflectionProbeUsage = UnityEngine.Rendering.ReflectionProbeUsage.Off;
        line.allowOcclusionWhenDynamic = false;
        return line;
    }

    static void SetBeam(LineRenderer line, Vector3 bottom, Vector3 top, float width, float alpha)
    {
        line.SetPosition(0, bottom);
        line.SetPosition(1, top);
        line.widthMultiplier = Mathf.Max(.05f, width);
        line.startColor = new Color(WaypointCyan.r, WaypointCyan.g, WaypointCyan.b, alpha);
        line.endColor = new Color(WaypointCyan.r, WaypointCyan.g, WaypointCyan.b, alpha * .3f);
    }

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
        EndAutoMovement();
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
        if (AutoNavigating && (head == null || ActiveRoute == null)) StopNavigation();
        if (ActiveRoute != null && head != null)
        {
            if (!ActiveRoute.IsBuilt || along == null || along.Length != ActiveRoute.points.Length)
            {
                StopNavigation();
                return;
            }
            UpdateProgress(false);
            if (AutoNavigating) UpdateAutoMovement(Time.deltaTime);
        }
    }

    void UpdateProgress(bool snap)
    {
        if (head == null) return;
        var points = ActiveRoute.points;
        Vector2 p = new Vector2(head.position.x, head.position.z);

        // Require a closer match before jumping over 10 m along the trail.
        // Avoid progress jumps where the trail folds back.
        int best = AutoNavigating ? Mathf.Clamp(autoWaypoint - 1, 0, points.Length - 2) : segment;
        float bestScore = float.MaxValue;
        for (int i = 0; !AutoNavigating && i < points.Length - 1; i++)
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
        EtaSeconds = RemainingMeters / Mathf.Max(0.1f, AutoNavigating ? demoAutoMoveSpeed : walkingSpeed);
    }

    public NavigationResponse StartDemoNavigation(string destinationId = "shuttle_001")
    {
        var response = FindNavigationRoute(new NavigationRequest { destination_id = destinationId });
        if (!response.success) return response;
        bool alreadyMoving = AutoNavigating && ActiveRoute != null && response.route_id == ActiveRoute.routeId;
        if (alreadyMoving) return response;
        if (head == null) head = Camera.main != null ? Camera.main.transform : null;
        var origin = head != null ? head.GetComponentInParent<XROrigin>() : null;
        var move = origin != null ? origin.GetComponentInChildren<ContinuousMoveProvider>() : null;
        var controller = origin != null ? origin.Origin.GetComponent<CharacterController>() : null;
        if (move == null || !move.isActiveAndEnabled || controller == null || !controller.enabled || move.enableFly ||
            (!Application.isEditor && !XRSettings.isDeviceActive))
        {
            response.success = false;
            response.status = "Demo movement needs the active Quest XR Origin and its grounded move provider";
            return Log("start_demo_navigation", new NavigationRequest { destination_id = destinationId }, response);
        }
        response = StartNavigation(response.route_id);
        UpdateProgress(true);
        if (Status == NavigationStatus.Arrived) return response;
        if (DistanceFromTrail > demoJoinMeters)
        {
            response.success = false;
            response.status = $"Move within {demoJoinMeters:0} m of the saved trail before starting demo movement";
            return Log("start_demo_navigation", new NavigationRequest { destination_id = destinationId }, response);
        }
        foreach (var point in ActiveRoute.points)
            if (!float.IsFinite(point.x) || !float.IsFinite(point.y) || !float.IsFinite(point.z))
            {
                StopNavigation();
                response.success = false;
                response.status = "The saved route contains an invalid waypoint";
                return response;
            }
        autoOrigin = origin;
        autoMove = move;
        autoFist = origin.GetComponent<FistLocomotion>();
        resumeFist = autoFist != null && autoFist.enabled;
        if (resumeFist) autoFist.enabled = false;
        leftMode = move.leftHandMoveInput.inputSourceMode;
        rightMode = move.rightHandMoveInput.inputSourceMode;
        leftValue = move.leftHandMoveInput.manualValue;
        rightValue = move.rightHandMoveInput.manualValue;
        savedForward = move.forwardSource;
        savedSpeed = move.moveSpeed;
        savedAirControl = move.inAirControlModifier;
        savedStrafe = move.enableStrafe;
        move.leftHandMoveInput.inputSourceMode = XRInputValueReader.InputSourceMode.ManualValue;
        move.rightHandMoveInput.inputSourceMode = XRInputValueReader.InputSourceMode.ManualValue;
        move.leftHandMoveInput.manualValue = move.rightHandMoveInput.manualValue = Vector2.zero;
        move.forwardSource = origin.Origin.transform;
        move.inAirControlModifier = 1;
        move.enableStrafe = true;
        autoPoints = ActiveRoute.points;
        autoWaypoint = segment + 1;
        autoSpeed = stalledSeconds = 0;
        lastAutoPosition = origin.Origin.transform.position;
        requireTracking = XRSettings.isDeviceActive;
        joiningTrail = DistanceFromTrail > backOnRouteMeters;
        AutoNavigating = true;
        response.status = "demo_auto_navigation";
        return response;
    }

    void UpdateAutoMovement(float deltaTime)
    {
        var headset = InputDevices.GetDeviceAtXRNode(XRNode.Head);
        bool editorPreview = !requireTracking && Application.isEditor;
        bool tracked = editorPreview || (XRSettings.isDeviceActive &&
            headset.TryGetFeatureValue(UnityEngine.XR.CommonUsages.isTracked, out bool isTracked) && isTracked);
        var left = InputDevices.GetDeviceAtXRNode(XRNode.LeftHand);
        bool manualMove = left.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primary2DAxis, out Vector2 stick) && stick.sqrMagnitude > .04f;
        if (autoOrigin == null || autoMove == null || !autoMove.isActiveAndEnabled || head == null ||
            autoPoints != ActiveRoute.points || !tracked || manualMove)
        {
            StopNavigation();
            return;
        }
        if (Status == NavigationStatus.Arrived || RemainingMeters <= arrivalMeters + .1f)
        {
            Status = NavigationStatus.Arrived;
            RemainingMeters = 0;
            EndAutoMovement();
            return;
        }
        if (joiningTrail && DistanceFromTrail <= backOnRouteMeters) joiningTrail = false;
        if ((!joiningTrail && DistanceFromTrail > offRouteMeters) || !Physics.Raycast(head.position, Vector3.down, out _,
                Mathf.Max(1, head.position.y - autoOrigin.Origin.transform.position.y) + .75f, ~4, QueryTriggerInteraction.Ignore))
        {
            StopNavigation();
            return;
        }
        var position = autoOrigin.Origin.transform.position;
        float moved = Vector3.ProjectOnPlane(position - lastAutoPosition, Vector3.up).magnitude;
        stalledSeconds = autoSpeed > .5f && moved < .002f ? stalledSeconds + deltaTime : 0;
        lastAutoPosition = position;
        if (stalledSeconds > 1)
        {
            StopNavigation();
            Debug.LogWarning("Demo navigation stopped at an obstruction.", this);
            return;
        }
        Vector3 target;
        if (DistanceFromTrail > .2f)
        {
            SegmentDistance(new Vector2(head.position.x, head.position.z), autoPoints, segment, out float t);
            target = Vector3.Lerp(autoPoints[segment], autoPoints[segment + 1], t);
            autoWaypoint = segment + 1;
        }
        else
        {
            autoWaypoint = Mathf.Max(autoWaypoint, segment + 1);
            target = autoPoints[Mathf.Min(autoWaypoint, autoPoints.Length - 1)];
            if (Vector3.ProjectOnPlane(target - head.position, Vector3.up).magnitude < .08f && autoWaypoint < autoPoints.Length - 1)
                target = autoPoints[++autoWaypoint];
        }
        var delta = Vector3.ProjectOnPlane(target - head.position, Vector3.up);
        if (!float.IsFinite(delta.sqrMagnitude) || !float.IsFinite(RemainingMeters))
        {
            StopNavigation();
            return;
        }
        float dt = Mathf.Min(deltaTime, .05f);
        float brakingSpeed = Mathf.Sqrt(2 * demoAcceleration * Mathf.Max(0, RemainingMeters - arrivalMeters));
        autoSpeed = Mathf.MoveTowards(autoSpeed, Mathf.Min(demoAutoMoveSpeed, brakingSpeed), demoAcceleration * dt);
        float step = Mathf.Min(delta.magnitude, autoSpeed * dt);
        autoMove.moveSpeed = Mathf.Max(.1f, autoSpeed);
        var local = autoOrigin.Origin.transform.InverseTransformDirection(delta.normalized);
        float input = step / Mathf.Max(.0001f, autoMove.moveSpeed * deltaTime * autoOrigin.Origin.transform.localScale.x);
        autoMove.leftHandMoveInput.manualValue = new Vector2(local.x, local.z) * input;
    }

    void EndAutoMovement()
    {
        if (!AutoNavigating) return;
        AutoNavigating = false;
        if (autoMove != null)
        {
            autoMove.transformation.motion = Vector3.zero;
            autoMove.moveSpeed = savedSpeed;
            autoMove.forwardSource = savedForward;
            autoMove.inAirControlModifier = savedAirControl;
            autoMove.enableStrafe = savedStrafe;
            autoMove.leftHandMoveInput.manualValue = leftValue;
            autoMove.rightHandMoveInput.manualValue = rightValue;
            autoMove.leftHandMoveInput.inputSourceMode = leftMode;
            autoMove.rightHandMoveInput.inputSourceMode = rightMode;
            bool wasEnabled = autoMove.enabled;
            autoMove.enabled = false;
            autoMove.enabled = wasEnabled;
        }
        if (resumeFist && autoFist != null) autoFist.enabled = true;
        autoSpeed = 0;
        autoPoints = null;
    }

    void OnApplicationFocus(bool focused)
    {
        if (!focused && AutoNavigating) StopNavigation();
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
