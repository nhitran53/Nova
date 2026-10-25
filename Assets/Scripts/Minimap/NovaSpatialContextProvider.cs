using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;

// Capture once per voice request. The tracked camera includes physical movement.
public class NovaSpatialContextProvider : MonoBehaviour
{
    [Serializable] public class Landmark
    {
        public string id, name, direction, relative_direction;
        public Vector3 position;
        public float distance_m, bearing_deg;
    }
    [Serializable] public class Navigation
    {
        public string destination_id, status;
        public bool active, auto_navigation;
        public float remaining_route_m;
    }
    [Serializable] public class Snapshot
    {
        public int schema_version = 1, seq;
        public bool available, coordinates_verified;
        public string error, environment, region, coordinate_reference;
        public double latitude, longitude;
        public Vector3 world_position, head_position, head_forward, world_north;
        public Landmark[] nearby_landmarks = Array.Empty<Landmark>();
        public Navigation navigation = new Navigation();
    }

    VoiceSessionClient voice;
    static RouteNavigator Navigator => RouteNavigator.Current != null ? RouteNavigator.Current : FindAnyObjectByType<RouteNavigator>();
    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Install()
    {
        var client = FindAnyObjectByType<VoiceSessionClient>(FindObjectsInactive.Include);
        if (client != null && client.GetComponent<NovaSpatialContextProvider>() == null)
            client.gameObject.AddComponent<NovaSpatialContextProvider>();
    }
    void Awake() => voice = GetComponent<VoiceSessionClient>();
    void OnEnable() { if (voice != null) voice.OnSpatialRequest.AddListener(Respond); }
    void OnDisable() { if (voice != null) voice.OnSpatialRequest.RemoveListener(Respond); }
    void Respond(string unused)
    {
        var snapshot = Capture();
        snapshot.seq = voice.SpatialSeq;
        voice.SendSpatialContext(JsonUtility.ToJson(snapshot));
    }

    [ContextMenu("Log Current Location")]
    public void LogCurrentLocation() => Debug.Log(JsonUtility.ToJson(Capture(), true), this);

    public Snapshot Capture()
    {
        var result = new Snapshot();
        var camera = Camera.main;
        if (camera == null || !camera.isActiveAndEnabled)
        { result.error = "No active tracked camera."; return result; }
        if (XRSettings.isDeviceActive)
        {
            var device = InputDevices.GetDeviceAtXRNode(XRNode.Head);
            if (!device.TryGetFeatureValue(CommonUsages.isTracked, out bool tracked) || !tracked)
            { result.error = "Headset tracking is unavailable."; return result; }
        }
        result.head_position = camera.transform.position;
        result.head_forward = camera.transform.forward;
        var head = result.head_position;
        Terrain ground = null;
        Vector3 point = head;
        float best = float.PositiveInfinity;
        foreach (var terrain in Terrain.activeTerrains)
        {
            var data = terrain.terrainData;
            if (data == null) continue;
            var local = terrain.transform.InverseTransformPoint(head);
            if (local.x < 0 || local.z < 0 || local.x >= data.size.x || local.z >= data.size.z) continue;
            int x = Mathf.Clamp((int)(local.x / data.size.x * data.holesResolution), 0, data.holesResolution - 1);
            int z = Mathf.Clamp((int)(local.z / data.size.z * data.holesResolution), 0, data.holesResolution - 1);
            if (data.IsHole(x, z)) continue;
            float height = data.GetInterpolatedHeight(local.x / data.size.x, local.z / data.size.z);
            var surface = terrain.transform.TransformPoint(new Vector3(local.x, height, local.z));
            float delta = Mathf.Abs(head.y - surface.y);
            if (delta >= best) continue;
            if (surface.y > head.y + .2f) continue;
            best = delta; ground = terrain; point = new Vector3(head.x, surface.y, head.z);
        }
        if (ground == null) { result.error = "No terrain surface below the headset."; return result; }
        result.available = true;
        result.world_position = point;
        var geo = ground.GetComponentInParent<JezeroGeoreference>();
        result.world_north = geo != null ? geo.North : Vector3.forward;
        result.coordinates_verified = geo != null && geo.TryCoordinates(point, out result.latitude, out result.longitude);
        if (result.coordinates_verified)
        {
            result.environment = "Mars"; result.region = "Jezero Crater area";
            result.coordinate_reference = "GCS_Mars_2000_Sphere spherical latitude; east-positive longitude 0 to 360 degrees";
        }
        else result.error = "Verified terrain georeferencing is unavailable.";
        result.nearby_landmarks = GetNearbyLandmarks(point, result.head_forward, result.world_north);
        var nav = Navigator;
        if (nav != null)
        {
            result.navigation.status = nav.Status.ToString();
            result.navigation.active = nav.Navigating;
            result.navigation.auto_navigation = nav.AutoNavigating;
            result.navigation.destination_id = nav.ActiveRoute != null ? nav.ActiveRoute.destinationId : null;
            result.navigation.remaining_route_m = nav.Navigating ? nav.RemainingMeters : 0;
        }
        return result;
    }

    public static Landmark Describe(string id, string name, Vector3 target, Vector3 player, Vector3 forward, Vector3 north)
    {
        var delta = Vector3.ProjectOnPlane(target - player, Vector3.up);
        float bearing = Mathf.Repeat(Vector3.SignedAngle(north, delta, Vector3.up), 360);
        var horizontalForward = Vector3.ProjectOnPlane(forward, Vector3.up);
        float relative = Vector3.SignedAngle(horizontalForward, delta, Vector3.up);
        string[] compass = { "north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest" };
        return new Landmark { id = id, name = name, position = target, distance_m = delta.magnitude, bearing_deg = bearing,
            direction = compass[Mathf.RoundToInt(bearing / 45) % 8], relative_direction = horizontalForward.sqrMagnitude < .000001f ? "unavailable" :
                Mathf.Abs(relative) <= 45 ? "in front of you" : Mathf.Abs(relative) >= 135 ? "behind you" :
                relative > 0 ? "to your right" : "to your left" };
    }

    public Landmark[] GetNearbyLandmarks(Vector3 player, Vector3 forward, Vector3 north)
    {
        var points = new Dictionary<string, Landmark>();
        void Add(string id, string name, Vector3 position)
        {
            if (!string.IsNullOrWhiteSpace(id) && !points.ContainsKey(id)) points[id] = Describe(id, name, position, player, forward, north);
        }
        foreach (var marker in FindObjectsByType<MapMarker>(FindObjectsSortMode.None))
            if (marker != null && marker.isActiveAndEnabled) Add(marker.code, marker.ToPoint().name, marker.transform.position);
        var nav = Navigator;
        if (nav != null) foreach (var destination in nav.Destinations)
            Add(destination.code, destination.name, new Vector3(destination.x_m, player.y, destination.z_m));
        var rover = FindAnyObjectByType<RoverDamageState>();
        if (rover != null)
        {
            Add("rover", rover.gameObject.name, rover.transform.position);
            if (rover.FrontRightWheel != null)
            {
                var wheel = rover.FrontRightWheel.GetComponent<Renderer>();
                Add("rover_front_right_wheel", "rover front right wheel", wheel != null ? wheel.bounds.center : rover.FrontRightWheel.position);
            }
            var spawn = rover.GetComponent<RoverInspectionSpawn>();
            if (spawn != null && spawn.inspectionAnchor != null) Add("rover_inspection", "rover inspection area", spawn.inspectionAnchor.position);
        }
        var sorted = new List<Landmark>(points.Values);
        sorted.Sort((a, b) => a.distance_m.CompareTo(b.distance_m));
        return sorted.GetRange(0, Mathf.Min(sorted.Count, 32)).ToArray();
    }
}
