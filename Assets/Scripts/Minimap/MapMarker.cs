using System;
using System.Collections.Generic;
using UnityEngine;

// Match the map_locations values in schema.sql.
// Keep lowercase names so ToString() matches the database.
public enum MapLocationType { habitat, rover, sample_site, landmark, crater, hazard, waypoint }
public enum MapHazardLevel { safe, caution, danger }

// Map point fields match the database columns and GET /map response.
[Serializable]
public class MapPoint
{
    public string code;
    public string name;
    public string location_type;
    public float x_m;
    public float z_m;
    public float radius_m;
    public string hazard_level;
    public bool o2_refill;
    public string source;
}

// Add to an empty scene object to mark a location on the minimap.
// Copy map_locations SQL exports these markers with their scene coordinates.
[DisallowMultipleComponent]
public class MapMarker : MonoBehaviour
{
    public static readonly List<MapMarker> All = new List<MapMarker>();

    [Tooltip("Short unique id, the same as map_locations.code (e.g. HAB).")]
    public string code = "HAB";
    public string displayName = "Habitat";
    public MapLocationType locationType = MapLocationType.habitat;
    [Tooltip("Zone radius in meters (map_locations.radius_m). Drawn as a ring on the minimap.")]
    [Min(0.1f)] public float radiusMeters = 15f;
    public MapHazardLevel hazardLevel = MapHazardLevel.safe;
    [Tooltip("Astronaut can refill oxygen here.")]
    public bool o2Refill;
    [TextArea] public string description;

    readonly MapPoint point = new MapPoint();

    void OnEnable() => All.Add(this);
    void OnDisable() => All.Remove(this);

    // Unity x -> x_m, Unity z -> z_m (y is up and not stored).
    public MapPoint ToPoint()
    {
        point.code = code;
        point.name = string.IsNullOrEmpty(displayName) ? code : displayName;
        point.location_type = locationType.ToString();
        point.x_m = transform.position.x;
        point.z_m = transform.position.z;
        point.radius_m = radiusMeters;
        point.hazard_level = hazardLevel.ToString();
        point.o2_refill = o2Refill;
        point.source = "scene";
        return point;
    }

    void OnDrawGizmos()
    {
        Color c = MinimapStyle.ColorFor(locationType.ToString(), hazardLevel.ToString());
        Gizmos.color = c;
        Vector3 p = transform.position;
        Gizmos.DrawLine(p, p + Vector3.up * 20f);   // a pole so it is easy to spot on huge terrain
        Gizmos.DrawSphere(p + Vector3.up * 20f, 1.5f);
#if UNITY_EDITOR
        UnityEditor.Handles.color = c;
        UnityEditor.Handles.DrawWireDisc(p, Vector3.up, radiusMeters);
        UnityEditor.Handles.Label(p + Vector3.up * 23f, $"{code}  {displayName}");
#endif
    }
}

// Shared colors for map markers and scene gizmos.
public static class MinimapStyle
{
    public static Color ColorFor(string locationType, string hazardLevel)
    {
        if (hazardLevel == "danger") return new Color(1f, 0.25f, 0.2f);
        if (hazardLevel == "caution") return new Color(1f, 0.72f, 0.15f);
        switch (locationType)
        {
            case "habitat": return new Color(0.3f, 0.9f, 1f);
            case "rover": return new Color(0.45f, 1f, 0.5f);
            case "sample_site": return new Color(1f, 0.92f, 0.35f);
            case "hazard": return new Color(1f, 0.72f, 0.15f);
            case "waypoint": return new Color(1f, 0.5f, 1f);
            default: return new Color(0.95f, 0.95f, 0.95f);   // landmark, crater, unknown
        }
    }
}
