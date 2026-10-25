using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEditor.Events;
using UnityEngine;

// Editor menu: Tools > Nova > Minimap
//   Add Minimap To Scene                one click setup, then press Play
//   Create Demo Markers Around Player   the schema.sql seed places, moved to where the explorer stands
//   Copy map_locations SQL              scene markers -> SQL for the database (fixes "placeholder coordinates")
public static class NovaMinimapTools
{
    const string Menu = "Tools/Nova/Minimap/";

    // The minimap lives on its own object, not on a camera: ExplorerRigs switches between the desktop
    // and Quest rigs, and the minimap follows whichever camera is active (docking into the Helmet HUD on Quest).
    [MenuItem(Menu + "Add Minimap To Scene")]
    static void AddMinimap()
    {
        foreach (var old in UnityEngine.Object.FindObjectsByType<MarsMinimap>(FindObjectsInactive.Include, FindObjectsSortMode.None))
        {
            if (old.GetComponent<Camera>() == null)
            {
                ConfigurePresentation(old);
                Selection.activeGameObject = old.gameObject;
                Debug.Log($"[Minimap] Already in the scene on '{old.name}'.");
                return;
            }
            // Older setup put it on a camera; move it to its own object.
            var link = old.GetComponent<MinimapApiLink>();
            if (link != null) Undo.DestroyObjectImmediate(link);
            Undo.DestroyObjectImmediate(old);
        }

        var go = new GameObject("Nova Minimap");
        Undo.RegisterCreatedObjectUndo(go, "Add Nova Minimap");
        ConfigurePresentation(Undo.AddComponent<MarsMinimap>(go));
        Selection.activeGameObject = go;
        EditorSceneManager.MarkSceneDirty(go.scene);
        Debug.Log("[Minimap] Added 'Nova Minimap'. Save the scene and press Play. Keys: M show/hide, [ ] zoom, N north-up.");
    }

    static void ConfigurePresentation(MarsMinimap map)
    {
        Undo.RecordObject(map, "Dock minimap in helmet");
        map.hudSlot = new Vector2(430, -245);
        map.hudDiameter = 200;
        var presentation = map.GetComponent<HolographicMinimap>();
        if (presentation == null) presentation = Undo.AddComponent<HolographicMinimap>(map.gameObject);
        var pinch = map.GetComponent<PinchToggle>();
        if (pinch == null) pinch = Undo.AddComponent<PinchToggle>(map.gameObject);
        var settings = new SerializedObject(pinch);
        settings.FindProperty("hand").enumValueIndex = 1;
        settings.FindProperty("helmet").objectReferenceValue =
            UnityEngine.Object.FindAnyObjectByType<HelmetHUD>(FindObjectsInactive.Include);
        settings.ApplyModifiedProperties();
        bool connected = false;
        for (int i = 0; i < pinch.OnPinch.GetPersistentEventCount(); i++)
            connected |= pinch.OnPinch.GetPersistentTarget(i) == presentation &&
                pinch.OnPinch.GetPersistentMethodName(i) == nameof(HolographicMinimap.Toggle);
        if (!connected) UnityEventTools.AddPersistentListener(pinch.OnPinch, presentation.Toggle);
        EditorUtility.SetDirty(map);
        EditorUtility.SetDirty(pinch);
        EditorSceneManager.MarkSceneDirty(map.gameObject.scene);
    }

    // Same rows as the map_locations seed in schema.sql. Offsets are meters from the habitat.
    struct Seed
    {
        public string code, name, description;
        public MapLocationType type;
        public float dx, dz, radius;
        public MapHazardLevel hazard;
        public bool o2;
    }

    static readonly Seed[] Seeds =
    {
        new Seed { code = "HAB",   name = "Habitat",         type = MapLocationType.habitat,     dx = 0,    dz = 0,    radius = 15, hazard = MapHazardLevel.safe,    o2 = true,  description = "Demo base with oxygen refill." },
        new Seed { code = "ROVER", name = "Rover",           type = MapLocationType.rover,       dx = 120,  dz = 40,   radius = 6,  hazard = MapHazardLevel.safe,    o2 = true,  description = "Demo pressurized rover with spare oxygen." },
        new Seed { code = "SITEA", name = "Sample Site A",   type = MapLocationType.sample_site, dx = 450,  dz = -200, radius = 10, hazard = MapHazardLevel.safe,    o2 = false, description = "Layered sediment outcrop." },
        new Seed { code = "CRATR", name = "Crater Rim",      type = MapLocationType.crater,      dx = 900,  dz = 350,  radius = 60, hazard = MapHazardLevel.caution, o2 = false, description = "Steep edges." },
        new Seed { code = "SLOPE", name = "Unstable Slope",  type = MapLocationType.hazard,      dx = 600,  dz = 500,  radius = 80, hazard = MapHazardLevel.danger,  o2 = false, description = "Loose regolith." },
        new Seed { code = "RIDGE", name = "Ridge Overlook",  type = MapLocationType.landmark,    dx = 1500, dz = -100, radius = 20, hazard = MapHazardLevel.safe,    o2 = false, description = "Demo observation point." },
    };

    [MenuItem(Menu + "Create Demo Markers Around Player")]
    static void CreateDemoMarkers()
    {
        // Mars Explorer has a CharacterController; fall back to the camera (no hard dependency on MarsExplorer.cs).
        Transform player = null;
        var walker = UnityEngine.Object.FindAnyObjectByType<CharacterController>();
        if (walker != null) player = walker.transform;
        else if (Camera.main != null) player = Camera.main.transform;
        if (player == null)
        {
            EditorUtility.DisplayDialog("Nova Minimap", "Could not find Mars Explorer or a Main Camera to place markers around.", "OK");
            return;
        }

        var existing = new HashSet<string>(AllMarkers().Select(m => m.code));
        GameObject parent = GameObject.Find("Map Markers");
        if (parent == null)
        {
            parent = new GameObject("Map Markers");
            Undo.RegisterCreatedObjectUndo(parent, "Create Map Markers");
        }

        int made = 0;
        foreach (var s in Seeds)
        {
            if (existing.Contains(s.code)) continue;
            float x = player.position.x + s.dx;
            float z = player.position.z + s.dz;
            float y = GroundHeight(x, z) ?? player.position.y;

            var go = new GameObject($"{s.code} - {s.name}");
            Undo.RegisterCreatedObjectUndo(go, "Create Map Marker");
            go.transform.SetParent(parent.transform, true);
            go.transform.position = new Vector3(x, y, z);
            var m = go.AddComponent<MapMarker>();
            m.code = s.code;
            m.displayName = s.name;
            m.locationType = s.type;
            m.radiusMeters = s.radius;
            m.hazardLevel = s.hazard;
            m.o2Refill = s.o2;
            m.description = s.description;
            made++;
        }

        Selection.activeGameObject = parent;
        EditorSceneManager.MarkSceneDirty(parent.scene);
        Debug.Log($"[Minimap] Created {made} markers around ({player.position.x:0}, {player.position.z:0}). " +
                  "Drag them onto real features, then use Copy map_locations SQL.");
    }

    [MenuItem(Menu + "Copy map_locations SQL")]
    static void CopySql()
    {
        var markers = AllMarkers().Where(m => !string.IsNullOrWhiteSpace(m.code)).OrderBy(m => m.code).ToList();
        if (markers.Count == 0)
        {
            EditorUtility.DisplayDialog("Nova Minimap", "No MapMarker objects in the open scene.", "OK");
            return;
        }
        var dupes = markers.GroupBy(m => m.code).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
        if (dupes.Count > 0)
        {
            EditorUtility.DisplayDialog("Nova Minimap", "Duplicate codes (must be unique): " + string.Join(", ", dupes), "OK");
            return;
        }

        var inv = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();
        sb.AppendLine($"-- Generated from the Unity scene ({DateTime.Now:yyyy-MM-dd HH:mm}) by Tools > Nova > Minimap > Copy map_locations SQL.");
        sb.AppendLine("-- x_m = Unity x, z_m = Unity z (1 unit = 1 m). Replace the map_locations INSERT in schema.sql with this,");
        sb.AppendLine("-- or run it once with psql: ON CONFLICT updates rows that already exist.");
        sb.AppendLine("INSERT INTO map_locations (");
        sb.AppendLine("    code, name, location_type, x_m, z_m, radius_m, hazard_level, o2_refill, description");
        sb.AppendLine(")");
        sb.AppendLine("VALUES");
        for (int i = 0; i < markers.Count; i++)
        {
            var m = markers[i];
            Vector3 p = m.transform.position;
            sb.Append("    (")
              .Append(Q(m.code)).Append(", ")
              .Append(Q(string.IsNullOrWhiteSpace(m.displayName) ? m.code : m.displayName)).Append(", ")
              .Append(Q(m.locationType.ToString())).Append(", ")
              .Append(p.x.ToString("0.##", inv)).Append(", ")
              .Append(p.z.ToString("0.##", inv)).Append(", ")
              .Append(m.radiusMeters.ToString("0.##", inv)).Append(", ")
              .Append(Q(m.hazardLevel.ToString())).Append(", ")
              .Append(m.o2Refill ? "TRUE" : "FALSE").Append(", ")
              .Append(string.IsNullOrWhiteSpace(m.description) ? "NULL" : Q(m.description.Trim()))
              .AppendLine(i < markers.Count - 1 ? ")," : ")");
        }
        sb.AppendLine("ON CONFLICT (code) DO UPDATE SET");
        sb.AppendLine("    name = EXCLUDED.name,");
        sb.AppendLine("    location_type = EXCLUDED.location_type,");
        sb.AppendLine("    x_m = EXCLUDED.x_m,");
        sb.AppendLine("    z_m = EXCLUDED.z_m,");
        sb.AppendLine("    radius_m = EXCLUDED.radius_m,");
        sb.AppendLine("    hazard_level = EXCLUDED.hazard_level,");
        sb.AppendLine("    o2_refill = EXCLUDED.o2_refill,");
        sb.AppendLine("    description = EXCLUDED.description;");

        EditorGUIUtility.systemCopyBuffer = sb.ToString();
        Debug.Log($"[Minimap] Copied SQL for {markers.Count} markers to the clipboard:\n{sb}");
    }

    static IEnumerable<MapMarker> AllMarkers() =>
        UnityEngine.Object.FindObjectsByType<MapMarker>(FindObjectsInactive.Include, FindObjectsSortMode.None);

    static string Q(string s) => "'" + s.Replace("'", "''") + "'";

    // Height of the terrain at (x, z), or null if there is no terrain there.
    static float? GroundHeight(float x, float z)
    {
        foreach (var t in Terrain.activeTerrains)
        {
            if (t == null || t.terrainData == null) continue;
            Vector3 p = t.GetPosition();
            Vector3 s = t.terrainData.size;
            if (x < p.x || x > p.x + s.x || z < p.z || z > p.z + s.z) continue;
            return p.y + t.SampleHeight(new Vector3(x, 0f, z));
        }
        return null;
    }
}
