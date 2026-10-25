using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Unity.XR.CoreUtils;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Build routes and add scene navigation from Tools > Nova > Navigation.
public static class NovaNavigationTools
{
    const string Menu = "Tools/Nova/Navigation/";
    const string Folder = "Assets/Scripts/Navigation/Routes";
    const string RoutePath = Folder + "/route_shuttle_001.asset";
    const string TrailMaterialPath = "Assets/Scripts/Navigation/Route Trail.mat";
    const string ShuttleModelPath = "Assets/Shuttle/Generated/Shuttle.fbx";
    const string TerrainName = "HiRISE Test Area";

    [MenuItem(Menu + "Rebuild Shuttle Route")]
    static void RebuildShuttleRoute()
    {
        var route = LoadOrCreateRoute();
        Build(route);
        Selection.activeObject = route;
    }

    [MenuItem(Menu + "Add Navigation To Scene")]
    static void AddNavigation()
    {
        var route = LoadOrCreateRoute();
        if (!route.IsBuilt) Build(route);

        var navigator = UnityEngine.Object.FindAnyObjectByType<RouteNavigator>(FindObjectsInactive.Include);
        if (navigator == null)
        {
            var go = new GameObject("Nova Navigation");
            Undo.RegisterCreatedObjectUndo(go, "Add Nova Navigation");
            navigator = Undo.AddComponent<RouteNavigator>(go);
        }
        if (navigator.GetComponent<NavigationHud>() == null) Undo.AddComponent<NavigationHud>(navigator.gameObject);
        var settings = new SerializedObject(navigator);
        var routes = settings.FindProperty("routes");
        bool listed = false;
        for (int i = 0; i < routes.arraySize; i++) listed |= routes.GetArrayElementAtIndex(i).objectReferenceValue == route;
        if (!listed)
        {
            routes.arraySize++;
            routes.GetArrayElementAtIndex(routes.arraySize - 1).objectReferenceValue = route;
        }
        settings.FindProperty("trailMaterial").objectReferenceValue = TrailMaterial();
        settings.FindProperty("testRequest").FindPropertyRelative("destination_id").stringValue = route.destinationId;
        settings.ApplyModifiedProperties();

        var marker = MapMarker.All.FirstOrDefault(m => m.code == route.destinationId) ??
            UnityEngine.Object.FindObjectsByType<MapMarker>(FindObjectsInactive.Include, FindObjectsSortMode.None).FirstOrDefault(m => m.code == route.destinationId);
        if (marker == null)
        {
            var go = new GameObject(route.destinationId);
            Undo.RegisterCreatedObjectUndo(go, "Add shuttle marker");
            marker = Undo.AddComponent<MapMarker>(go);
            marker.code = route.destinationId;
            marker.displayName = route.destinationName;
            marker.locationType = MapLocationType.landmark;
            marker.radiusMeters = 6;
            marker.description = "Viking lander used as the demo shuttle.";
        }
        Undo.RecordObject(marker.transform, "Move shuttle marker");
        marker.transform.position = route.Destination;
        AddShuttleModel(marker);

        var minimap = UnityEngine.Object.FindAnyObjectByType<MarsMinimap>(FindObjectsInactive.Include);
        if (minimap != null && minimap.GetComponent<MinimapNavigationPanel>() == null)
            Undo.AddComponent<MinimapNavigationPanel>(minimap.gameObject);
        else if (minimap == null)
            Debug.LogWarning("[Navigation] No Nova Minimap in the scene. Run Tools > Nova > Minimap > Add Minimap To Scene, then this again.");

        foreach (var origin in UnityEngine.Object.FindObjectsByType<XROrigin>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (origin.gameObject.scene == SceneManager.GetActiveScene() && origin.GetComponent<FistLocomotion>() == null)
                Undo.AddComponent<FistLocomotion>(origin.gameObject);

        Selection.activeGameObject = navigator.gameObject;
        EditorSceneManager.MarkSceneDirty(navigator.gameObject.scene);
        Debug.Log("[Navigation] Added. Save the scene and press Play. Nothing starts on its own: pick the shuttle on the map, " +
            "or on desktop G = find route, H = start, J = stop. Quest: hold a right fist to walk.");
    }

    static void AddShuttleModel(MapMarker marker)
    {
        var model = AssetDatabase.LoadAssetAtPath<GameObject>(ShuttleModelPath);
        if (model == null)
        {
            Debug.LogWarning($"[Navigation] No shuttle model at {ShuttleModelPath}; the shuttle marker stays empty.");
            return;
        }
        if (marker.transform.Find(model.name) != null) return;
        var shuttle = (GameObject)PrefabUtility.InstantiatePrefab(model, marker.transform);
        Undo.RegisterCreatedObjectUndo(shuttle, "Add shuttle model");
        shuttle.transform.localPosition = Vector3.zero;
        shuttle.transform.localRotation = Quaternion.identity;
        Undo.RecordObject(marker, "Describe shuttle marker");
        marker.description = "Viking lander used as the demo shuttle.";
    }

    static NavigationRoute LoadOrCreateRoute()
    {
        var route = AssetDatabase.LoadAssetAtPath<NavigationRoute>(RoutePath);
        if (route != null) return route;
        Directory.CreateDirectory(Folder);
        route = ScriptableObject.CreateInstance<NavigationRoute>();
        AssetDatabase.CreateAsset(route, RoutePath);
        return route;
    }

    static Material TrailMaterial()
    {
        var material = AssetDatabase.LoadAssetAtPath<Material>(TrailMaterialPath);
        if (material != null) return material;
        var shader = Shader.Find("Universal Render Pipeline/Unlit");
        if (shader == null) return null;   // Use the runtime fallback material.
        material = new Material(shader) { name = "Route Trail" };
        var cyan = new Color(0.25f, 0.9f, 1f);
        material.SetColor("_BaseColor", cyan);
        material.color = cyan;
        AssetDatabase.CreateAsset(material, TrailMaterialPath);
        return material;
    }

    struct Grid
    {
        public int n;
        public float dx, dz;
        public Vector3 origin;   // Terrain southwest corner in world space
        public float[] height;   // World heights ordered from south to north
        public float[] slope;    // Degrees
        public float[] penalty;  // Cell cost multiplier
        public bool[] blocked;

        public int Cell(float x, float z)
        {
            int i = Mathf.RoundToInt((x - origin.x) / dx), j = Mathf.RoundToInt((z - origin.z) / dz);
            return i < 0 || j < 0 || i >= n || j >= n ? -1 : j * n + i;
        }

        public Vector2 XZ(int cell) => new Vector2(origin.x + cell % n * dx, origin.z + cell / n * dz);

        public float Height(float x, float z)
        {
            float u = Mathf.Clamp((x - origin.x) / dx, 0, n - 1.001f), v = Mathf.Clamp((z - origin.z) / dz, 0, n - 1.001f);
            int i = (int)u, j = (int)v;
            float fu = u - i, fv = v - j;
            int k = j * n + i;
            return Mathf.Lerp(Mathf.Lerp(height[k], height[k + 1], fu), Mathf.Lerp(height[k + n], height[k + n + 1], fu), fv);
        }
    }

    public static void Build(NavigationRoute route)
    {
        var terrain = UnityEngine.Object.FindObjectsByType<Terrain>(FindObjectsInactive.Include, FindObjectsSortMode.None)
            .FirstOrDefault(t => t.name == TerrainName && t.gameObject.scene == SceneManager.GetActiveScene());
        if (terrain == null) throw new InvalidOperationException($"Open the scene with the '{TerrainName}' terrain first.");

        if (route.startAtSpawn)
        {
            var spawn = FindSpawn();
            if (spawn.HasValue) route.startXZ = new Vector2(spawn.Value.x, spawn.Value.z);
            else Debug.LogWarning("[Navigation] No XR Origin found, using the start from the route asset.");
        }

        var grid = ReadTerrain(terrain, route);
        var boulders = MarkBoulders(terrain, route, ref grid);
        int start = Walkable(grid, route.startXZ, "start");
        int goal = Walkable(grid, route.destinationXZ, "destination");

        var cells = AStar(grid, start, goal);
        if (cells == null) throw new InvalidOperationException("No walkable route between start and destination with these settings.");

        var corners = Smooth(grid, cells, route.startXZ, route.destinationXZ, route.smoothingTolerance);
        var points = Drape(grid, corners, route.pointSpacing);
        WriteResult(route, grid, points, corners.Count, boulders, terrain);

        EditorUtility.SetDirty(route);
        AssetDatabase.SaveAssets();
        Debug.Log($"[Navigation] {route.routeId}: {route.description}", route);
    }

    static Vector3? FindSpawn()
    {
        var origin = UnityEngine.Object.FindObjectsByType<XROrigin>(FindObjectsInactive.Include, FindObjectsSortMode.None)
            .FirstOrDefault(o => o.gameObject.scene == SceneManager.GetActiveScene());
        if (origin != null) return origin.transform.position;
        var player = UnityEngine.Object.FindObjectsByType<CharacterController>(FindObjectsInactive.Include, FindObjectsSortMode.None)
            .FirstOrDefault(p => p.gameObject.scene == SceneManager.GetActiveScene());
        return player != null ? player.transform.position : (Vector3?)null;
    }

    static Grid ReadTerrain(Terrain terrain, NavigationRoute route)
    {
        var data = terrain.terrainData;
        int n = data.heightmapResolution;
        var size = data.size;
        var grid = new Grid
        {
            n = n, dx = size.x / (n - 1), dz = size.z / (n - 1), origin = terrain.GetPosition(),
            height = new float[n * n], slope = new float[n * n], penalty = new float[n * n], blocked = new bool[n * n],
        };
        var raw = data.GetHeights(0, 0, n, n);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                grid.height[j * n + i] = grid.origin.y + raw[j, i] * size.y;

        int marginX = Mathf.CeilToInt(route.edgeMargin / grid.dx), marginZ = Mathf.CeilToInt(route.edgeMargin / grid.dz);
        int holes = data.holesResolution;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
            {
                int k = j * n + i;
                float gx = (grid.height[j * n + Mathf.Min(i + 1, n - 1)] - grid.height[j * n + Mathf.Max(i - 1, 0)]) / ((Mathf.Min(i + 1, n - 1) - Mathf.Max(i - 1, 0)) * grid.dx);
                float gz = (grid.height[Mathf.Min(j + 1, n - 1) * n + i] - grid.height[Mathf.Max(j - 1, 0) * n + i]) / ((Mathf.Min(j + 1, n - 1) - Mathf.Max(j - 1, 0)) * grid.dz);
                float s = Mathf.Atan(Mathf.Sqrt(gx * gx + gz * gz)) * Mathf.Rad2Deg;
                grid.slope[k] = s;
                float r = s / route.referenceSlopeDeg;
                grid.penalty[k] = 1 + route.slopeWeight * r * r;
                bool hole = data.IsHole(Mathf.Min(i * holes / (n - 1), holes - 1), Mathf.Min(j * holes / (n - 1), holes - 1));
                grid.blocked[k] = s > route.maxSlopeDeg || hole ||
                    i < marginX || j < marginZ || i >= n - marginX || j >= n - marginZ;
            }
        return grid;
    }

    // Only boulders block routes.
    static int MarkBoulders(Terrain terrain, NavigationRoute route, ref Grid grid)
    {
        var field = terrain.GetComponent<JezeroRockField>();
        if (field == null) { Debug.LogWarning("[Navigation] No JezeroRockField on the HiRISE terrain; boulders not avoided."); return 0; }
        if (field.Rocks.Count == 0) field.Rebuild();
        int count = 0;
        foreach (var rock in field.Rocks)
        {
            if (rock.kind != 2) continue;
            count++;
            Vector3 c = terrain.transform.TransformPoint(rock.position);
            float r = 0.5f * Mathf.Max(rock.scale.x, rock.scale.z) + route.boulderClearance;
            int i0 = Mathf.FloorToInt((c.x - r - grid.origin.x) / grid.dx), i1 = Mathf.CeilToInt((c.x + r - grid.origin.x) / grid.dx);
            int j0 = Mathf.FloorToInt((c.z - r - grid.origin.z) / grid.dz), j1 = Mathf.CeilToInt((c.z + r - grid.origin.z) / grid.dz);
            for (int j = Mathf.Max(0, j0); j <= Mathf.Min(grid.n - 1, j1); j++)
                for (int i = Mathf.Max(0, i0); i <= Mathf.Min(grid.n - 1, i1); i++)
                    if ((grid.XZ(j * grid.n + i) - new Vector2(c.x, c.z)).sqrMagnitude < r * r)
                        grid.blocked[j * grid.n + i] = true;
        }
        return count;
    }

    static int Walkable(Grid grid, Vector2 xz, string what)
    {
        int cell = grid.Cell(xz.x, xz.y);
        if (cell < 0) throw new InvalidOperationException($"The {what} ({xz.x:0.#}, {xz.y:0.#}) is outside the HiRISE terrain.");
        if (grid.blocked[cell]) throw new InvalidOperationException($"The {what} ({xz.x:0.#}, {xz.y:0.#}) is on blocked ground (too steep, a boulder or the edge margin).");
        return cell;
    }

    static List<int> AStar(Grid grid, int start, int goal)
    {
        int n = grid.n;
        var cost = new float[n * n];
        var parent = new int[n * n];
        var closed = new bool[n * n];
        for (int k = 0; k < cost.Length; k++) cost[k] = float.MaxValue;
        cost[start] = 0;
        parent[start] = -1;
        Vector2 target = grid.XZ(goal);
        var open = new MinHeap();
        open.Push(start, Vector2.Distance(grid.XZ(start), target));
        while (open.Count > 0)
        {
            int k = open.Pop();
            if (closed[k]) continue;
            closed[k] = true;
            if (k == goal) break;
            int i = k % n, j = k / n;
            for (int dj = -1; dj <= 1; dj++)
                for (int di = -1; di <= 1; di++)
                {
                    if (di == 0 && dj == 0) continue;
                    int u = i + di, v = j + dj;
                    if (u < 0 || v < 0 || u >= n || v >= n) continue;
                    int m = v * n + u;
                    if (closed[m] || grid.blocked[m]) continue;
                    float h = grid.height[m] - grid.height[k];
                    float step = Mathf.Sqrt(di * di * grid.dx * grid.dx + dj * dj * grid.dz * grid.dz + h * h);
                    float next = cost[k] + step * 0.5f * (grid.penalty[k] + grid.penalty[m]);
                    if (next >= cost[m]) continue;
                    cost[m] = next;
                    parent[m] = k;
                    open.Push(m, next + Vector2.Distance(grid.XZ(m), target));   // Distance is a lower bound because penalties are at least 1.
                }
        }
        if (!closed[goal]) return null;
        var path = new List<int>();
        for (int k = goal; k != -1; k = parent[k]) path.Add(k);
        path.Reverse();
        return path;
    }

    static float? SegmentCost(Grid grid, Vector2 a, Vector2 b)
    {
        float length = Vector2.Distance(a, b);
        int steps = Mathf.Max(1, Mathf.CeilToInt(length / 0.5f));
        float total = 0, previous = grid.Height(a.x, a.y);
        for (int s = 1; s <= steps; s++)
        {
            Vector2 p = Vector2.Lerp(a, b, s / (float)steps);
            int cell = grid.Cell(p.x, p.y);
            if (cell < 0 || grid.blocked[cell]) return null;
            float y = grid.Height(p.x, p.y);
            total += Mathf.Sqrt(length / steps * (length / steps) + (y - previous) * (y - previous)) * grid.penalty[cell];
            previous = y;
        }
        return total;
    }

    // Keep shortcuts within the A* cost tolerance.
    static List<Vector2> Smooth(Grid grid, List<int> cells, Vector2 start, Vector2 goal, float tolerance)
    {
        var path = cells.Select(grid.XZ).ToList();
        path[0] = start;
        path[path.Count - 1] = goal;
        var along = new float[path.Count];
        for (int k = 1; k < path.Count; k++) along[k] = along[k - 1] + (SegmentCost(grid, path[k - 1], path[k]) ?? 1e6f);

        var corners = new List<Vector2> { path[0] };
        int i = 0;
        while (i < path.Count - 1)
        {
            int next = i + 1;
            for (int j = path.Count - 1; j > i + 1; j--)
            {
                float? c = SegmentCost(grid, path[i], path[j]);
                if (c.HasValue && c.Value <= (along[j] - along[i]) * (1 + tolerance)) { next = j; break; }
            }
            corners.Add(path[next]);
            i = next;
        }
        return corners;
    }

    static List<Vector3> Drape(Grid grid, List<Vector2> corners, float spacing)
    {
        var points = new List<Vector3> { On(grid, corners[0]) };
        for (int k = 1; k < corners.Count; k++)
        {
            int steps = Mathf.Max(1, Mathf.CeilToInt(Vector2.Distance(corners[k - 1], corners[k]) / spacing));
            for (int s = 1; s <= steps; s++) points.Add(On(grid, Vector2.Lerp(corners[k - 1], corners[k], s / (float)steps)));
        }
        return points;
    }

    static Vector3 On(Grid grid, Vector2 xz) => new Vector3(xz.x, grid.Height(xz.x, xz.y), xz.y);

    static void WriteResult(NavigationRoute route, Grid grid, List<Vector3> points, int corners, int boulders, Terrain terrain)
    {
        float length = 0, gain = 0, loss = 0;
        var slopes = new List<float>();
        for (int k = 0; k < points.Count; k++)
        {
            if (k > 0)
            {
                length += Vector3.Distance(points[k - 1], points[k]);
                float dy = points[k].y - points[k - 1].y;
                if (dy > 0) gain += dy; else loss -= dy;
            }
            slopes.Add(grid.slope[grid.Cell(points[k].x, points[k].z)]);
        }
        slopes.Sort();
        route.points = points.ToArray();
        route.lengthMeters = length;
        route.elevationGainMeters = gain;
        route.elevationLossMeters = loss;
        route.maxSlopeOnRouteDeg = slopes[slopes.Count - 1];
        route.typicalSlopeDeg = slopes[Mathf.Min(slopes.Count - 1, Mathf.FloorToInt(slopes.Count * 0.9f))];
        route.straightLineMeters = Vector2.Distance(route.startXZ, route.destinationXZ);
        route.builtAt = DateTime.Now.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture);

        var inv = CultureInfo.InvariantCulture;
        route.description = string.Format(inv,
            "{0}: {1:0} m walk ({2:0} m straight line), {3:0.0} m up and {4:0.0} m down. Ground slope under the route: max {5:0.0} deg, 90% under {6:0.0} deg. {7} straight legs.",
            route.destinationName, length, route.straightLineMeters, gain, loss, route.maxSlopeOnRouteDeg, route.typicalSlopeDeg, corners - 1);
        var field = terrain.GetComponent<JezeroRockField>();
        route.source = string.Format(inv,
            "Offline A* (8 neighbours) over the '{0}' heightfield: {1}x{1} samples, {2:0.00} x {3:0.00} m cells, from the HiRISE 1 m DTM.\n" +
            "Cost = 3D step length x (1 + {4} x (slope / {5} deg)^2). Never walks ground over {6} deg or within {7} m of the terrain edge.\n" +
            "Avoids the {8} generated boulders (JezeroRockField seed {9}, radius {10} m) by their radius + {11} m. Medium rocks and pebbles are not obstacles.\n" +
            "Boulders are procedural placeholders, not mapped HiRISE rocks. Zigzags removed with straight-line shortcuts (<= {12:0}% extra cost), then draped on the terrain every {13} m.\n" +
            "Demo guidance only, not verified safe.",
            terrain.name, grid.n, grid.dx, grid.dz, route.slopeWeight, route.referenceSlopeDeg, route.maxSlopeDeg, route.edgeMargin,
            boulders, field != null ? field.seed : 0, field != null ? field.radius : 0, route.boulderClearance,
            route.smoothingTolerance * 100, route.pointSpacing);
    }

    class MinHeap
    {
        readonly List<(int cell, float priority)> items = new List<(int, float)>();
        public int Count => items.Count;

        public void Push(int cell, float priority)
        {
            items.Add((cell, priority));
            int i = items.Count - 1;
            while (i > 0)
            {
                int up = (i - 1) / 2;
                if (items[up].priority <= items[i].priority) break;
                (items[up], items[i]) = (items[i], items[up]);
                i = up;
            }
        }

        public int Pop()
        {
            int top = items[0].cell;
            int last = items.Count - 1;
            items[0] = items[last];
            items.RemoveAt(last);
            int i = 0;
            while (true)
            {
                int l = i * 2 + 1, r = l + 1, best = i;
                if (l < items.Count && items[l].priority < items[best].priority) best = l;
                if (r < items.Count && items[r].priority < items[best].priority) best = r;
                if (best == i) break;
                (items[best], items[i]) = (items[i], items[best]);
                i = best;
            }
            return top;
        }
    }
}
