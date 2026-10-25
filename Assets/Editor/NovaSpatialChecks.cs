using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class NovaSpatialChecks
{
    [MenuItem("Tools/NOVA/Check Spatial Mapping")]
    public static void Run()
    {
        var root = EditorUtility.CreateGameObjectWithHideFlags("Spatial mapping check", HideFlags.HideAndDontSave);
        try
        {
            var geo = root.AddComponent<JezeroGeoreference>();
            geo.metadata = JsonUtility.FromJson<JezeroGeoreference.Metadata>(
                File.ReadAllText("Assets/JezeroTerrain/Generated/terrain.json"));
            Check(geo.metadata.Valid, "Source metadata");
            Check(geo.TryCoordinates(Vector3.zero, out double lat, out double lon), "Southwest corner");
            Check(Math.Abs(lat - 17.580519910544755) < 1e-9 && Math.Abs(lon - 76.99743706146002) < 1e-9,
                "Source TIFF southwest reference");
            var parent = EditorUtility.CreateGameObjectWithHideFlags("Transformed terrain parent", HideFlags.HideAndDontSave);
            try
            {
                root.transform.SetParent(parent.transform, false);
                parent.transform.SetPositionAndRotation(new Vector3(-25494, 2567, -51220), Quaternion.Euler(0, 31, 0));
                parent.transform.localScale = new Vector3(1.2f, .9f, 1.4f);
                var local = new Vector3(25221.65625f, -2573, 50907.515625f);
                var world = root.transform.TransformPoint(local);
                Check(geo.TryCoordinates(world, out lat, out lon), "HiRISE shared CTX frame");
                Check(Math.Abs(lat - 18.439360490962454) < 1e-6 && Math.Abs(lon - 77.44604017758272) < 1e-6,
                    "HiRISE source reference with parent translation rotation and scale");
                Check(Vector3.Distance(world, geo.ToWorld(lat, lon, local.y)) < .03f, "Geographic round trip");
                Check(!geo.TryCoordinates(root.transform.TransformPoint(new Vector3(-10, 0, -10)), out _, out _), "Outside source bounds");
            }
            finally { UnityEngine.Object.DestroyImmediate(parent); }
            root = EditorUtility.CreateGameObjectWithHideFlags("Missing mapping check", HideFlags.HideAndDontSave);
            Check(!root.AddComponent<JezeroGeoreference>().TryCoordinates(Vector3.zero, out _, out _), "Missing georeferencing");
            var ahead = NovaSpatialContextProvider.Describe("test", "Test", new Vector3(3, 50, 4), Vector3.zero, Vector3.forward, Vector3.forward);
            Check(Mathf.Abs(ahead.distance_m - 5) < 1e-6 && ahead.direction == "northeast" && ahead.relative_direction == "in front of you",
                "Ground distance and bearing ignore landmark height");
            var behind = NovaSpatialContextProvider.Describe("test", "Test", new Vector3(3, 50, 4), Vector3.zero, Vector3.back, Vector3.forward);
            Check(behind.relative_direction == "behind you" && behind.distance_m == ahead.distance_m, "Head rotation changes relative direction only");
            var vertical = NovaSpatialContextProvider.Describe("test", "Test", Vector3.right, Vector3.zero, Vector3.up, Vector3.forward);
            Check(vertical.relative_direction == "unavailable", "Looking straight up has no horizontal viewing direction");
            var tracker = new VoiceStatusTracker();
            var status = new VoiceStatus { state = "processing", spatial_pending = true, spatial_seq = 1 };
            Check(tracker.Apply(status).SpatialRequested && !tracker.Apply(status).SpatialRequested, "One snapshot per request");
            status.spatial_seq++;
            Check(tracker.Apply(status).SpatialRequested, "New request gets a fresh snapshot");
            Debug.Log("Spatial mapping checks passed.");
            if (Application.isBatchMode) File.WriteAllText("Temp/SpatialChecks.txt", "PASS");
        }
        finally { if (root != null) UnityEngine.Object.DestroyImmediate(root); }
    }

    public static void SetupAndCheck()
    {
        JezeroTerrainImporter.UpdateGeoreference();
        Run();
        if (Application.isBatchMode)
        {
            EditorSceneManager.OpenScene("Assets/Scenes/SampleScene.unity");
            CheckCurrentScene();
        }
    }

    [MenuItem("Tools/NOVA/Check Current Spatial Context")]
    public static void CheckCurrentScene()
    {
        var scene = SceneManager.GetActiveScene();
        foreach (var root in scene.GetRootGameObjects())
            foreach (var child in root.GetComponentsInChildren<Transform>(true))
                Check(GameObjectUtility.GetMonoBehavioursWithMissingScriptCount(child.gameObject) == 0, "Scene script references");
        var client = UnityEngine.Object.FindAnyObjectByType<VoiceSessionClient>();
        Check(client != null, "Voice client");
        var probe = EditorUtility.CreateGameObjectWithHideFlags("Spatial scene check", HideFlags.HideAndDontSave);
        try
        {
            var provider = probe.AddComponent<NovaSpatialContextProvider>();
            var snapshot = provider.Capture();
            Check(snapshot.available && snapshot.coordinates_verified, "Live scene terrain mapping");
            Check(Array.Exists(snapshot.nearby_landmarks, p => p.id == "shuttle_001"), "Registered shuttle");
            Check(Array.Exists(snapshot.nearby_landmarks, p => p.id == "rover_front_right_wheel"), "Actual rover wheel");
            var camera = Camera.main.transform;
            if (Application.isBatchMode)
            {
                var position = camera.position;
                try
                {
                    camera.position += Vector3.right;
                    var moved = provider.Capture();
                    Check(moved.available && moved.coordinates_verified, "Moved headset mapping");
                    Check(Mathf.Abs(moved.world_position.x - snapshot.world_position.x - 1) < .001f, "Headset ground position");
                    Check(moved.longitude != snapshot.longitude, "Movement refreshes coordinates");
                }
                finally { camera.position = position; }
            }
            Debug.Log("Scene spatial snapshot: " + JsonUtility.ToJson(snapshot));
            Debug.Log("Spatial scene checks passed. SampleScene was not saved.");
        }
        finally { UnityEngine.Object.DestroyImmediate(probe); }
    }

    static void Check(bool condition, string name)
    {
        if (!condition) throw new InvalidOperationException("Spatial check failed: " + name);
    }
}
