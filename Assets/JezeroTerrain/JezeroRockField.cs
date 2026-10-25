using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

[ExecuteAlways]
[DisallowMultipleComponent]
public class JezeroRockField : MonoBehaviour
{
    public Terrain terrain;
    public Mesh[] nearMeshes, farMeshes;
    public Material[] materials;
    [Tooltip("Meters from the southwest corner of the HiRISE terrain")]
    public Vector2 center;
    public int seed = 2020;
    [Range(10, 150)] public float radius = 80;
    [Range(0, 5000)] public int smallStones = 1200;
    [Range(0, 500)] public int mediumRocks = 180;
    [Range(0, 50)] public int boulders = 18;
    [Range(0, 45)] public float maxSlope = 32;
    [Range(10, 150)] public float drawDistance = 90;
    [Range(5, 60)] public float smallStoneDistance = 30;
    [Range(5, 60)] public float lodDistance = 22;

    public struct Rock
    {
        public Vector3 position, scale;
        public Quaternion rotation;
        public int variant, kind;
    }

    readonly List<Rock> rocks = new List<Rock>();
    public IReadOnlyList<Rock> Rocks => rocks;
    const int BatchSize = 256;
    Matrix4x4[][] batches;
    int[] counts;

    void OnEnable()
    {
        Rebuild();
        RenderPipelineManager.beginCameraRendering += Draw;
    }

    void OnDisable() => RenderPipelineManager.beginCameraRendering -= Draw;

    public void Rebuild()
    {
        rocks.Clear();
        if (terrain == null || nearMeshes == null || farMeshes == null || nearMeshes.Length == 0 ||
            nearMeshes.Length != farMeshes.Length || materials == null || materials.Length == 0) return;
        for (int i = 0; i < nearMeshes.Length; i++)
            if (nearMeshes[i] == null || farMeshes[i] == null) return;
        var supports = new List<Vector3>[nearMeshes.Length * 2];
        for (int i = 0; i < nearMeshes.Length; i++)
        {
            for (int lod = 0; lod < 2; lod++)
            {
                int slot = i * 2 + lod;
                supports[slot] = new List<Vector3>();
                foreach (var vertex in (lod == 0 ? nearMeshes[i] : farMeshes[i]).vertices)
                    if (vertex.y < .35f && !supports[slot].Contains(vertex)) supports[slot].Add(vertex);
            }
        }
        AddRocks(2, Mathf.Clamp(boulders, 0, 50), supports);
        AddRocks(1, Mathf.Clamp(mediumRocks, 0, 500), supports);
        AddRocks(0, Mathf.Clamp(smallStones, 0, 5000), supports);
        batches = new Matrix4x4[nearMeshes.Length * 2][];
        counts = new int[batches.Length];
        for (int i = 0; i < batches.Length; i++) batches[i] = new Matrix4x4[BatchSize];
    }

    void AddRocks(int kind, int wanted, List<Vector3>[] supports)
    {
        var random = new System.Random(unchecked(seed + kind * 9973));
        var data = terrain.terrainData;
        var size = data.size;
        int made = 0;
        for (int attempt = 0; made < wanted && attempt < wanted * 30; attempt++)
        {
            float angle = Next(random, 0, Mathf.PI * 2);
            float distance = Mathf.Sqrt((float)random.NextDouble()) * Mathf.Max(1, radius);
            if (distance < (kind == 2 ? 7 : kind == 1 ? 2.5f : 1)) continue;
            float x = center.x + Mathf.Cos(angle) * distance;
            float z = center.y + Mathf.Sin(angle) * distance;
            if (x < 3 || z < 3 || x > size.x - 3 || z > size.z - 3) continue;
            float u = x / size.x, v = z / size.z;
            int hx = Mathf.Clamp((int)(u * data.holesResolution), 0, data.holesResolution - 1);
            int hz = Mathf.Clamp((int)(v * data.holesResolution), 0, data.holesResolution - 1);
            if (data.IsHole(hx, hz)) continue;
            var normal = data.GetInterpolatedNormal(u, v);
            if (Vector3.Angle(Vector3.up, normal) > maxSlope) continue;
            float patch = Mathf.PerlinNoise(x * .035f + seed * .001f, z * .035f);
            if (random.NextDouble() > Mathf.Lerp(.25f, 1, patch)) continue;
            float width = kind == 2 ? Next(random, 1.1f, 2.4f) : kind == 1 ? Next(random, .28f, .8f) : Next(random, .05f, .22f);
            var scale = new Vector3(width * Next(random, .8f, 1.25f), width * Next(random, .55f, .95f), width * Next(random, .8f, 1.25f));
            bool overlaps = false;
            if (kind == 2)
                foreach (var other in rocks)
                    if (Vector2.Distance(new Vector2(x,z), new Vector2(other.position.x,other.position.z)) <
                        Mathf.Max(scale.x,scale.z) + Mathf.Max(other.scale.x,other.scale.z) + 1) { overlaps = true; break; }
            if (overlaps) continue;
            int variant = random.Next(nearMeshes.Length);
            var rotation = Quaternion.FromToRotation(Vector3.up, normal) *
                Quaternion.Euler(Next(random,-10,10), Next(random,0,360), Next(random,-10,10));
            float y = float.PositiveInfinity;
            for (int lod = 0; lod < 2; lod++)
            {
                float supportHeight = float.NegativeInfinity;
                foreach (var vertex in supports[variant * 2 + lod])
                {
                    var point = rotation * Vector3.Scale(vertex, scale);
                    float ground = data.GetInterpolatedHeight((x + point.x) / size.x, (z + point.z) / size.z);
                    supportHeight = Mathf.Max(supportHeight, ground - point.y);
                }
                y = Mathf.Min(y, supportHeight);
            }
            // Sink the base slightly so uneven rocks touch the ground.
            y -= scale.y * .06f;
            rocks.Add(new Rock { position = new Vector3(x,y,z), rotation = rotation, scale = scale, variant = variant, kind = kind });
            made++;
        }
    }

    static float Next(System.Random random, float min, float max) => min + (float)random.NextDouble() * (max - min);

    void Draw(ScriptableRenderContext context, Camera camera)
    {
        if (terrain == null || batches == null || rocks.Count == 0 || !SystemInfo.supportsInstancing ||
            (camera.cameraType != CameraType.Game && camera.cameraType != CameraType.SceneView) ||
            (camera.cullingMask & (1 << gameObject.layer)) == 0) return;
#if UNITY_EDITOR
        if (UnityEditor.SceneManagement.EditorSceneManager.IsPreviewSceneObject(gameObject)) return;
#endif
        System.Array.Clear(counts, 0, counts.Length);
        var matrix = terrain.transform.localToWorldMatrix;
        var eye = camera.transform.position;
        var bounds = new Bounds(matrix.MultiplyPoint3x4(new Vector3(center.x, terrain.terrainData.size.y * .5f, center.y)),
            Vector3.one * (radius * 2 + 10));
        foreach (var rock in rocks)
        {
            var position = matrix.MultiplyPoint3x4(rock.position);
            float squared = (position - eye).sqrMagnitude;
            float limit = rock.kind == 0 ? Mathf.Min(drawDistance, smallStoneDistance) : rock.kind == 1 ? drawDistance * .8f : drawDistance;
            if (squared > limit * limit) continue;
            int bucket = rock.variant * 2 + (squared > lodDistance * lodDistance ? 1 : 0);
            batches[bucket][counts[bucket]++] = matrix * Matrix4x4.TRS(rock.position, rock.rotation, rock.scale);
            if (counts[bucket] == BatchSize) Flush(bucket, camera, bounds);
        }
        for (int i = 0; i < counts.Length; i++) if (counts[i] > 0) Flush(i, camera, bounds);
    }

    void Flush(int bucket, Camera camera, Bounds bounds)
    {
        int variant = bucket / 2;
        var mesh = bucket % 2 == 0 ? nearMeshes[variant] : farMeshes[variant];
        var material = materials[variant % materials.Length];
        if (mesh == null || material == null) { counts[bucket] = 0; return; }
        var parameters = new RenderParams(material)
        {
            camera = camera, worldBounds = bounds, layer = gameObject.layer,
            shadowCastingMode = ShadowCastingMode.Off, receiveShadows = true,
            lightProbeUsage = LightProbeUsage.BlendProbes, reflectionProbeUsage = ReflectionProbeUsage.Off
        };
        Graphics.RenderMeshInstanced(parameters, mesh, 0, batches[bucket], counts[bucket]);
        counts[bucket] = 0;
    }
}
