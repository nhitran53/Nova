using UnityEngine;
using UnityEngine.Rendering;

public sealed class HazardZoneOverlay : MonoBehaviour
{
    [Tooltip("World floor height in metres. The disc is lifted slightly to avoid z-fighting.")]
    [SerializeField] float floorY;
    const int Segments = 128;
    Mesh mesh;
    Material discMaterial, rimMaterial;
    MeshRenderer disc;
    LineRenderer rim;
    HazardZoneData hazard = new HazardZoneData();

    void Awake()
    {
        // Sprites/Default is unlit, transparent, double-sided and supports vertex tint.
        var shader = Shader.Find("Sprites/Default");
        if (shader == null)
        {
            Debug.LogError("Hazard overlay requires the Sprites/Default shader in the build.", this);
            enabled = false;
            return;
        }
        discMaterial = new Material(shader) { color = new Color(1, 0.05f, 0.03f, 0.18f) };
        rimMaterial = new Material(shader);
        mesh = new Mesh { name = "Meteor hazard unit disc" };
        var vertices = new Vector3[Segments + 1];
        var triangles = new int[Segments * 3];
        for (int i = 0; i < Segments; i++)
        {
            float angle = i * Mathf.PI * 2 / Segments;
            vertices[i + 1] = new Vector3(Mathf.Cos(angle), 0, Mathf.Sin(angle));
            triangles[i * 3] = 0;
            triangles[i * 3 + 1] = (i + 1) % Segments + 1;
            triangles[i * 3 + 2] = i + 1;
        }
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateBounds();
        var fill = new GameObject("Translucent floor disc");
        fill.transform.SetParent(transform, false);
        fill.AddComponent<MeshFilter>().sharedMesh = mesh;
        disc = fill.AddComponent<MeshRenderer>();
        disc.sharedMaterial = discMaterial;
        disc.shadowCastingMode = ShadowCastingMode.Off;
        disc.receiveShadows = false;
        var edge = new GameObject("Hazard rim");
        edge.transform.SetParent(transform, false);
        rim = edge.AddComponent<LineRenderer>();
        rim.sharedMaterial = rimMaterial;
        rim.useWorldSpace = false;
        rim.loop = true;
        rim.positionCount = Segments;
        rim.widthMultiplier = 0.12f;
        rim.shadowCastingMode = ShadowCastingMode.Off;
        rim.receiveShadows = false;
        SetHazard(hazard);
    }

    public void SetHazard(HazardZoneData value)
    {
        hazard = value ?? new HazardZoneData();
        if (disc == null || rim == null) return;
        bool visible = hazard.active && hazard.kind == "meteor" && hazard.radius_m > 0
            && IsFinite(hazard.radius_m) && IsFinite(hazard.center_x_m) && IsFinite(hazard.center_z_m);
        disc.enabled = rim.enabled = visible;
        if (!visible) return;
        transform.position = new Vector3(hazard.center_x_m, floorY + 0.025f, hazard.center_z_m);
        disc.transform.localScale = new Vector3(hazard.radius_m, 1, hazard.radius_m);
        for (int i = 0; i < Segments; i++)
        {
            float angle = i * Mathf.PI * 2 / Segments;
            rim.SetPosition(i, new Vector3(Mathf.Cos(angle) * hazard.radius_m, 0.015f,
                Mathf.Sin(angle) * hazard.radius_m));
        }
        UpdateRim();
    }

    void Update()
    {
        if (rim != null && rim.enabled) UpdateRim();
    }

    void UpdateRim()
    {
        var camera = Camera.main;
        bool? inside = camera != null ? HazardZoneGeometry.IsInside(camera.transform.position,
            hazard.center_x_m, hazard.center_z_m, hazard.radius_m)
            : hazard.insideUnknown ? (bool?)null : hazard.inside;
        rim.startColor = rim.endColor = SelectRimColor(inside, Time.unscaledTime);
    }

    // A missing position pulses red; green requires a positively known outside position.
    public static Color SelectRimColor(bool? inside, float unscaledTime)
    {
        Color tint = inside == false ? new Color(0.15f, 1, 0.45f) : new Color(1, 0.08f, 0.03f);
        tint.a = inside == true ? 1 : 0.6f + 0.4f * (0.5f + 0.5f * Mathf.Sin(unscaledTime * 4));
        return tint;
    }

    static bool IsFinite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);

    void OnDestroy()
    {
        if (mesh != null) Destroy(mesh);
        if (discMaterial != null) Destroy(discMaterial);
        if (rimMaterial != null) Destroy(rimMaterial);
    }
}
