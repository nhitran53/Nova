using System;
using UnityEngine;

// Routes use world coordinates. Rebuild after moving the origin.
[CreateAssetMenu(menuName = "Nova/Navigation Route", fileName = "route")]
public class NavigationRoute : ScriptableObject
{
    [Header("Destination")]
    public string routeId = "route_shuttle_001";
    public string destinationId = "shuttle_001";
    public string destinationName = "Shuttle";
    [Tooltip("Route preference matched against navigation requests.")]
    public string preference = "low_slope";
    [Tooltip("Scene x/z of the destination. Change it and rebuild to move the shuttle.")]
    public Vector2 destinationXZ = new Vector2(-114.9f, -80.5f);
    [Tooltip("Read the start from the astronaut spawn (Quest XR Origin) when rebuilding.")]
    public bool startAtSpawn = true;
    public Vector2 startXZ = new Vector2(-0.24f, -0.14f);

    [Header("Search")]
    [Tooltip("Ground steeper than this is never walked.")]
    [Range(5, 35)] public float maxSlopeDeg = 20;
    [Tooltip("Cost multiplier is 1 + slopeWeight * (slope / referenceSlope)^2.")]
    [Min(0)] public float slopeWeight = 4;
    [Range(1, 30)] public float referenceSlopeDeg = 8;
    [Tooltip("Free space kept around each generated boulder, on top of its own radius.")]
    [Min(0)] public float boulderClearance = 1.5f;
    [Tooltip("Stay this far inside the HiRISE terrain edge.")]
    [Min(0)] public float edgeMargin = 10;
    [Tooltip("Allowed shortcut cost increase. 0.03 means 3 percent.")]
    [Range(0, 0.2f)] public float smoothingTolerance = 0.03f;
    [Range(0.5f, 5)] public float pointSpacing = 1;

    [Header("Result (written by the rebuild menu)")]
    public Vector3[] points = Array.Empty<Vector3>();
    public float lengthMeters;
    public float elevationGainMeters;
    public float elevationLossMeters;
    public float maxSlopeOnRouteDeg;
    [Tooltip("90th percentile of the ground slope under the route.")]
    public float typicalSlopeDeg;
    public float straightLineMeters;
    [TextArea(2, 6)] public string description;
    [TextArea(3, 10)] public string source;
    public string builtAt;

    public bool IsBuilt => points != null && points.Length >= 2;
    public Vector3 Destination => IsBuilt ? points[points.Length - 1] : new Vector3(destinationXZ.x, 0, destinationXZ.y);
}
