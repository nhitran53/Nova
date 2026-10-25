using UnityEngine;

public static class HazardZoneGeometry
{
    // Signed clearance: negative inside, zero on the rim. Height never affects danger.
    public static float EdgeDistance(Vector3 position, float centerX, float centerZ, float radius)
    {
        return Vector2.Distance(new Vector2(position.x, position.z), new Vector2(centerX, centerZ)) - radius;
    }

    public static bool IsInside(Vector3 position, float centerX, float centerZ, float radius) =>
        EdgeDistance(position, centerX, centerZ, radius) <= 0;
}
