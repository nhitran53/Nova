using System;
using UnityEngine;

// The rendered CTX and HiRISE tiles share the CTX southwest origin.
public class JezeroGeoreference : MonoBehaviour
{
    [Serializable]
    public class Metadata
    {
        public string projectionMethod, projection, axes;
        public double sphereRadius, standardParallel, centralMeridian, latitudeOfOrigin;
        public double falseEasting, falseNorthing, west, south, widthMeters, lengthMeters;
        public bool Valid
        {
            get
            {
                foreach (double value in new[] { sphereRadius, standardParallel, centralMeridian, latitudeOfOrigin,
                    falseEasting, falseNorthing, west, south, widthMeters, lengthMeters })
                    if (double.IsNaN(value) || double.IsInfinity(value)) return false;
                return projectionMethod == "Equirectangular" && projection != null && projection.Contains("Mars_2000_Sphere") &&
                    axes == "X east, Y source elevation, Z north; X/Z origin at southwest raster corner" &&
                    sphereRadius > 0 && widthMeters > 0 && lengthMeters > 0 && Math.Abs(standardParallel) < 90;
            }
        }
    }

    public Metadata metadata;
    public Vector3 North => Vector3.ProjectOnPlane(transform.TransformVector(Vector3.forward), Vector3.up).normalized;

    public bool TryCoordinates(Vector3 world, out double latitude, out double longitude)
    {
        latitude = longitude = double.NaN;
        if (metadata == null || !metadata.Valid) return false;
        var p = transform.InverseTransformPoint(world);
        if (p.x < 0 || p.z < 0 || p.x > metadata.widthMeters || p.z > metadata.lengthMeters) return false;
        var m = metadata;
        latitude = (m.south + p.z - m.falseNorthing) / m.sphereRadius * 180 / Math.PI + m.latitudeOfOrigin;
        longitude = (m.west + p.x - m.falseEasting) / (m.sphereRadius * Math.Cos(m.standardParallel * Math.PI / 180))
            * 180 / Math.PI + m.centralMeridian;
        longitude = (longitude % 360 + 360) % 360;
        return !double.IsNaN(latitude) && !double.IsInfinity(latitude) && !double.IsNaN(longitude) &&
            !double.IsInfinity(longitude) && Math.Abs(latitude) <= 90;
    }

    public Vector3 ToWorld(double latitude, double longitude, float localElevation = 0)
    {
        if (metadata == null || !metadata.Valid) throw new InvalidOperationException("Terrain georeferencing is unavailable.");
        var m = metadata;
        double deltaLongitude = (longitude - m.centralMeridian + 540) % 360 - 180;
        double east = m.falseEasting + m.sphereRadius * deltaLongitude * Math.PI / 180 * Math.Cos(m.standardParallel * Math.PI / 180);
        double north = m.falseNorthing + m.sphereRadius * (latitude - m.latitudeOfOrigin) * Math.PI / 180;
        return transform.TransformPoint(new Vector3((float)(east - m.west), localElevation, (float)(north - m.south)));
    }
}
