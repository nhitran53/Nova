using System.Collections.Generic;
using System.Globalization;
using System.Text;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR;
using UnityEngine.XR.Hands;

// Resolve a pointing hand's Meta Aim ray when NOVA asks for a target.
// Match scene markers and rover bounds before checking terrain colliders.
public static class NovaPointingTarget
{
    public const float MaxDistance = 250;
    const float IndexExtended = 1.2f;   // fingertip-to-palm / hand size, as in TelephoneGestureDetector
    const float Curled = 0.65f;
    const float BoundsPadding = 0.3f;
    const int UiLayer = 5;

    public class Target
    {
        public string objectId, displayName, kind, description, source;
        public string part;   // a named part the ray passes through, e.g. front_right_wheel
        public bool hasDamaged, damaged;
        public Vector3 point;
        public float distance;
    }

    public class Pointing
    {
        public string hand;   // "right", "left", "editor_gaze" or null when no hand points
        public string reason;
        public Ray ray;
        public Target target;
    }

    static readonly List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();

    public static Pointing Resolve(Camera head)
    {
        var result = new Pointing();
        if (!TryGetRay(head, result)) return result;
        result.target = ResolveRay(result.ray, head != null ? head.transform.position : result.ray.origin);
        if (result.target == null) result.reason = "nothing in line with the pointing ray";
        return result;
    }

    static bool TryGetRay(Camera head, Pointing result)
    {
        XRHandSubsystem hands = null;
        SubsystemManager.GetSubsystems(subsystems);
        foreach (var s in subsystems) if (s.running) { hands = s; break; }
        if (hands == null && !XRSettings.isDeviceActive && head != null)
        {
            // Desktop / Editor without a headset: the view centre stands in for the hand ray.
            result.hand = "editor_gaze";
            result.ray = new Ray(head.transform.position, head.transform.forward);
            return true;
        }
        if (hands == null) { result.reason = "no hand tracking"; return false; }
        var origin = Object.FindAnyObjectByType<XROrigin>();
        var space = origin != null && origin.CameraFloorOffsetObject != null ? origin.CameraFloorOffsetObject.transform : null;
        foreach (bool right in new[] { true, false })
        {
            if (!IsPointing(right ? hands.rightHand : hands.leftHand)) continue;
            var aim = right ? MetaAimHand.right : MetaAimHand.left;
            if (aim == null || !aim.added || !aim.isTracked.isPressed) continue;
            var flags = (MetaAimFlags)aim.aimFlags.ReadValue();
            if ((flags & MetaAimFlags.Computed) == 0 || (flags & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) != 0) continue;
            Vector3 position = aim.devicePosition.ReadValue();
            Quaternion rotation = aim.deviceRotation.ReadValue();
            if (space != null)
            {
                position = space.TransformPoint(position);
                rotation = space.rotation * rotation;
            }
            result.hand = right ? "right" : "left";
            result.ray = new Ray(position, rotation * Vector3.forward);
            return true;
        }
        result.reason = "no hand is pointing";
        return false;
    }

    static bool IsPointing(XRHand hand)
    {
        if (!hand.isTracked ||
            !hand.GetJoint(XRHandJointID.Palm).TryGetPose(out Pose palm) ||
            !hand.GetJoint(XRHandJointID.Wrist).TryGetPose(out Pose wrist) ||
            !hand.GetJoint(XRHandJointID.MiddleProximal).TryGetPose(out Pose knuckle)) return false;
        float size = Vector3.Distance(wrist.position, knuckle.position);
        if (size < 0.03f) return false;
        return Reach(hand, XRHandJointID.IndexTip, palm, size) >= IndexExtended &&
               Reach(hand, XRHandJointID.MiddleTip, palm, size) <= Curled &&
               Reach(hand, XRHandJointID.RingTip, palm, size) <= Curled;
    }

    static float Reach(XRHand hand, XRHandJointID tip, Pose palm, float size) =>
        hand.GetJoint(tip).TryGetPose(out Pose pose) ? Vector3.Distance(pose.position, palm.position) / size : -1;

    // The identifiable object (or physics hit) along `ray`, distance measured from `head`; null when nothing is hit.
    public static Target ResolveRay(Ray ray, Vector3 head)
    {
        // The nearest hit that is not the astronaut's own body (the XR Origin capsule starts at the head).
        var hits = Physics.RaycastAll(ray, MaxDistance, ~(1 << UiLayer), QueryTriggerInteraction.Ignore);
        System.Array.Sort(hits, (a, b) => a.distance.CompareTo(b.distance));
        RaycastHit hit = default;
        bool hitSomething = false;
        foreach (var candidate in hits)
            if (!(candidate.collider is CharacterController) && candidate.collider.GetComponentInParent<XROrigin>() == null)
            {
                hit = candidate;
                hitSomething = true;
                break;
            }
        float limit = hitSomething ? hit.distance + BoundsPadding : MaxDistance;
        Target best = null;
        float bestDistance = float.MaxValue;
        foreach (var marker in MapMarker.All)
            Consider(marker.gameObject, ray, limit, ref best, ref bestDistance, () => new Target
            {
                objectId = marker.code, displayName = marker.displayName, kind = marker.locationType.ToString(),
                description = marker.description, source = "map_marker",
            });
        foreach (var rover in Object.FindObjectsByType<RoverDamageState>(FindObjectsSortMode.None))
            Consider(rover.gameObject, ray, limit, ref best, ref bestDistance, () => new Target
            {
                objectId = "perseverance", displayName = "Perseverance rover", kind = "rover", source = "rover",
                hasDamaged = true, damaged = rover.Damaged, part = Hits(ray, rover.FrontRightWheel) ? "front_right_wheel" : null,
            });
        if (best == null && hitSomething)
        {
            bool terrain = hit.collider is TerrainCollider;
            best = new Target
            {
                objectId = terrain ? "terrain" : hit.collider.gameObject.name,
                displayName = terrain ? "Mars terrain (the ground)" : hit.collider.gameObject.name,
                kind = terrain ? "terrain" : "object", source = terrain ? "terrain" : "collider", point = hit.point,
            };
        }
        if (best != null) best.distance = Vector3.Distance(head, best.point);
        return best;
    }

    // True when `ray` passes through the part's renderer bounds (a little padding for a shaky gaze or finger).
    static bool Hits(Ray ray, Transform part)
    {
        var renderer = part != null ? part.GetComponentInChildren<Renderer>() : null;
        if (renderer == null) return false;
        var bounds = renderer.bounds;
        bounds.Expand(0.15f);
        return bounds.IntersectRay(ray);
    }

    static void Consider(GameObject go, Ray ray, float limit, ref Target best, ref float bestDistance, System.Func<Target> make)
    {
        var renderers = go.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0) return;   // e.g. an empty map marker: nothing to point at
        var bounds = renderers[0].bounds;
        foreach (var r in renderers) bounds.Encapsulate(r.bounds);
        bounds.Expand(BoundsPadding);
        if (!bounds.IntersectRay(ray, out float distance) || distance > limit || distance >= bestDistance) return;
        bestDistance = distance;
        best = make();
        best.point = ray.GetPoint(Mathf.Max(distance, 0));
    }

    // Target facts as JSON with only the fields that are known (an unknown damage state is left out, never "false").
    public static string ToJson(Target t, string hand, Vector3 viewport)
    {
        var sb = new StringBuilder("{");
        void Text(string key, string value)
        {
            if (string.IsNullOrEmpty(value)) return;
            sb.Append('"').Append(key).Append("\":\"");
            foreach (char c in value)
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
            }
            sb.Append("\",");
        }
        void Number(string key, float value) => sb.Append('"').Append(key).Append("\":").Append(value.ToString("0.###", CultureInfo.InvariantCulture)).Append(',');
        Text("object_id", t.objectId);
        Text("display_name", t.displayName);
        Text("kind", t.kind);
        Text("description", t.description);
        Text("source", t.source);
        Text("part", t.part);
        Text("hand", hand);
        Number("distance_m", t.distance);
        if (viewport.z > 0)
        {
            Number("screen_x", Mathf.Clamp01(viewport.x));
            Number("screen_y", Mathf.Clamp01(viewport.y));
        }
        if (t.hasDamaged) sb.Append("\"damaged\":").Append(t.damaged ? "true" : "false").Append(',');
        sb.Length--;
        return sb.Append('}').ToString();
    }
}
