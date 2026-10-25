using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.Hands;

[RequireComponent(typeof(MarsMinimap))]
[DisallowMultipleComponent]
public class HolographicMinimap : MonoBehaviour
{
    [Tooltip("Wrist-local meters. -Y is above the palm; -Z points toward the elbow.")]
    [SerializeField] Vector3 palmOffset = new Vector3(0, -.12f, -.03f);
    [Tooltip("Adjustment after choosing an upright, viewer-facing orientation when opening.")]
    [SerializeField] Vector3 rotationOffset;
    [SerializeField, Range(.18f, .5f)] float diameterMeters = .32f;

    MarsMinimap map;
    XRHandTrackingEvents rightHand;
    Transform wrist, anchor, boundHead, trackingSpace;
    Quaternion palmRotation;
    float nextSearch;

    public bool RightHandTracked => rightHand != null && rightHand.isActiveAndEnabled &&
        rightHand.handIsTracked && wrist != null && wrist.gameObject.activeInHierarchy;

    void Awake() => map = GetComponent<MarsMinimap>();

    void Update()
    {
        if (boundHead != map.head)
        {
            ReleaseHand();
            boundHead = map.head;
            nextSearch = 0;
        }
        if ((rightHand == null || wrist == null) && map.head != null && Time.unscaledTime >= nextSearch)
        {
            nextSearch = Time.unscaledTime + .5f;
            ReleaseHand();
            var origin = map.head.GetComponentInParent<XROrigin>();
            if (origin == null) return; // Desktop controls stay available without hand tracking.
            trackingSpace = origin.CameraFloorOffsetObject.transform;
            foreach (var skeleton in origin.CameraFloorOffsetObject.GetComponentsInChildren<XRHandSkeletonDriver>(true))
            {
                var events = skeleton.handTrackingEvents;
                if (events == null || events.handedness != Handedness.Right || skeleton.rootTransform == null) continue;
                rightHand = events;
                wrist = skeleton.rootTransform;
                rightHand.trackingLost.AddListener(ReturnToHud);
                rightHand.poseUpdated.AddListener(UpdateAnchorPose);
                break;
            }
        }
        if (map.Holographic && !RightHandTracked) ReturnToHud();
    }

    [ContextMenu("Toggle Holographic Map")]
    public void Toggle()
    {
        if (map.Holographic) { ReturnToHud(); return; }
        if (!RightHandTracked || !map.DockedInHelmetHud) return;
        if (anchor == null)
        {
            anchor = new GameObject("Right Palm Map Anchor").transform;
            anchor.SetParent(trackingSpace, false);
        }
        // Freeze the readable orientation at opening; subsequent rotation belongs to the hand.
        Vector3 normal = map.head.position - wrist.TransformPoint(palmOffset);
        if (normal.sqrMagnitude < .001f) normal = -map.head.forward;
        normal.Normalize();
        Vector3 up = Vector3.ProjectOnPlane(map.head.up, normal);
        if (up.sqrMagnitude < .001f) up = Vector3.ProjectOnPlane(map.head.forward, normal);
        if (up.sqrMagnitude < .001f) up = wrist.forward;
        palmRotation = Quaternion.Inverse(wrist.rotation) * Quaternion.LookRotation(-normal, up.normalized);
        map.PresentOnHand(anchor, diameterMeters);
        UpdateAnchorPose(rightHand.rootPose);
    }

    public void ReturnToHud()
    {
        if (map != null) map.ReturnToHud();
    }

    void UpdateAnchorPose(Pose pose)
    {
        if (anchor == null || wrist == null || !map.Holographic) return;
        var space = wrist.parent;
        var rotation = space.rotation * pose.rotation;
        anchor.SetPositionAndRotation(space.TransformPoint(pose.position) + rotation * palmOffset,
            rotation * palmRotation * Quaternion.Euler(rotationOffset));
        anchor.localScale = Vector3.one;
    }

    void OnValidate() { if (anchor != null && rightHand != null) UpdateAnchorPose(rightHand.rootPose); }

    void OnDisable() => ReleaseHand();

    void ReleaseHand()
    {
        ReturnToHud();
        if (rightHand != null)
        {
            rightHand.trackingLost.RemoveListener(ReturnToHud);
            rightHand.poseUpdated.RemoveListener(UpdateAnchorPose);
        }
        rightHand = null;
        wrist = null;
        if (anchor != null) Destroy(anchor.gameObject);
        anchor = null;
    }
}
