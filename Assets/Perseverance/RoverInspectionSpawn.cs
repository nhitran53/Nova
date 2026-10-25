using System.Collections;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using UnityEngine.XR.Interaction.Toolkit.Locomotion.Gravity;
using UnityEngine.XR.Interaction.Toolkit.Locomotion.Movement;

[DisallowMultipleComponent]
[RequireComponent(typeof(RoverDamageState))]
public class RoverInspectionSpawn : MonoBehaviour
{
    public bool spawnOnStart = true;
    [Tooltip("Optional ground-level anchor. Leave empty to place it beside the damaged wheel.")]
    public Transform inspectionAnchor;
    [Min(.5f)] public float distanceFromWheel = 1.1f;
    [Tooltip("Meters along the rover's forward axis.")]
    public float foreAftOffset;
    [Min(0)] public float groundClearance = .03f;
    public bool faceWheelOnReset = true;
    [Range(.01f, .1f)] public float inspectionNearClip = .03f;
    [Range(.1f, .6f)] public float fadeSeconds = .2f;

    RoverDamageState rover;
    XROrigin origin;
    Coroutine reset;
    Image fade;
    Material fadeMaterial;
    bool generatedAnchor;
    ContinuousMoveProvider move;
    FistLocomotion fist;
    bool resumeMove, resumeFist, paused;

    void Awake() => rover = GetComponent<RoverDamageState>();

    void Start()
    {
        if (spawnOnStart) reset = StartCoroutine(ResetPosition(true));
    }

    [ContextMenu("Reset To Rover Inspection")]
    public void ResetInspectionPosition()
    {
        if (Application.isPlaying && isActiveAndEnabled && reset == null)
            reset = StartCoroutine(ResetPosition(false));
    }

    IEnumerator ResetPosition(bool initial)
    {
        yield return null;
        try
        {
            while (origin == null)
            {
                origin = FindFirstObjectByType<XROrigin>();
                if (origin == null) yield return null;
            }
            if (origin.Camera == null || !CreateFade()) yield break;
            if (initial) fade.color = Color.black;
            while (!TrackingReady())
            {
                if (!initial)
                {
                    Debug.LogWarning("Wait for headset floor tracking before resetting inspection.", this);
                    yield break;
                }
                yield return null;
            }
            yield return new WaitForEndOfFrame();
            if (!TryGetSpawn(out var ground, out var forward)) yield break;
            RouteNavigator.Current?.StopNavigation();
            PauseMovement();
            if (!initial) yield return FadeTo(1);
            if (!TrackingReady()) yield break;
            PlaceOrigin(ground, forward);
            yield return null;
            yield return FadeTo(0);
        }
        finally
        {
            RestoreMovement();
            if (fade != null) fade.gameObject.SetActive(false);
            reset = null;
        }
    }

    bool TrackingReady()
    {
        var headset = InputDevices.GetDeviceAtXRNode(XRNode.Head);
        return origin != null && origin.isActiveAndEnabled && origin.Camera != null && origin.Camera.gameObject.activeInHierarchy &&
            origin.CurrentTrackingOriginMode == TrackingOriginModeFlags.Floor &&
            origin.CameraInOriginSpaceHeight > .2f &&
            headset.TryGetFeatureValue(CommonUsages.isTracked, out bool tracked) && tracked &&
            headset.TryGetFeatureValue(CommonUsages.trackingState, out InputTrackingState state) &&
            (state & (InputTrackingState.Position | InputTrackingState.Rotation)) ==
            (InputTrackingState.Position | InputTrackingState.Rotation);
    }

    // Shared ground position for inspection resets and route building.
    public bool TryGetInspectionGround(out Vector3 ground)
    {
        ground = Vector3.zero;
        var wheel = (rover != null ? rover : GetComponent<RoverDamageState>()).FrontRightWheel;
        if (wheel == null) return false;
        var renderer = wheel.GetComponent<Renderer>();
        var center = renderer != null ? renderer.bounds.center : wheel.position;
        var candidate = inspectionAnchor != null ? inspectionAnchor.position
            : center + transform.right * ((Vector3.Dot(center - transform.position, transform.right) < 0 ? -1 : 1) * distanceFromWheel)
              + transform.forward * foreAftOffset;
        bool found = false;
        foreach (var hit in Physics.RaycastAll(candidate + Vector3.up * 10, Vector3.down, 30, ~0, QueryTriggerInteraction.Ignore))
            if (hit.collider is TerrainCollider && (!found || hit.point.y > ground.y)) { ground = hit.point; found = true; }
        return found;
    }

    bool TryGetSpawn(out Vector3 ground, out Vector3 forward)
    {
        Physics.SyncTransforms();
        forward = Vector3.zero;
        if (!TryGetInspectionGround(out ground))
        {
            Debug.LogWarning("The inspection anchor needs terrain underneath it.", this);
            return false;
        }
        if (inspectionAnchor == null)
        {
            inspectionAnchor = new GameObject("Rover Inspection Spawn").transform;
            inspectionAnchor.SetParent(transform, false);
            generatedAnchor = true;
        }
        ground += Vector3.up * groundClearance;
        var controller = origin.Origin.GetComponent<CharacterController>();
        if (controller == null || !controller.enabled) return false;
        float scale = origin.Origin.transform.lossyScale.y;
        float radius = controller.radius * scale + .02f;
        float height = Mathf.Max(controller.height * scale, radius * 2);
        foreach (var collider in Physics.OverlapCapsule(ground + Vector3.up * radius,
                     ground + Vector3.up * (height - radius), radius, ~0, QueryTriggerInteraction.Ignore))
            if (!(collider is TerrainCollider) && !collider.transform.IsChildOf(origin.transform))
            {
                Debug.LogWarning("The inspection spawn overlaps an object. Adjust its offset or anchor.", this);
                return false;
            }
        // Keep the rover body in view at startup.
        var body = new Bounds(transform.position, Vector3.zero);
        foreach (var part in GetComponentsInChildren<Renderer>()) body.Encapsulate(part.bounds);
        forward = Vector3.ProjectOnPlane(body.center - ground, Vector3.up).normalized;
        if (generatedAnchor) inspectionAnchor.SetPositionAndRotation(ground, Quaternion.LookRotation(forward, Vector3.up));
        if (!faceWheelOnReset) forward = Vector3.ProjectOnPlane(origin.Camera.transform.forward, Vector3.up).normalized;
        return forward.sqrMagnitude > .5f;
    }

    void PlaceOrigin(Vector3 ground, Vector3 forward)
    {
        float eyeHeight = origin.CameraInOriginSpaceHeight * origin.Origin.transform.lossyScale.y;
        origin.MatchOriginUpCameraForward(Vector3.up, forward);
        origin.MoveCameraToWorldLocation(ground + Vector3.up * eyeHeight);
        var controller = origin.Origin.GetComponent<CharacterController>();
        if (controller != null)
        {
            var headLocal = origin.CameraInOriginSpacePos;
            controller.center = new Vector3(headLocal.x, controller.center.y, headLocal.z);
        }
        origin.Camera.nearClipPlane = Mathf.Min(origin.Camera.nearClipPlane, inspectionNearClip);
        var gravity = origin.GetComponentInChildren<GravityProvider>();
        if (gravity != null)
        {
            gravity.ResetFallForce();
            gravity.transformation.motion = Vector3.zero;
        }
        Physics.SyncTransforms();
    }

    void PauseMovement()
    {
        fist = origin.GetComponent<FistLocomotion>();
        move = origin.GetComponentInChildren<ContinuousMoveProvider>();
        resumeFist = fist != null && fist.enabled;
        resumeMove = move != null && move.enabled;
        paused = true;
        if (resumeFist) fist.enabled = false;
        if (move != null)
        {
            move.transformation.motion = Vector3.zero;
            move.enabled = false;
        }
    }

    void RestoreMovement()
    {
        if (!paused) return;
        if (move != null) move.enabled = resumeMove;
        if (fist != null) fist.enabled = resumeFist;
        paused = false;
    }

    bool CreateFade()
    {
        if (fade == null)
        {
            var shader = Shader.Find("Nova/Visor UI");
            if (shader == null) return false;
            var go = new GameObject("Inspection Reset Fade", typeof(RectTransform), typeof(Canvas), typeof(Image));
            go.layer = LayerMask.NameToLayer("UI");
            var hud = origin.Camera.GetComponentInChildren<HelmetHUD>(true);
            var rect = go.GetComponent<RectTransform>();
            rect.SetParent(hud != null ? hud.transform : origin.Camera.transform, false);
            rect.sizeDelta = new Vector2(4000, 4000);
            if (hud == null) { rect.localPosition = Vector3.forward * .5f; rect.localScale = Vector3.one * .001f; }
            var canvas = go.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            canvas.worldCamera = origin.Camera;
            canvas.overrideSorting = true;
            canvas.sortingOrder = 32767;
            fadeMaterial = new Material(shader);
            fade = go.GetComponent<Image>();
            fade.material = fadeMaterial;
            fade.raycastTarget = false;
            fade.color = Color.clear;
        }
        fade.gameObject.SetActive(true);
        return true;
    }

    IEnumerator FadeTo(float alpha)
    {
        float start = fade.color.a;
        for (float elapsed = 0; elapsed < fadeSeconds; elapsed += Time.unscaledDeltaTime)
        {
            fade.color = new Color(0, 0, 0, Mathf.Lerp(start, alpha, elapsed / fadeSeconds));
            yield return null;
        }
        fade.color = new Color(0, 0, 0, alpha);
    }

    void OnDisable()
    {
        StopAllCoroutines();
        reset = null;
        RestoreMovement();
        if (fade != null) fade.gameObject.SetActive(false);
    }

    void OnDestroy()
    {
        if (fade != null) Destroy(fade.gameObject);
        if (fadeMaterial != null) Destroy(fadeMaterial);
    }
}
