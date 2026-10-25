using System.Collections.Generic;
using TMPro;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.XR.Hands;
using UnityEngine.XR.Interaction.Toolkit.Inputs.Readers;
using UnityEngine.XR.Interaction.Toolkit.Locomotion.Movement;

// Hold a right fist to walk. Open it or lose tracking to stop.
[DefaultExecutionOrder(-100)]   // Read fist input before pinch and movement.
[DisallowMultipleComponent]
public class FistLocomotion : MonoBehaviour
{
    [Tooltip("Leave empty to use the one on this XR Origin.")]
    [SerializeField] ContinuousMoveProvider moveProvider;
    [SerializeField, Min(0.1f)] float walkSpeed = 1.2f;
    [Tooltip("How long the fist has to be held before walking starts.")]
    [SerializeField, Min(0)] float holdTime = 0.4f;
    [Tooltip("Curl threshold as a fraction of hand size.")]
    [SerializeField, Range(0.2f, 1)] float curlEnter = 0.5f;
    [Tooltip("Release threshold. Keep above Curl Enter to prevent flicker.")]
    [SerializeField, Range(0.2f, 1.2f)] float curlExit = 0.65f;

    [Header("Indicator")]
    [Tooltip("Leave empty to use the Helmet HUD on this rig.")]
    [SerializeField] HelmetHUD helmet;
    [SerializeField] Vector2 indicatorSlot = new Vector2(0, -160);

    public static bool Walking { get; private set; }
    public static bool RightFist { get; private set; }
    public static bool BlocksPinch(bool rightHand) => Walking || (rightHand && RightFist);

    static readonly List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();
    static readonly XRHandJointID[] Tips = { XRHandJointID.IndexTip, XRHandJointID.MiddleTip, XRHandJointID.RingTip, XRHandJointID.LittleTip };

    XRHandSubsystem hands;
    XRInputValueReader.InputSourceMode originalMode;
    readonly bool[] curled = new bool[4];
    float fistSince = -1;
    TMP_Text indicator;
    bool warnedSpeed;

    void OnEnable()
    {
        if (moveProvider == null) moveProvider = GetComponentInChildren<ContinuousMoveProvider>(true);
        if (moveProvider == null)
        {
            Debug.LogError("Fist walking needs the Continuous Move Provider on the XR Origin.", this);
            enabled = false;
            return;
        }
        originalMode = moveProvider.rightHandMoveInput.inputSourceMode;
        moveProvider.rightHandMoveInput.inputSourceMode = XRInputValueReader.InputSourceMode.ManualValue;
        moveProvider.rightHandMoveInput.manualValue = Vector2.zero;
    }

    void OnDisable()
    {
        SetWalking(false);
        RightFist = false;
        fistSince = -1;
        if (moveProvider != null) moveProvider.rightHandMoveInput.inputSourceMode = originalMode;
    }

    void Update()
    {
        bool fist = ReadFist();
        RightFist = fist;
        // Left pinch blocks new walks only.
        var left = MetaAimHand.left;
        bool leftPinch = left != null && left.added && left.indexPressed.isPressed;
        if (!fist || (!Walking && (leftPinch || MinimapNavigationPanel.FingerNearMap))) fistSince = -1;
        else if (fistSince < 0) fistSince = Time.unscaledTime;
        SetWalking(fist && Time.unscaledTime - fistSince >= holdTime);
    }

    bool ReadFist()
    {
        if (hands == null || !hands.running)
        {
            hands = null;
            SubsystemManager.GetSubsystems(subsystems);
            foreach (var s in subsystems) if (s.running) { hands = s; break; }
        }
        var aim = MetaAimHand.right;
        if (aim != null && aim.added)
        {
            var flags = (MetaAimFlags)aim.aimFlags.ReadValue();
            if ((flags & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) != 0) return ResetCurl();
        }
        if (hands == null) return ResetCurl();
        var hand = hands.rightHand;
        if (!hand.isTracked ||
            !hand.GetJoint(XRHandJointID.Palm).TryGetPose(out Pose palm) ||
            !hand.GetJoint(XRHandJointID.Wrist).TryGetPose(out Pose wrist) ||
            !hand.GetJoint(XRHandJointID.MiddleProximal).TryGetPose(out Pose knuckle))
            return ResetCurl();
        float size = Vector3.Distance(wrist.position, knuckle.position);
        if (size < 0.03f) return ResetCurl();

        bool all = true;
        for (int i = 0; i < Tips.Length; i++)
        {
            if (!hand.GetJoint(Tips[i]).TryGetPose(out Pose tip)) return ResetCurl();
            float share = Vector3.Distance(tip.position, palm.position) / size;
            curled[i] = curled[i] ? share < curlExit : share < curlEnter;
            all &= curled[i];
        }
        return all;
    }

    bool ResetCurl()
    {
        for (int i = 0; i < curled.Length; i++) curled[i] = false;
        return false;
    }

    void SetWalking(bool on)
    {
        if (moveProvider != null)
        {
            float speed = moveProvider.moveSpeed;
            if (on && speed < walkSpeed && !warnedSpeed)
            {
                warnedSpeed = true;
                Debug.LogWarning($"Continuous Move Provider speed is {speed} m/s, below the {walkSpeed} m/s fist walk.", this);
            }
            moveProvider.rightHandMoveInput.manualValue = on ? new Vector2(0, Mathf.Clamp01(walkSpeed / Mathf.Max(0.01f, speed))) : Vector2.zero;
        }
        if (on == Walking && (indicator == null || indicator.gameObject.activeSelf == on)) return;
        Walking = on;
        ShowIndicator(on);
    }

    void ShowIndicator(bool on)
    {
        if (indicator == null)
        {
            if (!on) return;
            if (helmet == null) helmet = GetComponentInParent<XROrigin>()?.GetComponentInChildren<HelmetHUD>(true);
            if (helmet == null) return;
            indicator = HudText.Create(helmet, "Walk Indicator", indicatorSlot, new Vector2(240, 36), 22);
            indicator.text = "WALK ACTIVE";
            indicator.color = new Color(0.35f, 0.9f, 1);
        }
        indicator.gameObject.SetActive(on);
    }
}
