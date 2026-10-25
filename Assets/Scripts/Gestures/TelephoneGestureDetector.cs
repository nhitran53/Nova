using System;
using System.Collections.Generic;
using Nova.Gestures;
using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Hands;

// Read the telephone pose when other hand controls are idle.
public class TelephoneGestureDetector : MonoBehaviour
{
    public enum Hand { Left, Right }

    [SerializeField] Hand hand = Hand.Right;
    [Tooltip("Fingertip-to-palm distance as a fraction of hand size (wrist to middle knuckle). Below Curl Enter is curled.")]
    [SerializeField, Range(0.2f, 1)] float curlEnter = 0.5f;
    [Tooltip("Keep above Curl Enter to prevent flicker.")]
    [SerializeField, Range(0.2f, 1.2f)] float curlExit = 0.65f;
    [Tooltip("Thumb and pinky count as extended above this fraction of hand size.")]
    [SerializeField, Range(0.5f, 2)] float extendEnter = 0.9f;
    [Tooltip("Keep below Extend Enter to prevent flicker.")]
    [SerializeField, Range(0.4f, 2)] float extendExit = 0.75f;
    [Tooltip("How long the pose has to be held before it starts.")]
    [SerializeField, Min(0)] float holdTime = 0.35f;
    [Tooltip("How long the pose has to be lost before it ends.")]
    [SerializeField, Min(0)] float releaseTime = 0.15f;
    [Tooltip("After it ends, no new start for this long.")]
    [SerializeField, Min(0)] float cooldown = 0.5f;
    [SerializeField] bool diagnostics = true;
    [SerializeField] UnityEvent onTelephoneStart = new UnityEvent();
    [SerializeField] UnityEvent onTelephoneEnd = new UnityEvent();

    public UnityEvent OnTelephoneStart => onTelephoneStart;
    public UnityEvent OnTelephoneEnd => onTelephoneEnd;
    public bool Active { get; private set; }
    public int StartCount { get; private set; }
    public string Diagnostic { get; private set; }

    // True while any detector is active. Other scripts can read it or subscribe to changes.
    public static bool AnyActive => activeCount > 0;
    public static event Action<bool> AnyActiveChanged;

    static int activeCount;
    static readonly List<XRHandSubsystem> subsystems = new List<XRHandSubsystem>();

    TelephoneGestureLogic logic;
    XRHandSubsystem hands;
    Hand loggedHand;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void ResetStatics()
    {
        activeCount = 0;
        AnyActiveChanged = null;
        subsystems.Clear();
    }

    void OnEnable()
    {
        logic = new TelephoneGestureLogic(BuildSettings());
        loggedHand = hand;
        Diagnostic = null;
    }

    void OnDisable()
    {
        EndGesture(false);
        logic = null;
        hands = null;
    }

    void OnValidate()
    {
        curlExit = Mathf.Max(curlExit, curlEnter);
        extendExit = Mathf.Min(extendExit, extendEnter);
        // The curled range must stay below the extended range.
        extendExit = Mathf.Max(extendExit, curlExit);
        extendEnter = Mathf.Max(extendEnter, extendExit);
        if (logic != null) logic.Configure(BuildSettings());
    }

    TelephoneGestureSettings BuildSettings() => new TelephoneGestureSettings
    {
        CurledEnter = curlEnter, CurledExit = curlExit,
        ExtendedEnter = extendEnter, ExtendedExit = extendExit,
        HoldTime = holdTime, ReleaseTime = releaseTime, Cooldown = cooldown
    };

    void Update()
    {
        // Switching hands at runtime drops the current gesture.
        if (hand != loggedHand)
        {
            EndGesture(true);
            logic.Reset();
            loggedHand = hand;
        }

        var sample = ReadHand(out string reason);
        var result = logic.Tick(Time.unscaledDeltaTime, sample);
        if (result == TelephoneGestureEvent.Started) StartGesture();
        else if (result == TelephoneGestureEvent.Ended) EndGesture(true);

        string state = Active ? "ACTIVE" : reason ?? (logic.PoseMatches ? "HOLDING" : "IDLE");
        Report($"{hand.ToString().ToUpperInvariant()} / {state} / {StartCount}");
    }

    TelephoneHandSample ReadHand(out string reason)
    {
        reason = null;
        var sample = new TelephoneHandSample();
        bool right = hand == Hand.Right;
        if (!TryGetSubsystem()) { reason = "NO HAND SUBSYSTEM"; return sample; }

        // System gestures and the menu gesture own the hand, same as the fist and pinch scripts.
        var aim = right ? MetaAimHand.right : MetaAimHand.left;
        if (aim != null && aim.added &&
            ((MetaAimFlags)aim.aimFlags.ReadValue() & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) != 0)
        {
            sample.Blocked = true;
            reason = "SYSTEM GESTURE";
            return sample;
        }

        var joints = right ? hands.rightHand : hands.leftHand;
        if (!joints.isTracked ||
            !joints.GetJoint(XRHandJointID.Palm).TryGetPose(out Pose palm) ||
            !joints.GetJoint(XRHandJointID.Wrist).TryGetPose(out Pose wrist) ||
            !joints.GetJoint(XRHandJointID.MiddleProximal).TryGetPose(out Pose knuckle))
        {
            reason = "NOT TRACKED";
            return sample;
        }
        float size = Vector3.Distance(wrist.position, knuckle.position);
        if (size < 0.03f) { reason = "NOT TRACKED"; return sample; }
        if (!TryReach(joints, XRHandJointID.IndexTip, palm, size, out sample.IndexReach) ||
            !TryReach(joints, XRHandJointID.MiddleTip, palm, size, out sample.MiddleReach) ||
            !TryReach(joints, XRHandJointID.RingTip, palm, size, out sample.RingReach) ||
            !TryReach(joints, XRHandJointID.ThumbTip, palm, size, out sample.ThumbReach) ||
            !TryReach(joints, XRHandJointID.LittleTip, palm, size, out sample.PinkyReach))
        {
            reason = "NOT TRACKED";
            return sample;
        }
        sample.Tracked = true;

        if (FistLocomotion.BlocksPinch(right))
        {
            sample.Blocked = true;
            reason = FistLocomotion.Walking ? "WALKING" : "FIST";
        }
        else if (MinimapNavigationPanel.FingerNearMap)
        {
            sample.Blocked = true;
            reason = "MAP TOUCH";
        }
        return sample;
    }

    static bool TryReach(XRHand joints, XRHandJointID tipId, Pose palm, float size, out float reach)
    {
        reach = 0;
        if (!joints.GetJoint(tipId).TryGetPose(out Pose tip)) return false;
        reach = Vector3.Distance(tip.position, palm.position) / size;
        return true;
    }

    bool TryGetSubsystem()
    {
        if (hands == null || !hands.running)
        {
            hands = null;
            SubsystemManager.GetSubsystems(subsystems);
            foreach (var s in subsystems) if (s.running) { hands = s; break; }
        }
        return hands != null;
    }

    void StartGesture()
    {
        if (Active) return;
        Active = true;
        StartCount++;
        SetActiveCount(activeCount + 1);
        onTelephoneStart.Invoke();
    }

    void EndGesture(bool invokeEvent)
    {
        if (!Active) return;
        Active = false;
        SetActiveCount(activeCount - 1);
        if (invokeEvent) onTelephoneEnd.Invoke();
    }

    static void SetActiveCount(int value)
    {
        bool before = AnyActive;
        activeCount = Mathf.Max(0, value);
        if (before != AnyActive) AnyActiveChanged?.Invoke(AnyActive);
    }

    void Report(string message)
    {
        if (message == Diagnostic) return;
        Diagnostic = message;
        if (diagnostics) Debug.Log("Telephone: " + message, this);
    }

    [ContextMenu("Report Telephone State")]
    public void ReportState() => Debug.Log("Telephone: " + (Diagnostic ?? "WAITING FOR HAND DATA"), this);
}
