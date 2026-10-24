using UnityEngine;
using UnityEngine.Events;
using UnityEngine.XR.Hands;

public class PinchToggle : MonoBehaviour
{
    public enum Hand { Left, Right }

    [SerializeField] Hand hand = Hand.Right;
    [SerializeField, Min(0)] float debounce = 0.35f;
    [SerializeField, Min(0)] float releaseTime = 0.08f;
    [SerializeField] bool enableToggle = true;
    [SerializeField] bool diagnostics = true;
    [SerializeField] HelmetHUD helmet;
    [SerializeField] UnityEvent onPinch = new UnityEvent();

    public bool InputAvailable { get; private set; }
    public bool Pinching { get; private set; }
    public int ToggleCount { get; private set; }
    public string Diagnostic { get; private set; }
    public UnityEvent OnPinch => onPinch;

    bool armed, wasPressed;
    float nextToggle;
    float releasedAt = -1;
    string reportedState;
    int reportedCount = -1;
    Hand reportedHand;

    void OnEnable()
    {
        armed = wasPressed = false;
        nextToggle = 0;
        releasedAt = -1;
        Diagnostic = null;
        reportedState = null;
    }

    void OnDisable()
    {
        armed = wasPressed = false;
        InputAvailable = Pinching = false;
        releasedAt = -1;
    }

    void Update()
    {
        ReadHand(hand == Hand.Left ? MetaAimHand.left : MetaAimHand.right);
    }

    void ReadHand(MetaAimHand device)
    {
        var flags = device != null ? (MetaAimFlags)device.aimFlags.ReadValue() : MetaAimFlags.None;
        bool tracked = device != null && device.added && device.isTracked.isPressed;
        // Valid describes the aim ray, which is not required to read the native pinch button.
        bool valid = tracked && (flags & MetaAimFlags.Computed) != 0 &&
            (flags & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) == 0;
        string reason = device == null ? "NO AIM DEVICE" : !tracked ? "NOT TRACKED" :
            (flags & (MetaAimFlags.SystemGesture | MetaAimFlags.MenuPressed)) != 0 ? "SYSTEM GESTURE" : "NO AIM DATA";
        ProcessInput(valid, valid && device.indexPressed.isPressed, Time.unscaledTime, reason);
    }

    void ProcessInput(bool available, bool pressed, float now, string reason)
    {
        InputAvailable = available;
        Pinching = available && pressed;
        if (!available)
        {
            armed = wasPressed = false;
            releasedAt = -1;
        }
        else
        {
            if (!pressed)
            {
                if (wasPressed || releasedAt < 0) releasedAt = now;
                if (now - releasedAt >= releaseTime) armed = true;
            }
            else if (!wasPressed && releasedAt >= 0 && now - releasedAt >= releaseTime)
                armed = true;
            if (pressed && !wasPressed && armed)
            {
                armed = false;
                if (enableToggle && now >= nextToggle)
                {
                    nextToggle = now + debounce;
                    ToggleCount++;
                    onPinch.Invoke();
                }
            }
            if (pressed) releasedAt = -1;
            wasPressed = pressed;
        }
        string state = available ? pressed ? "PINCHED" : "RELEASED" : reason;
        if (state == reportedState && ToggleCount == reportedCount && hand == reportedHand) return;
        reportedState = state;
        reportedCount = ToggleCount;
        reportedHand = hand;
        string message = $"{hand.ToString().ToUpperInvariant()} / {state} / {ToggleCount}";
        if (message == Diagnostic) return;
        Diagnostic = message;
        if (diagnostics)
        {
            Debug.Log("Pinch: " + message, this);
            if (helmet != null) helmet.ReportPinch(message);
        }
    }

    [ContextMenu("Report Pinch State")]
    public void ReportState()
    {
        Debug.Log("Pinch: " + Diagnostic, this);
        if (helmet != null) helmet.ReportPinch(Diagnostic ?? "WAITING FOR AIM DATA");
    }
}
