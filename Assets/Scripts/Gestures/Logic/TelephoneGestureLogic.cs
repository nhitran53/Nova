using System;

namespace Nova.Gestures
{
    public enum TelephoneGestureEvent { None, Started, Ended }

    // Reach is fingertip distance from the palm divided by hand size.
    // Lower values mean a curled finger.
    public struct TelephoneHandSample
    {
        public bool Tracked;
        // Another control is using this hand.
        public bool Blocked;
        public float IndexReach, MiddleReach, RingReach, ThumbReach, PinkyReach;
    }

    public struct TelephoneGestureSettings
    {
        // Reach below CurledEnter counts as curled until it rises above CurledExit.
        public float CurledEnter, CurledExit;
        // Reach above ExtendedEnter counts as extended until it falls below ExtendedExit.
        public float ExtendedEnter, ExtendedExit;
        // Pose must be held this long before it starts.
        public float HoldTime;
        // Pose must be lost this long before it ends. Blocking ends it at once.
        public float ReleaseTime;
        // After it ends, no new start for this long.
        public float Cooldown;

        public static TelephoneGestureSettings Default => new TelephoneGestureSettings
        {
            CurledEnter = 0.5f, CurledExit = 0.65f,
            ExtendedEnter = 0.9f, ExtendedExit = 0.75f,
            HoldTime = 0.35f, ReleaseTime = 0.15f, Cooldown = 0.5f
        };
    }

    // Telephone pose: index, middle and ring curled, thumb and pinky extended.
    public sealed class TelephoneGestureLogic
    {
        TelephoneGestureSettings settings;
        bool indexCurled, middleCurled, ringCurled, thumbExtended, pinkyExtended;
        float held, lost, cooldownLeft;

        public TelephoneGestureLogic(TelephoneGestureSettings settings)
        {
            Configure(settings);
        }

        public bool Active { get; private set; }
        // The fingers match before the hold timer completes.
        public bool PoseMatches { get; private set; }

        public void Configure(TelephoneGestureSettings value)
        {
            if (value.CurledExit < value.CurledEnter)
                throw new ArgumentException("CurledExit must be at least CurledEnter.");
            if (value.ExtendedExit > value.ExtendedEnter)
                throw new ArgumentException("ExtendedExit must be at most ExtendedEnter.");
            if (value.CurledExit > value.ExtendedExit)
                throw new ArgumentException("Curled and extended ranges must not overlap.");
            if (value.HoldTime < 0 || value.ReleaseTime < 0 || value.Cooldown < 0)
                throw new ArgumentException("Times must not be negative.");
            settings = value;
        }

        // Clear the pose and timers without sending an end event.
        public void Reset()
        {
            Active = PoseMatches = false;
            indexCurled = middleCurled = ringCurled = thumbExtended = pinkyExtended = false;
            held = lost = cooldownLeft = 0;
        }

        public TelephoneGestureEvent Tick(float dt, TelephoneHandSample sample)
        {
            if (dt < 0 || float.IsNaN(dt)) dt = 0;
            if (cooldownLeft > 0) cooldownLeft = Math.Max(0, cooldownLeft - dt);

            if (!sample.Tracked || float.IsNaN(sample.IndexReach + sample.MiddleReach + sample.RingReach + sample.ThumbReach + sample.PinkyReach))
            {
                indexCurled = middleCurled = ringCurled = thumbExtended = pinkyExtended = false;
                PoseMatches = false;
            }
            else
            {
                indexCurled = Curled(indexCurled, sample.IndexReach);
                middleCurled = Curled(middleCurled, sample.MiddleReach);
                ringCurled = Curled(ringCurled, sample.RingReach);
                thumbExtended = Extended(thumbExtended, sample.ThumbReach);
                pinkyExtended = Extended(pinkyExtended, sample.PinkyReach);
                PoseMatches = indexCurled && middleCurled && ringCurled && thumbExtended && pinkyExtended;
            }

            if (sample.Blocked)
            {
                held = lost = 0;
                if (!Active) return TelephoneGestureEvent.None;
                return End();
            }

            if (Active)
            {
                if (PoseMatches) { lost = 0; return TelephoneGestureEvent.None; }
                lost += dt;
                return lost >= settings.ReleaseTime ? End() : TelephoneGestureEvent.None;
            }

            if (!PoseMatches) { held = 0; return TelephoneGestureEvent.None; }
            held += dt;
            if (held < settings.HoldTime || cooldownLeft > 0) return TelephoneGestureEvent.None;
            Active = true;
            held = lost = 0;
            return TelephoneGestureEvent.Started;
        }

        TelephoneGestureEvent End()
        {
            Active = false;
            held = lost = 0;
            cooldownLeft = settings.Cooldown;
            return TelephoneGestureEvent.Ended;
        }

        bool Curled(bool was, float reach) => was ? reach < settings.CurledExit : reach < settings.CurledEnter;
        bool Extended(bool was, float reach) => was ? reach > settings.ExtendedExit : reach > settings.ExtendedEnter;
    }
}
