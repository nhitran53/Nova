using System;

namespace Nova.Gestures
{
    public sealed class FrontPoke
    {
        int armedTarget = -1;
        bool hasSample;
        float previousDepth, nextPress;

        public void Reset()
        {
            armedTarget = -1;
            hasSample = false;
        }

        public bool TryPress(int target, float depth, float now, float pressDistance, float releaseDistance, float debounce)
        {
            if (target < 0 || float.IsNaN(depth) || float.IsInfinity(depth) || float.IsNaN(now) || float.IsInfinity(now))
            {
                Reset();
                return false;
            }
            pressDistance = Math.Max(0, pressDistance);
            releaseDistance = Math.Max(pressDistance + .005f, releaseDistance);
            bool crossing = hasSample && depth > previousDepth && previousDepth < -pressDistance && depth >= -pressDistance;
            previousDepth = depth;
            hasSample = true;

            // Negative depth is in front. Pull back fully before another press.
            if (depth <= -releaseDistance && now >= nextPress) armedTarget = target;
            else if (armedTarget != target || depth > releaseDistance) armedTarget = -1;
            if (!crossing || armedTarget != target || now < nextPress) return false;

            armedTarget = -1;
            nextPress = now + Math.Max(0, debounce);
            return true;
        }
    }
}
