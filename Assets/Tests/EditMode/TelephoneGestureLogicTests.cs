using System;
using NUnit.Framework;

namespace Nova.Gestures.Tests
{
    public class TelephoneGestureLogicTests
    {
        const float Dt = 0.05f;

        static TelephoneGestureSettings Settings => new TelephoneGestureSettings
        {
            CurledEnter = 0.5f, CurledExit = 0.65f,
            ExtendedEnter = 0.9f, ExtendedExit = 0.75f,
            HoldTime = 0.3f, ReleaseTime = 0.2f, Cooldown = 1f
        };

        static TelephoneHandSample Phone => new TelephoneHandSample
        {
            Tracked = true, IndexReach = 0.3f, MiddleReach = 0.3f, RingReach = 0.3f, ThumbReach = 1.1f, PinkyReach = 1.1f
        };

        static TelephoneHandSample Open => new TelephoneHandSample
        {
            Tracked = true, IndexReach = 1.2f, MiddleReach = 1.2f, RingReach = 1.2f, ThumbReach = 1.1f, PinkyReach = 1.1f
        };

        static TelephoneHandSample Fist => new TelephoneHandSample
        {
            Tracked = true, IndexReach = 0.3f, MiddleReach = 0.3f, RingReach = 0.3f, ThumbReach = 0.4f, PinkyReach = 0.3f
        };

        // Ticks for the given seconds and returns every non-None event.
        static TelephoneGestureEvent[] Run(TelephoneGestureLogic logic, TelephoneHandSample sample, float seconds)
        {
            var events = new System.Collections.Generic.List<TelephoneGestureEvent>();
            int steps = (int)Math.Round(seconds / Dt);
            for (int i = 0; i < steps; i++)
            {
                var e = logic.Tick(Dt, sample);
                if (e != TelephoneGestureEvent.None) events.Add(e);
            }
            return events.ToArray();
        }

        [Test]
        public void StartsAfterHoldTime()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Assert.IsEmpty(Run(logic, Phone, 0.2f));
            Assert.IsFalse(logic.Active);
            Assert.AreEqual(new[] { TelephoneGestureEvent.Started }, Run(logic, Phone, 0.2f));
            Assert.IsTrue(logic.Active);
        }

        [Test]
        public void StartsOnlyOnceWhileHeld()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Assert.AreEqual(new[] { TelephoneGestureEvent.Started }, Run(logic, Phone, 3f));
        }

        [Test]
        public void ShortPoseResetsHoldTimer()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.2f);
            Run(logic, Open, 0.05f);
            Assert.IsEmpty(Run(logic, Phone, 0.2f));
            Assert.IsFalse(logic.Active);
        }

        [Test]
        public void EndsAfterReleaseTimeNotBefore()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.4f);
            Assert.IsEmpty(Run(logic, Open, 0.1f));
            Assert.IsTrue(logic.Active);
            Assert.AreEqual(new[] { TelephoneGestureEvent.Ended }, Run(logic, Open, 0.2f));
            Assert.IsFalse(logic.Active);
        }

        [Test]
        public void BriefDropoutKeepsGestureActive()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.4f);
            Run(logic, Open, 0.1f);
            Assert.IsEmpty(Run(logic, Phone, 1f));
            Assert.IsTrue(logic.Active);
            // The release timer restarted when the pose came back.
            Assert.IsEmpty(Run(logic, Open, 0.1f));
            Assert.IsTrue(logic.Active);
        }

        [Test]
        public void TrackingLossEndsAfterReleaseTime()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.4f);
            var lostTracking = Phone;
            lostTracking.Tracked = false;
            Assert.AreEqual(new[] { TelephoneGestureEvent.Ended }, Run(logic, lostTracking, 0.3f));
        }

        [Test]
        public void CooldownDelaysRestart()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.4f);
            Run(logic, Open, 0.3f);
            Assert.IsFalse(logic.Active);
            Assert.IsEmpty(Run(logic, Phone, 0.5f));
            Assert.IsFalse(logic.Active);
            Assert.AreEqual(new[] { TelephoneGestureEvent.Started }, Run(logic, Phone, 0.5f));
        }

        [Test]
        public void ZeroCooldownAllowsImmediateRestart()
        {
            var s = Settings;
            s.Cooldown = 0;
            var logic = new TelephoneGestureLogic(s);
            Run(logic, Phone, 0.4f);
            Run(logic, Open, 0.3f);
            Assert.AreEqual(new[] { TelephoneGestureEvent.Started }, Run(logic, Phone, 0.4f));
        }

        [Test]
        public void FistNeverStartsEvenWithThumbCurled()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Assert.IsEmpty(Run(logic, Fist, 3f));
            Assert.IsFalse(logic.PoseMatches);
        }

        [Test]
        public void OpenHandNeverStarts()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Assert.IsEmpty(Run(logic, Open, 3f));
        }

        [Test]
        public void PointingIndexOrCurledPinkyOrCurledThumbDoNotMatch()
        {
            var logic = new TelephoneGestureLogic(Settings);
            var index = Phone; index.IndexReach = 1.2f;
            var pinky = Phone; pinky.PinkyReach = 0.4f;
            var thumb = Phone; thumb.ThumbReach = 0.4f;
            var middle = Phone; middle.MiddleReach = 1.2f;
            var ring = Phone; ring.RingReach = 1.2f;
            foreach (var sample in new[] { index, pinky, thumb, middle, ring })
            {
                logic.Reset();
                Assert.IsEmpty(Run(logic, sample, 2f));
                Assert.IsFalse(logic.PoseMatches);
            }
        }

        [Test]
        public void BlockedNeverStarts()
        {
            var logic = new TelephoneGestureLogic(Settings);
            var blocked = Phone;
            blocked.Blocked = true;
            Assert.IsEmpty(Run(logic, blocked, 3f));
        }

        [Test]
        public void BlockingEndsActiveGestureImmediately()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.4f);
            var blocked = Phone;
            blocked.Blocked = true;
            Assert.AreEqual(TelephoneGestureEvent.Ended, logic.Tick(Dt, blocked));
            Assert.IsFalse(logic.Active);
        }

        [Test]
        public void BlockingResetsHoldTimerAndStartsCooldown()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.2f);
            var blocked = Phone;
            blocked.Blocked = true;
            Run(logic, blocked, 0.05f);
            // Hold restarts from zero after unblocking.
            Assert.IsEmpty(Run(logic, Phone, 0.2f));
            Assert.AreEqual(new[] { TelephoneGestureEvent.Started }, Run(logic, Phone, 0.2f));

            logic.Tick(Dt, blocked);
            Assert.IsEmpty(Run(logic, Phone, 0.6f));
            Assert.IsFalse(logic.Active);
        }

        [Test]
        public void HysteresisHoldsCurledBetweenEnterAndExit()
        {
            var s = Settings;
            s.HoldTime = 0;
            var logic = new TelephoneGestureLogic(s);
            logic.Tick(Dt, Phone);
            Assert.IsTrue(logic.PoseMatches);
            var between = Phone; between.IndexReach = 0.6f;   // above enter 0.5, below exit 0.65
            logic.Tick(Dt, between);
            Assert.IsTrue(logic.PoseMatches);
            var above = Phone; above.IndexReach = 0.7f;
            logic.Tick(Dt, above);
            Assert.IsFalse(logic.PoseMatches);
            // Once released, it needs to drop below enter, not just exit.
            logic.Tick(Dt, between);
            Assert.IsFalse(logic.PoseMatches);
        }

        [Test]
        public void HysteresisHoldsExtendedBetweenEnterAndExit()
        {
            var s = Settings;
            s.HoldTime = 0;
            var logic = new TelephoneGestureLogic(s);
            var between = Phone; between.PinkyReach = 0.8f;   // below enter 0.9, above exit 0.75
            logic.Tick(Dt, between);
            Assert.IsFalse(logic.PoseMatches);
            logic.Tick(Dt, Phone);
            Assert.IsTrue(logic.PoseMatches);
            logic.Tick(Dt, between);
            Assert.IsTrue(logic.PoseMatches);
            var below = Phone; below.PinkyReach = 0.7f;
            logic.Tick(Dt, below);
            Assert.IsFalse(logic.PoseMatches);
        }

        [Test]
        public void NaNReachIsTreatedAsLostPose()
        {
            var logic = new TelephoneGestureLogic(Settings);
            var bad = Phone; bad.RingReach = float.NaN;
            Assert.IsEmpty(Run(logic, bad, 2f));
            Assert.IsFalse(logic.PoseMatches);
        }

        [Test]
        public void NegativeDeltaTimeIsIgnored()
        {
            var logic = new TelephoneGestureLogic(Settings);
            for (int i = 0; i < 20; i++) Assert.AreEqual(TelephoneGestureEvent.None, logic.Tick(-1f, Phone));
        }

        [Test]
        public void ResetEndsSilentlyAndClearsTimers()
        {
            var logic = new TelephoneGestureLogic(Settings);
            Run(logic, Phone, 0.4f);
            logic.Reset();
            Assert.IsFalse(logic.Active);
            Assert.IsEmpty(Run(logic, Phone, 0.2f));
            Assert.AreEqual(new[] { TelephoneGestureEvent.Started }, Run(logic, Phone, 0.2f));
        }

        [Test]
        public void ConfigureRejectsInvalidSettings()
        {
            var s = Settings; s.CurledExit = 0.4f;
            Assert.Throws<ArgumentException>(() => new TelephoneGestureLogic(s));
            s = Settings; s.ExtendedExit = 0.95f;
            Assert.Throws<ArgumentException>(() => new TelephoneGestureLogic(s));
            s = Settings; s.CurledExit = 0.8f;
            Assert.Throws<ArgumentException>(() => new TelephoneGestureLogic(s));
            s = Settings; s.Cooldown = -1;
            Assert.Throws<ArgumentException>(() => new TelephoneGestureLogic(s));
            Assert.DoesNotThrow(() => new TelephoneGestureLogic(TelephoneGestureSettings.Default));
        }
    }
}
