using NUnit.Framework;

namespace Nova.Gestures.Tests
{
    public class FrontPokeTests
    {
        FrontPoke poke;
        float time;

        [SetUp]
        public void SetUp()
        {
            poke = new FrontPoke();
            time = 0;
        }

        bool Touch(float depth, int target = 0) => poke.TryPress(target, depth, time += .1f, .008f, .025f, .35f);

        [Test]
        public void PassThroughAndWithdrawalOnlyClickOnce()
        {
            Assert.IsFalse(Touch(-.04f));
            Assert.IsTrue(Touch(-.005f));
            foreach (float depth in new[] { .01f, .04f, .02f, .005f, -.005f, -.02f, -.04f })
                Assert.IsFalse(Touch(depth));
            Assert.IsTrue(Touch(-.005f));
        }

        [Test]
        public void ReturningFromBehindDoesNotClick()
        {
            foreach (float depth in new[] { .04f, .02f, -.005f, -.02f, -.04f })
                Assert.IsFalse(Touch(depth));
            Assert.IsTrue(Touch(-.005f));
        }

        [Test]
        public void HoveringAtTheSurfaceDoesNotRepeat()
        {
            Touch(-.04f);
            Assert.IsTrue(Touch(-.005f));
            for (int i = 0; i < 20; i++) Assert.IsFalse(Touch(i % 2 == 0 ? -.009f : -.005f));
        }

        [Test]
        public void TrackingOrRequestResetRequiresAnotherFrontApproach()
        {
            Touch(-.04f);
            poke.Reset();
            Assert.IsFalse(Touch(-.005f));
            Assert.IsFalse(Touch(.01f));
            Assert.IsFalse(Touch(-.005f));
            Assert.IsFalse(Touch(-.04f));
            Assert.IsTrue(Touch(-.005f));
        }

        [Test]
        public void SlidingToAnotherButtonBehindTheSurfaceDoesNotClick()
        {
            Touch(-.04f);
            Assert.IsFalse(Touch(-.005f, 1));
            Assert.IsFalse(Touch(.01f, 1));
            Assert.IsFalse(Touch(-.005f, 1));
            Assert.IsFalse(Touch(-.04f, 1));
            Assert.IsTrue(Touch(-.005f, 1));
        }

        [Test]
        public void DeepJumpAndLeavingButtonsCancelThePress()
        {
            Touch(-.04f);
            Assert.IsFalse(Touch(.2f));
            Assert.IsFalse(Touch(-.005f));
            Touch(-.04f);
            Assert.IsFalse(Touch(-.02f, -1));
            Assert.IsFalse(Touch(-.005f));
        }

        [Test]
        public void FastRetouchStillRespectsDebounce()
        {
            Touch(-.04f);
            Assert.IsTrue(Touch(-.005f));
            Assert.IsFalse(Touch(-.04f));
            Assert.IsFalse(Touch(-.005f));
        }
    }
}
