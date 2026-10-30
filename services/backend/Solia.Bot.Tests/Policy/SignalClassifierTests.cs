using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Policy;

namespace Solia.Bot.Tests.Policy;

public class SignalClassifierTests
{
    private static readonly BotOptions Opts = PolicyFixtures.Options();

    [Fact]
    public void Strong_pump_low_rug_high_confidence_is_buy()
    {
        Assert.Equal(TradeSignal.Buy, SignalClassifier.Classify(PolicyFixtures.Buyable(), Opts));
    }

    [Fact]
    public void Thresholds_are_inclusive_at_the_boundary()
    {
        // pump exactly at MinPumpProb and rug exactly at MaxRugProb still buys.
        ModelPrediction atBoundary = PolicyFixtures.Buyable() with
        {
            PumpProb = Opts.MinPumpProb,
            RugProb = Opts.MaxRugProb,
        };
        Assert.Equal(TradeSignal.Buy, SignalClassifier.Classify(atBoundary, Opts));
    }

    [Fact]
    public void Weak_pump_with_acceptable_rug_is_hold()
    {
        ModelPrediction weak = PolicyFixtures.Buyable() with { PumpProb = Opts.MinPumpProb - 0.01 };
        Assert.Equal(TradeSignal.Hold, SignalClassifier.Classify(weak, Opts));
    }

    [Fact]
    public void Elevated_rug_is_sell_even_with_a_strong_pump()
    {
        ModelPrediction risky = PolicyFixtures.Buyable() with { RugProb = 0.5 };
        Assert.Equal(TradeSignal.Sell, SignalClassifier.Classify(risky, Opts));
    }

    [Fact]
    public void Low_confidence_collapses_to_hold_whatever_the_heads_say()
    {
        ModelPrediction unsure = PolicyFixtures.Buyable() with { Confidence = 10, RugProb = 0.9 };
        Assert.Equal(TradeSignal.Hold, SignalClassifier.Classify(unsure, Opts));
    }

    [Theory]
    [MemberData(nameof(PolicyFixtures.Malformed), MemberType = typeof(PolicyFixtures))]
    public void A_malformed_prediction_is_hold(ModelPrediction malformed)
    {
        Assert.Equal(TradeSignal.Hold, SignalClassifier.Classify(malformed, Opts));
    }

    [Fact]
    public void Confidence_gate_is_inclusive_and_fails_closed_on_nan()
    {
        Assert.True(SignalClassifier.IsConfident(PolicyFixtures.Buyable() with { Confidence = Opts.MinConfidence }, Opts));
        Assert.False(SignalClassifier.IsConfident(PolicyFixtures.Buyable() with { Confidence = Opts.MinConfidence - 0.01 }, Opts));
        Assert.False(SignalClassifier.IsConfident(PolicyFixtures.Buyable() with { Confidence = double.NaN }, Opts));
    }

    [Fact]
    public void Null_arguments_throw()
    {
        Assert.Throws<ArgumentNullException>(() => SignalClassifier.Classify(null!, Opts));
        Assert.Throws<ArgumentNullException>(() => SignalClassifier.Classify(PolicyFixtures.Buyable(), null!));
        Assert.Throws<ArgumentNullException>(() => SignalClassifier.IsConfident(null!, Opts));
    }
}
