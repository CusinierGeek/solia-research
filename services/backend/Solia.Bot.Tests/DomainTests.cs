using System.Reflection;
using Solia.Bot.Domain;

namespace Solia.Bot.Tests;

public class DomainTests
{
    private static ModelPrediction Prediction(Dictionary<string, double?> features) => new()
    {
        Mint = "mint1",
        Ts = new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero),
        Horizon = "1m",
        PumpProb = 0.7,
        RugProb = 0.1,
        ExpectedReturn = 0.35,
        Confidence = 80,
        Volatility = 0.05,
        ModelVersion = "v1",
        Features = features,
    };

    [Fact]
    public void Predictions_with_equal_content_are_equal()
    {
        ModelPrediction a = Prediction(new() { ["mc"] = 1.5, ["holders"] = null });
        ModelPrediction b = Prediction(new() { ["holders"] = null, ["mc"] = 1.5 }); // other instance, other order

        Assert.Equal(a, b);
        Assert.True(a == b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.Equal<BotEvent>(new BotEvent.Prediction(a), new BotEvent.Prediction(b));
    }

    [Theory]
    [InlineData("mc", 2.0)]     // different value
    [InlineData("volume", 1.5)] // different key
    public void Predictions_with_different_features_are_not_equal(string key, double value)
    {
        ModelPrediction a = Prediction(new() { ["mc"] = 1.5 });
        ModelPrediction b = Prediction(new() { [key] = value });

        Assert.NotEqual(a, b);
    }

    [Fact]
    public void Predictions_with_different_scalar_fields_are_not_equal()
    {
        ModelPrediction a = Prediction(new() { ["mc"] = 1.5 });

        Assert.NotEqual(a, a with { PumpProb = 0.71 });
        Assert.NotEqual(a, a with { Volatility = null });
        Assert.NotEqual(a, a with { Features = new Dictionary<string, double?> { ["mc"] = 1.5, ["x"] = 0 } });
        Assert.False(a.Equals(null));
    }

    [Fact]
    public void Bot_event_union_has_exactly_the_two_nested_cases()
    {
        Type[] cases = typeof(BotEvent).Assembly.GetTypes()
            .Where(t => t.IsSubclassOf(typeof(BotEvent)))
            .OrderBy(t => t.Name)
            .ToArray();

        Assert.Equal([typeof(BotEvent.Prediction), typeof(BotEvent.Price)], cases);
        Assert.All(cases, t => Assert.True(t.IsSealed));
    }

    [Fact]
    public void Bot_event_has_an_abstract_member_other_assemblies_cannot_implement()
    {
        // A record's protected copy constructor would let outside code derive; a private protected
        // abstract member is what stops it (no subtype outside this assembly can implement it).
        Assert.Contains(
            typeof(BotEvent).GetMethods(BindingFlags.Instance | BindingFlags.NonPublic),
            m => m.IsAbstract && m.IsFamilyAndAssembly);
    }
}
