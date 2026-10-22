using Solia.Bot.Domain;
using Solia.Bot.Policy;

namespace Solia.Bot.Tests.Policy;

public class PredictionValidationTests
{
    [Fact]
    public void A_buyable_prediction_is_well_formed()
    {
        Assert.True(PredictionValidation.IsWellFormed(PolicyFixtures.Buyable()));
    }

    [Fact]
    public void Range_bounds_are_inclusive()
    {
        ModelPrediction atBounds = PolicyFixtures.Buyable() with
        {
            PumpProb = 1.0,
            RugProb = 0.0,
            Confidence = 100,
            ExpectedReturn = -1.0, // a total loss is the floor of a long position's return
        };
        Assert.True(PredictionValidation.IsWellFormed(atBounds));
    }

    [Theory]
    [MemberData(nameof(PolicyFixtures.Malformed), MemberType = typeof(PolicyFixtures))]
    public void Nan_infinite_or_out_of_range_fields_are_malformed(ModelPrediction malformed)
    {
        Assert.False(PredictionValidation.IsWellFormed(malformed));
    }

    [Fact]
    public void Null_throws()
    {
        Assert.Throws<ArgumentNullException>(() => PredictionValidation.IsWellFormed(null!));
    }
}
