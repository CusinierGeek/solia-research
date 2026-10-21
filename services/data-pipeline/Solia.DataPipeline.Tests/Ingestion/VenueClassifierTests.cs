using Solia.DataPipeline.Ingestion;

namespace Solia.DataPipeline.Tests.Ingestion;

public class VenueClassifierTests
{
    [Theory]
    [InlineData("pump")]
    [InlineData("raydium-launchpad")]
    [InlineData("meteora-launchpad")]
    public void Bonding_venues_are_not_amm(string venue)
    {
        Assert.False(VenueClassifier.IsAmm(venue));
    }

    [Theory]
    [InlineData("pump-amm")]
    [InlineData("raydium-cpmm")]
    [InlineData("meteora-damm-v1")]
    [InlineData("orca-whirlpool")]
    [InlineData("some-future-amm")] // unknown, but the "amm" hint classifies it as migrated
    public void Amm_venues_are_migrated(string venue)
    {
        Assert.True(VenueClassifier.IsAmm(venue));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("something-unknown")]
    public void Unknown_or_empty_defaults_to_not_amm(string? venue)
    {
        Assert.False(VenueClassifier.IsAmm(venue));
    }
}
