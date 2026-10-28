namespace Solia.DataPipeline.Features;

/// <summary>
/// Look-back lengths (seconds) of the momentum &amp; acceleration features: the short, medium and
/// long price returns, the short/long volume burst ratio, and the short window the price and volume
/// accelerations compare against the one before it. Validated by
/// <see cref="Options.PipelineOptions.HasConsistentMomentumWindows"/>.
/// </summary>
public readonly record struct MomentumWindows(int ShortSeconds, int MediumSeconds, int LongSeconds)
{
    public static readonly MomentumWindows Default = new(10, 30, 60);
}
