using System.ComponentModel.DataAnnotations;
using Solia.Shared.Logging;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests;

public class OptionsValidationTests
{
    private static bool IsValid(PipelineOptions o) =>
        Validator.TryValidateObject(o, new ValidationContext(o), null, validateAllProperties: true)
        && o.HasConsistentWindow();

    [Fact]
    public void Defaults_are_valid()
    {
        Assert.True(IsValid(new PipelineOptions()));
        Assert.True(new PumpApiOptions().HasValidUrls());
    }

    [Theory]
    [InlineData(false, false, true)]
    [InlineData(true, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, true, false)] // both would open the single live connection
    public void At_most_one_live_consumer_may_be_enabled(bool ingest, bool probe, bool valid)
    {
        Assert.Equal(valid, new PipelineOptions { Ingest = ingest, StreamProbe = probe }.HasExclusiveLiveConsumer());
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("2000000")]
    public void Out_of_range_large_buy_threshold_is_rejected(string sol)
    {
        var o = new PipelineOptions { LargeBuySol = decimal.Parse(sol, System.Globalization.CultureInfo.InvariantCulture) };

        Assert.False(IsValid(o));
    }

    [Theory]
    [InlineData(0, 300, 5000)]
    [InlineData(-1, 300, 5000)]
    [InlineData(10, 0, 5000)]
    [InlineData(10, 300, 0)]
    [InlineData(60, 30, 5000)] // window shorter than cadence
    public void Out_of_range_pipeline_values_are_rejected(int cadence, int window, int maxTokens)
    {
        var o = new PipelineOptions
        {
            SnapshotCadenceSeconds = cadence,
            WindowSeconds = window,
            MaxTrackedTokens = maxTokens,
        };

        Assert.False(IsValid(o));
    }

    [Theory]
    [InlineData("https://stream.pumpapi.io/", "https://replay.pumpapi.io")] // ws must be ws(s)
    [InlineData("wss://stream.pumpapi.io/", "wss://replay.pumpapi.io")]   // replay must be http(s)
    [InlineData("stream.pumpapi.io", "https://replay.pumpapi.io")]        // not absolute
    public void Invalid_pumpapi_urls_are_rejected(string ws, string replay)
    {
        Assert.False(new PumpApiOptions { WsUrl = ws, ReplayBaseUrl = replay }.HasValidUrls());
    }

    private static bool IsValid(PumpApiOptions o) =>
        Validator.TryValidateObject(o, new ValidationContext(o), null, validateAllProperties: true)
        && o.HasValidUrls() && o.HasConsistentBackoff();

    [Fact]
    public void Pumpapi_defaults_are_valid()
    {
        Assert.True(IsValid(new PumpApiOptions()));
    }

    [Theory]
    [InlineData(0, 30, 60, 1_048_576)]      // base delay below range
    [InlineData(1, 0, 60, 1_048_576)]       // max delay below range
    [InlineData(1, 3601, 60, 1_048_576)]    // max delay above range
    [InlineData(10, 5, 60, 1_048_576)]      // max < base
    [InlineData(1, 30, 1, 1_048_576)]       // idle timeout too short
    [InlineData(1, 30, 60, 100)]            // message cap too small
    [InlineData(1, 30, 60, 100_000_000)]    // message cap too large
    public void Out_of_range_pumpapi_values_are_rejected(int baseDelay, int maxDelay, int idle, int maxBytes)
    {
        var o = new PumpApiOptions
        {
            ReconnectBaseDelaySeconds = baseDelay,
            ReconnectMaxDelaySeconds = maxDelay,
            ReceiveIdleTimeoutSeconds = idle,
            MaxMessageBytes = maxBytes,
        };

        Assert.False(IsValid(o));
    }

    [Fact]
    public void Every_log_severity_maps_to_a_value_the_logs_table_accepts()
    {
        // Mirrors CHECK (level IN ('debug','info','warn','error')) in 0001_init.sql.
        string[] allowed = ["debug", "info", "warn", "error"];

        Assert.All(Enum.GetValues<LogSeverity>(), s => Assert.Contains(s.ToDbValue(), allowed));
    }
}
