using System.ComponentModel.DataAnnotations;
using Solia.DataPipeline.Logging;
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

    [Fact]
    public void Every_log_severity_maps_to_a_value_the_logs_table_accepts()
    {
        // Mirrors CHECK (level IN ('debug','info','warn','error')) in 0001_init.sql.
        string[] allowed = ["debug", "info", "warn", "error"];

        Assert.All(Enum.GetValues<LogSeverity>(), s => Assert.Contains(s.ToDbValue(), allowed));
    }
}
