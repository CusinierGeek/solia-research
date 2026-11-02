using Microsoft.Extensions.Hosting;
using Solia.DataPipeline.Ingestion.Replay;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public class ReplayCommandLineTests
{
    private static readonly DateTimeOffset Now = new(2026, 5, 10, 12, 30, 0, TimeSpan.Zero);

    private static ReplayCommandLine? Parse(params string[] args) => ReplayCommandLine.TryParse(args, Now, out _);

    private static string Error(params string[] args)
    {
        Assert.Null(ReplayCommandLine.TryParse(args, Now, out string? error));
        return Assert.IsType<string>(error);
    }

    [Fact]
    public void Parses_window_and_flags()
    {
        ReplayCommandLine cmd = Parse("--from", "2026-05-01T03:00Z", "--to", "2026-05-01T06:00Z", "--mirror", "/data/m", "--rebuild")!;

        Assert.Equal(new DateTimeOffset(2026, 5, 1, 3, 0, 0, TimeSpan.Zero), cmd.Window.StartUtc);
        Assert.Equal(new DateTimeOffset(2026, 5, 1, 6, 0, 0, TimeSpan.Zero), cmd.Window.EndUtc);
        Assert.Equal("/data/m", cmd.MirrorDirectory);
        Assert.True(cmd.Rebuild);
        Assert.False(cmd.MirrorOnly);
    }

    [Fact]
    public void A_warm_up_reads_the_hours_before_the_window()
    {
        ReplayCommandLine cmd = Parse("--from", "2026-05-02", "--to", "2026-05-03", "--warmup-hours", "24")!;

        Assert.Equal(24, cmd.WarmupHours);
        Assert.Equal(new DateTimeOffset(2026, 5, 2, 0, 0, 0, TimeSpan.Zero), cmd.Window.StartUtc); // what is written
        Assert.Equal(new DateTimeOffset(2026, 5, 1, 0, 0, 0, TimeSpan.Zero), cmd.SourceWindow.StartUtc); // what is read
        Assert.Equal(48, cmd.SourceWindow.Hours().Count());
        Assert.Equal(0, Parse("--from", "2026-05-02", "--to", "2026-05-03")!.WarmupHours);
    }

    [Theory]
    [InlineData("-1")]
    [InlineData("1.5")]
    [InlineData("abc")]
    [InlineData("337")]
    public void An_invalid_warm_up_is_rejected(string hours)
    {
        Assert.Contains("--warmup-hours", Error("--from", "2026-05-02", "--to", "2026-05-03", "--warmup-hours", hours));
    }

    [Fact]
    public void A_warm_up_reaching_before_the_archive_is_rejected()
    {
        Assert.Contains("before the archive begins", Error("--from", "2026-04-18T12:00Z", "--to", "2026-04-19", "--warmup-hours", "24"));
        Assert.Contains("needs a value", Error("--from", "2026-05-02", "--to", "2026-05-03", "--warmup-hours"));
    }

    [Fact]
    public void Defaults_to_direct_streaming_without_rebuild()
    {
        ReplayCommandLine cmd = Parse("--from", "2026-05-01", "--to", "2026-05-02")!;

        Assert.Null(cmd.MirrorDirectory);
        Assert.False(cmd.MirrorOnly);
        Assert.False(cmd.Rebuild);
        Assert.Equal(24, cmd.Window.Hours().Count()); // bare dates = midnight UTC
    }

    [Fact]
    public void Values_without_offset_are_utc_and_offsets_are_converted()
    {
        ReplayCommandLine cmd = Parse("--from", "2026-05-01T03:00", "--to", "2026-05-01T06:00+02:00")!;

        Assert.Equal(TimeSpan.Zero, cmd.Window.StartUtc.Offset);
        Assert.Equal(new DateTimeOffset(2026, 5, 1, 3, 0, 0, TimeSpan.Zero), cmd.Window.StartUtc);
        Assert.Equal(new DateTimeOffset(2026, 5, 1, 4, 0, 0, TimeSpan.Zero), cmd.Window.EndUtc);
    }

    [Fact]
    public void Mirror_only_needs_a_mirror()
    {
        Assert.Contains("--mirror", Error("--from", "2026-05-01", "--to", "2026-05-02", "--mirror-only"));
        Assert.True(Parse("--from", "2026-05-01", "--to", "2026-05-02", "--mirror-only", "--mirror", "m")!.MirrorOnly);
    }

    [Fact]
    public void Rebuild_with_mirror_only_is_rejected()
    {
        Assert.Contains("--rebuild", Error("--from", "2026-05-01", "--to", "2026-05-02", "--mirror", "m", "--mirror-only", "--rebuild"));
    }

    [Theory]
    [InlineData("2026-05-02", "2026-05-01")] // inverted
    [InlineData("2026-05-01", "2026-05-01")] // empty
    public void Rejects_unordered_window(string from, string to)
    {
        Assert.Contains("StartUtc < EndUtc", Error("--from", from, "--to", to));
    }

    [Fact]
    public void Rejects_window_before_the_archive_start()
    {
        Assert.Contains("before the archive", Error("--from", "2026-04-17T23:00Z", "--to", "2026-04-18T02:00Z"));
    }

    [Fact]
    public void Rejects_window_ending_in_the_future()
    {
        Assert.Contains("hours that are over", Error("--from", "2026-05-10T10:00Z", "--to", "2026-05-10T13:00Z"));
    }

    [Fact]
    public void Rejects_window_reaching_into_the_current_unpublished_hour()
    {
        // Now = 12:30: the 12:00 archive isn't published yet, so it would be wrongly reported as missing.
        Assert.Contains("current hour", Error("--from", "2026-05-10T10:00Z", "--to", "2026-05-10T12:30Z"));
        Assert.Contains("current hour", Error("--from", "2026-05-10T10:00Z", "--to", "2026-05-10T12:01Z"));
        Assert.NotNull(Parse("--from", "2026-05-10T10:00Z", "--to", "2026-05-10T12:00Z")); // up to the hour's start
    }

    [Fact]
    public void Collects_configuration_overrides_for_the_host()
    {
        ReplayCommandLine cmd = Parse("--from", "2026-05-01", "--to", "2026-05-02",
            "--Pipeline:SnapshotCadenceSeconds=5", "--PumpApi:ReplayBaseUrl", "https://m.example", "--environment", "Development")!;

        Assert.Equal(
            new[] { "--Pipeline:SnapshotCadenceSeconds=5", "--PumpApi:ReplayBaseUrl", "https://m.example", "--environment", "Development" },
            cmd.ConfigArgs);
        Assert.Empty(Parse("--from", "2026-05-01", "--to", "2026-05-02")!.ConfigArgs);
    }

    [Fact]
    public void Configuration_override_without_value_is_rejected()
    {
        Assert.Contains("needs a value", Error("--from", "2026-05-01", "--to", "2026-05-02", "--Pipeline:SnapshotCadenceSeconds"));
        Assert.Contains("needs a value", Error("--from", "2026-05-01", "--to", "2026-05-02", "--environment", "--rebuild"));
    }

    [Fact]
    public void Host_builder_applies_overrides_and_reads_aspnetcore_prefixed_variables()
    {
        const string variable = "ASPNETCORE_SOLIA_REPLAY_TEST_KEY";
        Environment.SetEnvironmentVariable(variable, "from-aspnetcore");
        try
        {
            HostApplicationBuilder builder = ReplayCommand.CreateHostBuilder(
                ["--Pipeline:SnapshotCadenceSeconds=5", "--environment", "Staging"]);

            Assert.Equal("5", builder.Configuration["Pipeline:SnapshotCadenceSeconds"]);
            Assert.Equal("Staging", builder.Environment.EnvironmentName);
            Assert.Equal("from-aspnetcore", builder.Configuration["SOLIA_REPLAY_TEST_KEY"]);
        }
        finally
        {
            Environment.SetEnvironmentVariable(variable, null);
        }
    }

    [Theory]
    [InlineData("--to", "2026-05-02")]
    [InlineData("--from", "2026-05-01")]
    [InlineData]
    public void Requires_from_and_to(params string[] args)
    {
        Assert.Contains("required", Error(args));
    }

    [Theory]
    [InlineData("--from")]
    [InlineData("--from", "--to", "2026-05-02")]
    [InlineData("--from", "2026-05-01", "--to", "2026-05-02", "--mirror")]
    public void Rejects_a_flag_without_value(params string[] args)
    {
        Assert.Contains("needs a value", Error(args));
    }

    [Fact]
    public void Rejects_unparseable_dates_and_unknown_arguments()
    {
        Assert.Contains("ISO 8601", Error("--from", "yesterday", "--to", "2026-05-02"));
        Assert.Contains("unknown argument", Error("--from", "2026-05-01", "--to", "2026-05-02", "--fast"));
    }

    [Fact]
    public async Task Command_returns_usage_exit_code_on_bad_arguments_without_touching_anything()
    {
        Assert.Equal(ReplayCommand.ExitUsage, await ReplayCommand.RunAsync(["--from", "2026-05-02", "--to", "2026-05-01"]));
    }

    [Fact]
    public async Task Command_rejects_a_missing_mirror_directory()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"solia-no-such-mirror-{Guid.NewGuid():N}");

        int code = await ReplayCommand.RunAsync(["--from", "2026-05-01", "--to", "2026-05-02", "--mirror", missing]);

        Assert.Equal(ReplayCommand.ExitUsage, code);
        Assert.False(Directory.Exists(missing)); // not created behind the user's back
    }

    [Fact]
    public void Verb_is_detected_only_as_first_argument()
    {
        Assert.True(ReplayCommand.IsRequested(["replay", "--from", "x"]));
        Assert.True(ReplayCommand.IsRequested(["REPLAY"]));
        Assert.False(ReplayCommand.IsRequested([]));
        Assert.False(ReplayCommand.IsRequested(["--healthcheck", "replay"]));
    }
}
