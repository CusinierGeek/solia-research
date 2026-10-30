using System.ComponentModel.DataAnnotations;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Npgsql;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Retention;
using Solia.Shared.Logging;

namespace Solia.DataPipeline.Tests.Retention;

public class RetentionConfigTests
{
    private static bool IsValid(RetentionOptions o) =>
        Validator.TryValidateObject(o, new ValidationContext(o), null, validateAllProperties: true);

    [Fact]
    public void Defaults_are_valid_and_off()
    {
        var o = new RetentionOptions();

        Assert.True(IsValid(o));
        Assert.False(o.Enabled);
        Assert.Equal(TimeSpan.FromDays(14), o.TtlOf(RetainedTable.Features));
        Assert.Equal(TimeSpan.FromDays(14), o.TtlOf(RetainedTable.Logs));
        Assert.Equal(TimeSpan.FromDays(30), o.TtlOf(RetainedTable.Predictions));
    }

    [Theory]
    [InlineData(nameof(RetentionOptions.FeaturesTtlDays), 0)]
    [InlineData(nameof(RetentionOptions.LogsTtlDays), -1)]
    [InlineData(nameof(RetentionOptions.PredictionsTtlDays), 0)]
    [InlineData(nameof(RetentionOptions.IntervalMinutes), 0)]
    [InlineData(nameof(RetentionOptions.BatchSize), 0)]
    public void Non_positive_values_are_rejected(string property, int value)
    {
        var o = new RetentionOptions();
        typeof(RetentionOptions).GetProperty(property)!.SetValue(o, value);

        Assert.False(IsValid(o));
    }

    [Fact]
    public void Invalid_retention_config_fails_the_host_at_startup()
    {
        using WebApplicationFactory<Program> factory = new WebApplicationFactory<Program>().WithWebHostBuilder(b =>
            b.ConfigureAppConfiguration((_, config) => config.AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["SOLIA_DB_CONNECTION"] = "Host=127.0.0.1;Port=1",
                ["Retention:FeaturesTtlDays"] = "0", // flat keys are mapped before the factory's config is added
            })));

        var ex = Assert.Throws<OptionsValidationException>(() => factory.CreateClient());
        Assert.Contains("FeaturesTtlDays", ex.Message);
    }

    [Fact]
    public void Flat_keys_map_onto_the_retention_section()
    {
        var config = new ConfigurationManager();
        config.AddInMemoryCollection(new Dictionary<string, string?>
        {
            ["RETENTION_ENABLED"] = "yes",
            ["RETENTION_INTERVAL_MINUTES"] = "15",
            ["RETENTION_FEATURES_TTL_DAYS"] = "7",
            ["RETENTION_LOGS_TTL_DAYS"] = "3",
            ["RETENTION_PREDICTIONS_TTL_DAYS"] = "60",
            ["RETENTION_BATCH_SIZE"] = "5000",
        });

        config.AddFlatEnvAliases();

        RetentionOptions o = config.GetSection(RetentionOptions.Section).Get<RetentionOptions>()!;
        Assert.True(o.Enabled);
        Assert.Equal(15, o.IntervalMinutes);
        Assert.Equal(7, o.FeaturesTtlDays);
        Assert.Equal(3, o.LogsTtlDays);
        Assert.Equal(60, o.PredictionsTtlDays);
        Assert.Equal(5000, o.BatchSize);
    }

    [Fact]
    public void Command_line_parses_dry_run_and_config_overrides()
    {
        var parsed = RetentionCommand.TryParse(["--dry-run", "--Retention:FeaturesTtlDays=7", "--Retention:BatchSize", "500"], out string? error);

        Assert.Null(error);
        Assert.True(parsed!.Value.DryRun);
        Assert.Equal(["--Retention:FeaturesTtlDays=7", "--Retention:BatchSize", "500"], parsed.Value.ConfigArgs);
        Assert.False(RetentionCommand.TryParse([], out _)!.Value.DryRun);
    }

    [Theory]
    [InlineData("--dryrun")]
    [InlineData("--Retention:BatchSize")]
    public void Command_line_rejects_bad_arguments(string arg)
    {
        Assert.Null(RetentionCommand.TryParse([arg], out string? error));
        Assert.NotNull(error);
    }

    [Fact]
    public async Task Invalid_config_on_the_command_is_a_usage_error()
    {
        // Options are validated before the database is ever touched.
        Assert.Equal(RetentionCommand.ExitUsage, await RetentionCommand.RunAsync(["--dry-run", "--Retention:LogsTtlDays=0"]));
        Assert.Equal(RetentionCommand.ExitUsage, await RetentionCommand.RunAsync(["--bogus"]));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task An_unreachable_database_fails_every_table_without_throwing(bool dryRun)
    {
        await using NpgsqlDataSource unreachable = NpgsqlDataSource.Create("Host=127.0.0.1;Port=1;Timeout=2");
        var job = new RetentionJob(new PostgresRetentionStore(unreachable), new PostgresLogSink(unreachable, "data-pipeline"),
            Microsoft.Extensions.Options.Options.Create(new RetentionOptions()), TimeProvider.System, NullLogger<RetentionJob>.Instance);

        RetentionReport report = await job.RunAsync(dryRun, default);

        Assert.Equal(RetentionCommand.ExitFailed, RetentionCommand.ExitCodeFor(report));
        Assert.All(report.Tables, t => Assert.NotNull(t.Error));
    }

    [Fact]
    public void Replay_accepts_allow_expired()
    {
        var now = new DateTimeOffset(2026, 10, 30, 12, 0, 0, TimeSpan.Zero);

        ReplayCommandLine? cmd = ReplayCommandLine.TryParse(["--from", "2026-10-01", "--to", "2026-10-02", "--allow-expired"], now, out _);

        Assert.True(cmd!.AllowExpired);
        Assert.False(ReplayCommandLine.TryParse(["--from", "2026-10-01", "--to", "2026-10-02"], now, out _)!.AllowExpired);
    }
}
