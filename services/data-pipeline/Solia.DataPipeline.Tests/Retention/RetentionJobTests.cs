using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Time.Testing;
using Solia.DataPipeline.Retention;
using Solia.Shared.Logging;
using MsOptions = Microsoft.Extensions.Options.Options;

namespace Solia.DataPipeline.Tests.Retention;

public class RetentionJobTests
{
    private static readonly DateTimeOffset Now = new(2026, 10, 30, 12, 0, 0, TimeSpan.Zero);

    private readonly FakeRetentionStore _store = new();
    private readonly RecordingLogSink _sink = new();

    private RetentionJob Job(RetentionOptions? options = null) =>
        new(_store, _sink, MsOptions.Create(options ?? new RetentionOptions()), new FakeTimeProvider(Now),
            NullLogger<RetentionJob>.Instance);

    [Fact]
    public async Task Each_table_is_cut_off_at_its_own_ttl()
    {
        await Job().RunAsync(dryRun: false, default);

        Assert.Equal(Now.AddDays(-14), _store.Cutoffs[RetainedTable.Features]);
        Assert.Equal(Now.AddDays(-14), _store.Cutoffs[RetainedTable.Logs]);
        Assert.Equal(Now.AddDays(-30), _store.Cutoffs[RetainedTable.Predictions]);
    }

    [Fact]
    public async Task Configured_ttls_are_used()
    {
        await Job(new RetentionOptions { FeaturesTtlDays = 3, LogsTtlDays = 5, PredictionsTtlDays = 7 }).RunAsync(false, default);

        Assert.Equal(Now.AddDays(-3), _store.Cutoffs[RetainedTable.Features]);
        Assert.Equal(Now.AddDays(-5), _store.Cutoffs[RetainedTable.Logs]);
        Assert.Equal(Now.AddDays(-7), _store.Cutoffs[RetainedTable.Predictions]);
    }

    [Fact]
    public void Only_features_logs_and_predictions_are_ever_purged()
    {
        Assert.Equal(["predictions", "features", "logs"], RetainedTables.All.Select(t => t.SqlName()));
    }

    [Theory]
    [InlineData(250, 100, 3)] // 100 + 100 + 50: the short batch ends it
    [InlineData(200, 100, 3)] // an exact multiple needs one empty batch to know it's done
    [InlineData(0, 100, 1)]
    public async Task Deletes_in_bounded_batches_until_a_short_one(long expired, int batchSize, int expectedBatches)
    {
        _store.With(RetainedTable.Features, expired);

        RetentionReport report = await Job(new RetentionOptions { BatchSize = batchSize }).RunAsync(false, default);

        List<(RetainedTable Table, int BatchSize)> batches = _store.Deletes.Where(d => d.Table == RetainedTable.Features).ToList();
        Assert.Equal(expectedBatches, batches.Count);
        Assert.All(batches, b => Assert.Equal(batchSize, b.BatchSize));
        Assert.Equal(0, _store.Remaining(RetainedTable.Features));
        Assert.Equal(expired, report.Tables.Single(t => t.Table == RetainedTable.Features).Removed.Rows);
    }

    [Fact]
    public async Task Reports_rows_and_time_span_removed_per_table()
    {
        _store.With(RetainedTable.Features, 250).With(RetainedTable.Logs, 3);

        RetentionReport report = await Job(new RetentionOptions { BatchSize = 100 }).RunAsync(false, default);

        TablePurge features = report.Tables.Single(t => t.Table == RetainedTable.Features);
        Assert.Equal(250, features.Removed.Rows);
        Assert.Equal(new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero), features.Removed.Oldest);
        Assert.Equal(new DateTimeOffset(2026, 1, 1, 1, 39, 0, TimeSpan.Zero), features.Removed.Newest); // 100th row of a batch
        Assert.Equal(0, report.Tables.Single(t => t.Table == RetainedTable.Predictions).Removed.Rows);
        Assert.Null(report.Tables.Single(t => t.Table == RetainedTable.Predictions).Removed.Oldest);
        Assert.False(report.Failed);
    }

    [Fact]
    public async Task Only_tables_that_lost_rows_are_vacuumed()
    {
        _store.With(RetainedTable.Features, 5);

        await Job().RunAsync(false, default);

        Assert.Equal([RetainedTable.Features], _store.Vacuums);
    }

    [Fact]
    public async Task Dry_run_counts_and_changes_nothing()
    {
        _store.With(RetainedTable.Features, 42).With(RetainedTable.Predictions, 7);

        RetentionReport report = await Job().RunAsync(dryRun: true, default);

        Assert.True(report.DryRun);
        Assert.Equal(42, report.Tables.Single(t => t.Table == RetainedTable.Features).Removed.Rows);
        Assert.Equal(7, report.Tables.Single(t => t.Table == RetainedTable.Predictions).Removed.Rows);
        Assert.Empty(_store.Deletes);
        Assert.Empty(_store.Vacuums);
        Assert.Equal(42, _store.Remaining(RetainedTable.Features));
        Assert.Equal("retention dry run", Assert.Single(_sink.Entries).Message);
    }

    [Fact]
    public async Task Summary_is_written_to_the_logs_table()
    {
        _store.With(RetainedTable.Logs, 2);

        await Job().RunAsync(false, default);

        (LogSeverity level, string message, object? context) = Assert.Single(_sink.Entries);
        Assert.Equal(LogSeverity.Info, level);
        Assert.Equal("retention run", message);
        string json = PostgresLogSink.SerializeContext(context)!;
        Assert.Contains("\"table\":\"logs\",", json);
        Assert.Contains("\"rows\":2,", json);
        Assert.Contains("\"durationMs\":", json);
    }

    [Fact]
    public async Task A_failing_table_is_reported_and_the_others_still_purged()
    {
        _store.With(RetainedTable.Features, 10).With(RetainedTable.Logs, 10);
        _store.Failing.Add(RetainedTable.Features);

        RetentionReport report = await Job().RunAsync(false, default);

        Assert.True(report.Failed);
        Assert.Equal("Features failed", report.Tables.Single(t => t.Table == RetainedTable.Features).Error);
        Assert.Equal(0, _store.Remaining(RetainedTable.Logs));
        Assert.Equal(LogSeverity.Warn, Assert.Single(_sink.Entries).Level);
    }

    [Fact]
    public async Task An_unreachable_database_fails_the_run_without_throwing()
    {
        _store.Unreachable = true;
        _sink.Throws = true; // the summary can't be written either

        RetentionReport report = await Job().RunAsync(false, default);

        Assert.All(report.Tables, t => Assert.Equal("database unreachable", t.Error));
    }

    [Fact]
    public async Task Established_rows_past_their_own_ttl_go_and_the_band_is_scanned_up_to_the_features_cutoff()
    {
        _store.With(RetainedTable.Features, 5);

        RetentionReport report = await Job(new RetentionOptions { EstablishedTtlDays = 60 }).RunAsync(false, default);

        TablePurge features = report.Tables.Single(t => t.Table == RetainedTable.Features);
        Assert.Equal(Now.AddDays(-14), features.Cutoff);
        Assert.Equal(Now.AddDays(-60), features.EstablishedCutoff);
        Assert.Equal(5, features.Removed.Rows);
        var call = Assert.Single(_store.BandCalls);
        Assert.Equal(FeaturesCursor.At(Now.AddDays(-14) - RetentionJob.BandStartupLookback), call.After);
        Assert.Equal(Now.AddDays(-14), call.Cutoff);
        Assert.Equal(new EstablishedRule(TimeSpan.FromHours(6), TimeSpan.FromMinutes(60), TimeSpan.FromSeconds(60)), call.Rule);
    }

    [Fact]
    public async Task The_band_is_walked_in_batches_and_only_the_rows_the_rule_drops_go()
    {
        DateTimeOffset start = Now.AddDays(-15);
        _store.WithBand(start, 250, i => i % 2 == 0); // 125 doomed, 125 kept

        RetentionReport report = await Job(new RetentionOptions { BatchSize = 100 }).RunAsync(false, default);

        Assert.Equal(3, _store.BandCalls.Count);
        Assert.Equal(new FeaturesCursor(start.AddMinutes(99), 100), _store.BandCalls[1].After); // resumes after the last row scanned
        Assert.Equal(125, _store.Band.Count);
        Assert.All(_store.Band, r => Assert.False(r.Doomed));
        TablePurge features = report.Tables.Single(t => t.Table == RetainedTable.Features);
        Assert.Equal(125, features.Removed.Rows);
        Assert.Equal(start, features.Removed.Oldest);
        Assert.Contains(RetainedTable.Features, _store.Vacuums);
    }

    [Fact]
    public async Task The_next_run_resumes_the_band_where_the_last_one_ended()
    {
        RetentionJob job = Job();
        await job.RunAsync(false, default);
        await job.RunAsync(false, default); // same clock: nothing new to scan, starts at the previous cutoff

        Assert.Equal(FeaturesCursor.At(Now.AddDays(-14)), _store.BandCalls[1].After);
    }

    [Fact]
    public async Task A_dry_run_or_a_failed_run_does_not_move_the_band_start()
    {
        RetentionJob job = Job();
        await job.RunAsync(dryRun: true, default);
        _store.Failing.Add(RetainedTable.Features);
        RetentionReport failed = await job.RunAsync(false, default);
        _store.Failing.Clear();
        await job.RunAsync(false, default);

        Assert.True(failed.Failed);
        FeaturesCursor fresh = FeaturesCursor.At(Now.AddDays(-14) - RetentionJob.BandStartupLookback);
        Assert.Equal(fresh, _store.BandCalls[0].After);
        Assert.Equal(fresh, _store.BandCalls[^1].After);
        Assert.True(_store.BandCalls[0].DryRun);
    }

    [Fact]
    public async Task Dry_run_counts_the_band_without_deleting()
    {
        _store.WithBand(Now.AddDays(-15), 10, _ => true);

        RetentionReport report = await Job().RunAsync(dryRun: true, default);

        Assert.Equal(10, report.Tables.Single(t => t.Table == RetainedTable.Features).Removed.Rows);
        Assert.Equal(10, _store.Band.Count);
    }

    [Fact]
    public async Task The_band_never_starts_before_the_established_cutoff()
    {
        // TTLs 14 and 15 days: the 3-day startup lookback would reach past the established cutoff.
        await Job(new RetentionOptions { EstablishedTtlDays = 15 }).RunAsync(false, default);

        Assert.Equal(FeaturesCursor.At(Now.AddDays(-15)), Assert.Single(_store.BandCalls).After);
    }

    [Fact]
    public async Task Equal_ttls_mean_a_plain_purge_without_a_band()
    {
        _store.With(RetainedTable.Features, 3);

        RetentionReport report = await Job(new RetentionOptions { EstablishedTtlDays = 14 }).RunAsync(false, default);

        Assert.Empty(_store.BandCalls);
        Assert.Equal(Now.AddDays(-14), _store.Cutoffs[RetainedTable.Features]);
        Assert.Null(report.Tables.Single(t => t.Table == RetainedTable.Features).EstablishedCutoff);
    }

    [Fact]
    public async Task Cancellation_propagates()
    {
        _store.With(RetainedTable.Predictions, 10);

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Job().RunAsync(false, new CancellationToken(true)));
    }
}
