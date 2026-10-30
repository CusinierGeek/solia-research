using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Time.Testing;
using Npgsql;
using Solia.DataPipeline.Retention;
using Solia.Shared.Logging;
using MsOptions = Microsoft.Extensions.Options.Options;

namespace Solia.DataPipeline.Tests.Retention;

/// <summary>The purge against the real schema. Each test uses its own mints and clock, far apart, so they don't interfere.</summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class PostgresRetentionTests(PostgresFixture db) : IClassFixture<PostgresFixture>
{
    private RetentionJob Job(DateTimeOffset now, RetentionOptions? options = null) =>
        new(new PostgresRetentionStore(db.DataSource), new PostgresLogSink(db.DataSource, "data-pipeline"),
            MsOptions.Create(options ?? new RetentionOptions()), new FakeTimeProvider(now), NullLogger<RetentionJob>.Instance);

    private async Task<long> Exec(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = db.DataSource.CreateCommand(sql + " RETURNING id");
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        return (long)(await cmd.ExecuteScalarAsync())!;
    }

    private Task Token(string mint) => db.ScalarAsync<int>(
        "INSERT INTO tokens_history (mint, first_seen_at) VALUES (@m, '2000-01-01') ON CONFLICT DO NOTHING RETURNING 1", ("m", mint));

    private async Task<long> Feature(string mint, DateTimeOffset ts)
    {
        await Token(mint);
        return await Exec("INSERT INTO features (mint, ts) VALUES (@m, @ts)", ("m", mint), ("ts", ts));
    }

    private async Task<long> Prediction(string mint, DateTimeOffset ts, long? featureId = null)
    {
        await Token(mint);
        return await Exec("INSERT INTO predictions (mint, ts, horizon, feature_id) VALUES (@m, @ts, '15m', @f)",
            ("m", mint), ("ts", ts), ("f", (object?)featureId ?? DBNull.Value));
    }

    private Task<long> Log(DateTimeOffset ts, string message) =>
        Exec("INSERT INTO logs (ts, service, level, message) VALUES (@ts, 'test', 'info', @msg)", ("ts", ts), ("msg", message));

    private Task<long> Count(string table, string where, params (string Name, object Value)[] parameters) =>
        db.ScalarAsync<long>($"SELECT count(*) FROM {table} WHERE {where}", parameters);

    [Fact]
    public async Task Rows_just_outside_each_ttl_go_and_rows_just_inside_stay()
    {
        var now = new DateTimeOffset(2030, 1, 1, 0, 0, 0, TimeSpan.Zero);
        DateTimeOffset featureCutoff = now.AddDays(-14), logCutoff = now.AddDays(-14), predictionCutoff = now.AddDays(-30);
        long fOld = await Feature("CUT1", featureCutoff.AddSeconds(-1));
        long fEdge = await Feature("CUT1", featureCutoff); // exactly the cutoff: not older than the TTL
        long fNew = await Feature("CUT1", featureCutoff.AddSeconds(1));
        long pOld = await Prediction("CUT1", predictionCutoff.AddSeconds(-1));
        long pNew = await Prediction("CUT1", predictionCutoff.AddSeconds(1));
        long lOld = await Log(logCutoff.AddSeconds(-1), "cut-old");
        long lNew = await Log(logCutoff.AddSeconds(1), "cut-new");

        RetentionReport report = await Job(now).RunAsync(dryRun: false, default);

        Assert.False(report.Failed);
        Assert.Equal(0, await Count("features", "id = @id", ("id", fOld)));
        Assert.Equal(2, await Count("features", "id = ANY(@ids)", ("ids", new[] { fEdge, fNew })));
        Assert.Equal(0, await Count("predictions", "id = @id", ("id", pOld)));
        Assert.Equal(1, await Count("predictions", "id = @id", ("id", pNew)));
        Assert.Equal(0, await Count("logs", "id = @id", ("id", lOld)));
        Assert.Equal(1, await Count("logs", "id = @id", ("id", lNew)));
        TablePurge features = report.Tables.Single(t => t.Table == RetainedTable.Features);
        Assert.Equal(featureCutoff, features.Cutoff);
        Assert.True(features.Removed.Rows >= 1);
        Assert.True(features.Removed.Newest < featureCutoff);
    }

    [Fact]
    public async Task Protected_tables_are_never_touched()
    {
        var now = new DateTimeOffset(2031, 1, 1, 0, 0, 0, TimeSpan.Zero);
        DateTimeOffset ancient = now.AddYears(-5);
        await Token("KEEP1");
        long model = await Exec("INSERT INTO models (version, created_at, path) VALUES ('keep-v1', @ts, '/m')", ("ts", ancient));
        long training = await Exec(
            "INSERT INTO training_events (started_at, finished_at, decision, candidate_model_id) VALUES (@ts, @ts, 'rejected', @m)",
            ("ts", ancient), ("m", model));
        long trade = await Exec("INSERT INTO trades (mint, opened_at, closed_at, status) VALUES ('KEEP1', @ts, @ts, 'closed')",
            ("ts", ancient));

        await Job(now).RunAsync(false, default);

        Assert.Equal(1, await Count("models", "id = @id", ("id", model)));
        Assert.Equal(1, await Count("training_events", "id = @id", ("id", training)));
        Assert.Equal(1, await Count("trades", "id = @id", ("id", trade)));
        Assert.Equal(1, await Count("tokens_history", "mint = 'KEEP1'"));
    }

    [Fact]
    public async Task Rows_referencing_purged_rows_survive_with_null_references()
    {
        var now = new DateTimeOffset(2032, 1, 1, 0, 0, 0, TimeSpan.Zero);
        long feature = await Feature("FK1", now.AddDays(-20)); // past the 14-day features TTL
        long prediction = await Prediction("FK1", now.AddDays(-20), feature); // inside the 30-day predictions TTL
        long oldPrediction = await Prediction("FK1", now.AddDays(-40));
        long trade = await Exec("INSERT INTO trades (mint, prediction_id, opened_at) VALUES ('FK1', @p, @ts)",
            ("p", oldPrediction), ("ts", now.AddDays(-40)));

        await Job(now).RunAsync(false, default);

        Assert.Equal(0, await Count("features", "id = @id", ("id", feature)));
        Assert.Equal(1, await Count("predictions", "id = @id AND feature_id IS NULL", ("id", prediction)));
        Assert.Equal(0, await Count("predictions", "id = @id", ("id", oldPrediction)));
        Assert.Equal(1, await Count("trades", "id = @id AND prediction_id IS NULL", ("id", trade)));
    }

    [Fact]
    public async Task Purges_in_batches_oldest_first_and_reports_the_span()
    {
        var now = new DateTimeOffset(2033, 1, 1, 0, 0, 0, TimeSpan.Zero);
        DateTimeOffset first = now.AddDays(-20);
        await Token("BATCH1");
        await db.ScalarAsync<int>(
            "INSERT INTO features (mint, ts) SELECT 'BATCH1', @t0 + g * interval '1 minute' FROM generate_series(0, 249) g RETURNING 1",
            ("t0", first));

        RetentionReport report = await Job(now, new RetentionOptions { BatchSize = 100 }).RunAsync(false, default);

        TablePurge features = report.Tables.Single(t => t.Table == RetainedTable.Features);
        Assert.Equal(0, await Count("features", "mint = 'BATCH1'"));
        Assert.True(features.Removed.Rows >= 250);
        Assert.True(features.Removed.Oldest <= first);
        Assert.True(features.Removed.Newest >= first.AddMinutes(249));
    }

    [Fact]
    public async Task Dry_run_reports_and_deletes_nothing()
    {
        var now = new DateTimeOffset(2034, 1, 1, 0, 0, 0, TimeSpan.Zero);
        long feature = await Feature("DRY1", now.AddDays(-15));

        RetentionReport report = await Job(now).RunAsync(dryRun: true, default);

        Assert.True(report.Tables.Single(t => t.Table == RetainedTable.Features).Removed.Rows >= 1);
        Assert.Equal(1, await Count("features", "id = @id", ("id", feature)));
    }

    [Fact]
    public async Task Each_run_leaves_a_summary_row_in_logs()
    {
        var now = new DateTimeOffset(2035, 1, 1, 0, 0, 0, TimeSpan.Zero);
        await Feature("SUM1", now.AddDays(-15));

        await Job(now).RunAsync(false, default);

        Assert.True(await Count("logs",
            "service = 'data-pipeline' AND message = 'retention run' AND context -> 'tables' @> '[{\"table\": \"features\"}]'") >= 1);
    }
}
