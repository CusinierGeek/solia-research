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

    private async Task<long> Feature(string mint, DateTimeOffset ts, int ageSec)
    {
        await Token(mint);
        return await Exec("INSERT INTO features (mint, ts, token_age_sec) VALUES (@m, @ts, @age)",
            ("m", mint), ("ts", ts), ("age", ageSec));
    }

    /// <summary>A token active from <paramref name="start"/>: one row every <paramref name="step"/> up to <paramref name="maxAgeSec"/>.</summary>
    private async Task Activity(string mint, DateTimeOffset start, int maxAgeSec, TimeSpan step)
    {
        await Token(mint);
        await db.ScalarAsync<int>(
        """
        INSERT INTO features (mint, ts, token_age_sec)
        SELECT @m, @t0 + make_interval(secs => a), a FROM generate_series(0, @max, @step) a
        UNION ALL SELECT @m, @t0 + make_interval(secs => @max), @max -- the exact max age, whatever the step
        ON CONFLICT DO NOTHING
        RETURNING 1
        """,
        ("m", mint), ("t0", start), ("max", maxAgeSec), ("step", (int)step.TotalSeconds));
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
        long feature = await Feature("FK1", now.AddDays(-16)); // past the 14-day features TTL, not established
        long prediction = await Prediction("FK1", now.AddDays(-16), feature); // inside the 30-day predictions TTL
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
        DateTimeOffset first = now.AddDays(-16);
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
    public async Task A_token_reaching_the_age_keeps_its_rows_past_the_ttl_and_one_a_second_short_loses_them()
    {
        var now = new DateTimeOffset(2036, 1, 1, 0, 0, 0, TimeSpan.Zero);
        DateTimeOffset start = now.AddDays(-16);
        await Activity("AGE6H", start, 21_600, TimeSpan.FromMinutes(10)); // reaches exactly 6 h
        await Activity("AGE6H-1S", start, 21_599, TimeSpan.FromMinutes(10)); // one second short
        // Its first hour is past the features cutoff, the 6 h it reaches are after it: still kept.
        await Activity("AGELATE", now.AddDays(-14).AddHours(-1), 21_600, TimeSpan.FromMinutes(10));
        long before = await Count("features", "mint IN ('AGE6H', 'AGE6H-1S', 'AGELATE')");

        RetentionReport report = await Job(now).RunAsync(false, default);

        Assert.False(report.Failed);
        Assert.Equal(37, await Count("features", "mint = 'AGE6H'"));
        Assert.Equal(0, await Count("features", "mint = 'AGE6H-1S'"));
        Assert.Equal(37, await Count("features", "mint = 'AGELATE'"));
        Assert.Equal(before - 37, await Count("features", "mint IN ('AGE6H', 'AGE6H-1S', 'AGELATE')"));
    }

    [Fact]
    public async Task Established_rows_go_after_the_established_ttl()
    {
        var now = new DateTimeOffset(2037, 1, 1, 0, 0, 0, TimeSpan.Zero);
        long expired = await Feature("TTL90", now.AddDays(-30).AddSeconds(-1), 30_000);
        long kept = await Feature("TTL90", now.AddDays(-29), 30_000);

        RetentionReport report = await Job(now, new RetentionOptions { EstablishedTtlDays = 30 }).RunAsync(false, default);

        Assert.Equal(0, await Count("features", "id = @id", ("id", expired)));
        Assert.Equal(1, await Count("features", "id = @id", ("id", kept)));
        TablePurge features = report.Tables.Single(t => t.Table == RetainedTable.Features);
        Assert.Equal(now.AddDays(-14), features.Cutoff);
        Assert.Equal(now.AddDays(-30), features.EstablishedCutoff);
    }

    [Theory]
    [InlineData(60, 360 + 301)] // first hour whole (360 rows), then the first row of each minute (ages 3600..21600)
    [InlineData(0, 2161)] // no thinning: every row
    public async Task Established_rows_past_the_ttl_are_thinned_after_the_full_resolution_age(int sampleSeconds, long expected)
    {
        var now = new DateTimeOffset(2038 + sampleSeconds, 1, 1, 0, 0, 0, TimeSpan.Zero);
        string mint = $"THIN{sampleSeconds}";
        await Activity(mint, now.AddDays(-16), 21_600, TimeSpan.FromSeconds(10));
        var options = new RetentionOptions { BatchSize = 100, EstablishedSampleSeconds = sampleSeconds };

        await Job(now, options).RunAsync(false, default);

        Assert.Equal(expected, await Count("features", "mint = @m", ("m", mint)));
        Assert.Equal(360, await Count("features", "mint = @m AND token_age_sec < 3600", ("m", mint)));
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
