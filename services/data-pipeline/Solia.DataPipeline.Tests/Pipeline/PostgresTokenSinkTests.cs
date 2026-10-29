using Solia.DataPipeline.Features;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class PostgresTokenSinkTests(PostgresFixture db) : IClassFixture<PostgresFixture>
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    private PostgresTokenSink Sink => new(db.DataSource);

    private static FeatureSnapshot Snap(string mint, DateTimeOffset ts, decimal price, DateTimeOffset? firstSeen = null,
        DateTimeOffset? createdAt = null, bool migrated = false) =>
        new()
        {
            Mint = mint,
            Ts = ts,
            FirstSeenUtc = firstSeen ?? ts,
            CreatedAtUtc = createdAt,
            Price = price,
            Volume = 1m,
            Holders = 3,
            Venue = "pumpfun",
            IsMigrated = migrated,
            TrendingScore = null,
            Rsi = null,
        };

    private static FlushBatch Batch(DateTimeOffset ts, IReadOnlyList<FeatureSnapshot> snapshots, params string[] deactivated) =>
        new(ts, snapshots, deactivated);

    private Task<T> Live<T>(string column, string mint) =>
        db.ScalarAsync<T>($"SELECT {column} FROM tokens_live WHERE mint = @m", ("m", mint));

    [Fact]
    public async Task Replaying_an_older_window_does_not_roll_the_live_row_back()
    {
        await Sink.WriteAsync(Batch(T0.AddHours(3), [Snap("LIVE1", T0.AddHours(3), 5m, migrated: true)]), default);

        await Sink.WriteAsync(Batch(T0, [Snap("LIVE1", T0, 1m)]), default); // an older replay

        Assert.Equal(5m, await Live<decimal>("last_price", "LIVE1"));
        Assert.Equal(T0.AddHours(3).UtcDateTime, await Live<DateTime>("updated_at", "LIVE1"));
        Assert.True(await Live<bool>("is_migrated", "LIVE1"));
        Assert.Equal(2, await db.ScalarAsync<long>("SELECT count(*) FROM features WHERE mint = 'LIVE1'")); // history kept
    }

    [Fact]
    public async Task History_only_widens_first_and_last_seen_and_fills_created_at()
    {
        await Sink.WriteAsync(Batch(T0.AddHours(2), [Snap("HIST1", T0.AddHours(2), 1m, firstSeen: T0.AddHours(1))]), default);
        await Sink.WriteAsync(Batch(T0.AddMinutes(30),
            [Snap("HIST1", T0.AddMinutes(30), 1m, firstSeen: T0.AddMinutes(10), createdAt: T0)]), default);

        Assert.Equal(T0.AddMinutes(10).UtcDateTime,
            await db.ScalarAsync<DateTime>("SELECT first_seen_at FROM tokens_history WHERE mint = 'HIST1'"));
        Assert.Equal(T0.AddHours(2).UtcDateTime,
            await db.ScalarAsync<DateTime>("SELECT last_seen_at FROM tokens_history WHERE mint = 'HIST1'"));
        Assert.Equal(T0.UtcDateTime,
            await db.ScalarAsync<DateTime>("SELECT created_at FROM tokens_history WHERE mint = 'HIST1'"));
    }

    [Fact]
    public async Task Deactivated_mints_are_marked_inactive()
    {
        await Sink.WriteAsync(Batch(T0, [Snap("GONE1", T0, 1m), Snap("STAY1", T0, 1m)]), default);

        await Sink.WriteAsync(Batch(T0.AddMinutes(10), [Snap("STAY1", T0.AddMinutes(10), 1m)], "GONE1"), default);

        Assert.False(await Live<bool>("is_active", "GONE1"));
        Assert.True(await Live<bool>("is_active", "STAY1"));
    }

    [Fact]
    public async Task A_mint_deactivated_and_re_tracked_in_one_flush_stays_active()
    {
        await Sink.WriteAsync(Batch(T0, [Snap("BACK1", T0, 1m)]), default);

        await Sink.WriteAsync(Batch(T0.AddMinutes(5), [Snap("BACK1", T0.AddMinutes(5), 2m)], "BACK1"), default);

        Assert.True(await Live<bool>("is_active", "BACK1"));
    }

    [Fact]
    public async Task An_older_deactivation_does_not_override_newer_live_data()
    {
        await Sink.WriteAsync(Batch(T0.AddHours(1), [Snap("NEWER1", T0.AddHours(1), 1m)]), default);

        await Sink.WriteAsync(Batch(T0, [], "NEWER1"), default);

        Assert.True(await Live<bool>("is_active", "NEWER1"));
    }

    [Fact]
    public async Task A_full_working_set_is_written_in_one_flush()
    {
        DateTimeOffset ts = T0.AddDays(1);
        FeatureSnapshot[] snapshots = Enumerable.Range(0, 5000).Select(i => Snap($"BULK{i}", ts, i)).ToArray();

        await Sink.WriteAsync(Batch(ts, snapshots), default);

        Assert.Equal(5000, await db.ScalarAsync<long>("SELECT count(*) FROM features WHERE mint LIKE 'BULK%'"));
        Assert.Equal(5000, await db.ScalarAsync<long>("SELECT count(*) FROM tokens_live WHERE mint LIKE 'BULK%'"));
    }

    [Fact]
    public async Task Startup_reset_deactivates_everything_without_touching_updated_at_and_snapshots_reactivate()
    {
        DateTimeOffset ts = T0.AddDays(2);
        await Sink.WriteAsync(Batch(ts, [Snap("RESET1", ts, 1m), Snap("RESET2", ts, 1m)]), default);

        await Sink.DeactivateAllAsync(default);

        Assert.Equal(0, await db.ScalarAsync<long>("SELECT count(*) FROM tokens_live WHERE is_active"));
        Assert.Equal(ts.UtcDateTime, await Live<DateTime>("updated_at", "RESET1"));

        await Sink.WriteAsync(Batch(ts.AddSeconds(10), [Snap("RESET1", ts.AddSeconds(10), 2m)]), default);
        Assert.True(await Live<bool>("is_active", "RESET1"));
        Assert.False(await Live<bool>("is_active", "RESET2"));
    }

    [Fact]
    public async Task Buy_pressure_columns_are_persisted()
    {
        FeatureSnapshot snap = Snap("BUYP1", T0, 1m) with
        {
            NetSolFlow = -1.234567890123m,
            BuyVolumeRatio = 0.25,
            AvgBuySol = 0.5m,
            MaxBuySol = 2m,
            LargeBuyCount = 3,
        };
        await Sink.WriteAsync(Batch(T0, [snap, Snap("BUYP2", T0, 1m)]), default);

        Assert.Equal(-1.234567890123m, await Feature<decimal>("net_sol_flow", "BUYP1"));
        Assert.Equal(0.25, await Feature<double>("buy_volume_ratio", "BUYP1"));
        Assert.Equal(0.5m, await Feature<decimal>("avg_buy_sol", "BUYP1"));
        Assert.Equal(2m, await Feature<decimal>("max_buy_sol", "BUYP1"));
        Assert.Equal(3, await Feature<int>("large_buy_count", "BUYP1"));
        Assert.Equal(1L, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features WHERE mint = 'BUYP2' AND net_sol_flow IS NULL AND avg_buy_sol IS NULL " +
            "AND max_buy_sol IS NULL AND buy_volume_ratio IS NULL AND large_buy_count = 0"));
    }

    [Fact]
    public async Task Momentum_columns_are_persisted()
    {
        FeatureSnapshot snap = Snap("MOM1", T0, 1m) with
        {
            ReturnShort = 0.1,
            ReturnMedium = 0.2,
            ReturnLong = -0.3,
            PriceAcceleration = 0.05,
            EmaSlope = 0.01,
            VolumeAcceleration = 2.5,
            VolumeBurstRatio = 3,
            PriceHistorySec = 42.5,
        };
        await Sink.WriteAsync(Batch(T0, [snap, Snap("MOM2", T0, 1m)]), default);

        Assert.Equal(0.1, await Feature<double>("return_short", "MOM1"));
        Assert.Equal(0.2, await Feature<double>("return_medium", "MOM1"));
        Assert.Equal(-0.3, await Feature<double>("return_long", "MOM1"));
        Assert.Equal(0.05, await Feature<double>("price_acceleration", "MOM1"));
        Assert.Equal(0.01, await Feature<double>("ema_slope", "MOM1"));
        Assert.Equal(2.5, await Feature<double>("volume_acceleration", "MOM1"));
        Assert.Equal(3.0, await Feature<double>("volume_burst_ratio", "MOM1"));
        Assert.Equal(42.5, await Feature<double>("price_history_sec", "MOM1"));
        Assert.Equal(1L, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features WHERE mint = 'MOM2' AND return_short IS NULL AND return_medium IS NULL " +
            "AND return_long IS NULL AND price_acceleration IS NULL AND ema_slope IS NULL " +
            "AND volume_acceleration IS NULL AND volume_burst_ratio = 1 AND price_history_sec = 0"));
    }

    [Fact]
    public async Task Concentration_columns_are_persisted()
    {
        FeatureSnapshot snap = Snap("CONC1", T0, 1m) with
        {
            UniqueTraders = 7,
            TopTraderShare = 0.6,
            TraderHhi = 0.4,
            NewTraderRatio = 0.25,
        };
        await Sink.WriteAsync(Batch(T0, [snap, Snap("CONC2", T0, 1m)]), default);

        Assert.Equal(7, await Feature<int>("unique_traders", "CONC1"));
        Assert.Equal(0.6, await Feature<double>("top_trader_share", "CONC1"));
        Assert.Equal(0.4, await Feature<double>("trader_hhi", "CONC1"));
        Assert.Equal(0.25, await Feature<double>("new_trader_ratio", "CONC1"));
        Assert.Equal(1L, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features WHERE mint = 'CONC2' AND unique_traders = 0 " +
            "AND top_trader_share IS NULL AND trader_hhi IS NULL AND new_trader_ratio IS NULL"));
    }

    [Fact]
    public async Task Burned_liquidity_is_persisted_and_null_when_unknown()
    {
        FeatureSnapshot snap = Snap("BURN1", T0, 1m) with { BurnedLiquidity = 57.5 };
        await Sink.WriteAsync(Batch(T0, [snap, Snap("BURN2", T0, 1m)]), default);

        Assert.Equal(57.5, await Feature<double>("burned_liquidity", "BURN1"));
        Assert.Equal(1L, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features WHERE mint = 'BURN2' AND burned_liquidity IS NULL"));
    }

    [Fact]
    public async Task Migration_0009_backfills_burned_liquidity_from_onchain_signals()
    {
        // Rows written before 0008 only carry it in the JSONB: number -> column, null/absent/text -> NULL.
        // (The UPDATE leaves the insert trigger out of it, so the column is NULL as on a pre-0008 row.)
        await Sink.WriteAsync(Batch(T0, new[] { "BF1", "BF2", "BF3", "BF4" }.Select(m => Snap(m, T0, 1m)).ToList()), default);
        await using (var preMigration = db.DataSource.CreateCommand(
            "UPDATE features SET burned_liquidity = NULL, onchain_signals = CASE mint " +
            "WHEN 'BF1' THEN '{\"burnedLiquidityPercent\": 100}'::jsonb " +
            "WHEN 'BF2' THEN '{\"burnedLiquidityPercent\": null}'::jsonb " +
            "WHEN 'BF3' THEN '{}'::jsonb " +
            "ELSE '{\"burnedLiquidityPercent\": \"lots\"}'::jsonb END " +
            "WHERE mint IN ('BF1', 'BF2', 'BF3', 'BF4')"))
        {
            await preMigration.ExecuteNonQueryAsync();
        }

        string backfill = await File.ReadAllTextAsync(
            Path.Combine(PostgresFixture.FindMigrationsDir(), "0009_backfill_burned_liquidity.sql"));
        await using (var update = db.DataSource.CreateCommand(backfill)) // batched, commits per batch
        {
            await update.ExecuteNonQueryAsync();
        }

        Assert.Equal(100.0, await Feature<double>("burned_liquidity", "BF1"));
        Assert.Equal(3L, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features WHERE mint IN ('BF2', 'BF3', 'BF4') AND burned_liquidity IS NULL"));
    }

    private Task<T> Feature<T>(string column, string mint) =>
        db.ScalarAsync<T>($"SELECT {column} FROM features WHERE mint = @m", ("m", mint));

    [Fact]
    public async Task Empty_batch_is_a_no_op()
    {
        await Sink.WriteAsync(Batch(T0, []), default); // must not even need a connection
    }

    [Fact]
    public async Task Default_mode_keeps_existing_features_rows_and_counts_only_inserts()
    {
        var sink = new PostgresTokenSink(db.DataSource);
        await sink.WriteAsync(Batch(T0, [Snap("KEEP1", T0, 1m) with { Rsi = null }]), default);

        await sink.WriteAsync(Batch(T0, [Snap("KEEP1", T0, 2m) with { Rsi = 55 }]), default); // same (mint, ts)

        Assert.Equal(1m, await db.ScalarAsync<decimal>("SELECT price FROM features WHERE mint = 'KEEP1'"));
        Assert.Null(await db.ScalarAsync<double?>("SELECT rsi FROM features WHERE mint = 'KEEP1'"));
        Assert.Equal(1, sink.FeatureRowsWritten);
    }

    [Fact]
    public async Task Rebuild_mode_refreshes_existing_features_rows_in_place()
    {
        await Sink.WriteAsync(Batch(T0, [Snap("REB1", T0, 1m) with { Rsi = null }]), default); // "old" row, column unset
        long id = await db.ScalarAsync<long>("SELECT id FROM features WHERE mint = 'REB1'");
        var rebuild = new PostgresTokenSink(db.DataSource, refreshExistingFeatures: true);

        await rebuild.WriteAsync(
            Batch(T0, [Snap("REB1", T0, 2m, createdAt: T0.AddMinutes(-5)) with { Rsi = 55, TopTraderShare = 0.4 }]), default);

        Assert.Equal(1, await db.ScalarAsync<long>("SELECT count(*) FROM features WHERE mint = 'REB1'"));
        Assert.Equal(id, await db.ScalarAsync<long>("SELECT id FROM features WHERE mint = 'REB1'"));
        Assert.Equal(2m, await db.ScalarAsync<decimal>("SELECT price FROM features WHERE mint = 'REB1'"));
        Assert.Equal(55d, await db.ScalarAsync<double>("SELECT rsi FROM features WHERE mint = 'REB1'"));
        Assert.Equal(0.4, await db.ScalarAsync<double>("SELECT top_trader_share FROM features WHERE mint = 'REB1'"));
        Assert.Equal(1, rebuild.FeatureRowsWritten);
    }

    [Fact]
    public async Task Rebuild_mode_keeps_rows_of_tokens_whose_creation_the_run_did_not_see()
    {
        // PRE1 was born before the rebuild window: its recomputed snapshot comes from a truncated history.
        await Sink.WriteAsync(Batch(T0, [Snap("PRE1", T0, 1m) with { Holders = 40 }, Snap("NEW1", T0, 1m)]), default);
        var rebuild = new PostgresTokenSink(db.DataSource, refreshExistingFeatures: true);

        await rebuild.WriteAsync(Batch(T0, [
            Snap("PRE1", T0, 2m) with { Holders = 2 },
            Snap("NEW1", T0, 2m, createdAt: T0.AddMinutes(-1)),
            Snap("PRE1B", T0, 3m), // no stored row yet: still inserted
        ]), default);

        Assert.Equal(40, await db.ScalarAsync<int>("SELECT holders FROM features WHERE mint = 'PRE1'"));
        Assert.Equal(1m, await db.ScalarAsync<decimal>("SELECT price FROM features WHERE mint = 'PRE1'"));
        Assert.Equal(2m, await db.ScalarAsync<decimal>("SELECT price FROM features WHERE mint = 'NEW1'"));
        Assert.Equal(3m, await db.ScalarAsync<decimal>("SELECT price FROM features WHERE mint = 'PRE1B'"));
        Assert.Equal(2, rebuild.FeatureRowsWritten); // NEW1 refreshed + PRE1B inserted
    }

    [Fact]
    public void Only_snapshots_with_an_observed_creation_can_be_refreshed()
    {
        Assert.True(PostgresTokenSink.CanRefresh(Snap("C1", T0, 1m, createdAt: T0)));
        Assert.False(PostgresTokenSink.CanRefresh(Snap("C2", T0, 1m)));
    }

    [Fact]
    public async Task Release_batch_after_a_snapshot_at_the_same_instant_deactivates_it()
    {
        await Sink.WriteAsync(Batch(T0.AddDays(1), [Snap("END1", T0.AddDays(1), 1m)]), default);

        await Sink.WriteAsync(Batch(T0.AddDays(1), [], "END1"), default); // replay end: same ts as the last flush

        Assert.False(await Live<bool>("is_active", "END1"));
    }

    [Fact]
    public void Refresh_statement_updates_every_column_but_the_key()
    {
        string sql = PostgresTokenSink.FeaturesRefreshSql;

        Assert.Contains("behavioral_signals = EXCLUDED.behavioral_signals", sql);
        Assert.Contains("burned_liquidity = EXCLUDED.burned_liquidity", sql);
        Assert.DoesNotContain("mint = EXCLUDED", sql);
        Assert.DoesNotContain(" ts = EXCLUDED", sql);
    }
}
