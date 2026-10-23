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
    public async Task Empty_batch_is_a_no_op()
    {
        await Sink.WriteAsync(Batch(T0, []), default); // must not even need a connection
    }
}
