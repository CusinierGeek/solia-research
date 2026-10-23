using System.Text.Json;
using Npgsql;
using Solia.Bot.Domain;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

/// <summary>Integration tests against a real PostgreSQL (Testcontainers) — run in CI.</summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class PostgresTradeStoreTests(PostgresFixture db) : IClassFixture<PostgresFixture>
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private PostgresTradeStore Store => new(db.DataSource);

    /// <summary>A fresh mint per test, present in tokens_history (the FK target), so tests don't share rows.</summary>
    private async Task<string> SeedMintAsync()
    {
        string mint = $"mint-{Guid.NewGuid():N}";
        await db.ScalarAsync<int>("INSERT INTO tokens_history (mint) VALUES (@m) RETURNING 1", ("m", mint));
        return mint;
    }

    private static TradeOpening Opening(string mint, ModelPrediction? prediction = null)
    {
        ModelPrediction p = prediction ?? PolicyFixtures.Buyable() with { Mint = mint };
        return new TradeOpening(mint, T0, 0.0001m, 2m, ExitLevels.Compute(p, 0.0001m, PolicyFixtures.Options()), p);
    }

    private static ClosedTrade Settle(long id, TradeOpening opening, int afterSec, decimal exitPrice, ExitReason reason) =>
        ClosedTrade.Settle(new OpenPosition(id, opening, T0.AddSeconds(900)), T0.AddSeconds(afterSec), exitPrice, reason);

    [Fact]
    public async Task Open_then_close_persists_the_full_row()
    {
        string mint = await SeedMintAsync();
        ModelPrediction prediction = PolicyFixtures.Buyable() with
        {
            Mint = mint,
            Features = new Dictionary<string, double?> { ["rsi"] = 71.5, ["volatility"] = null, ["ema"] = double.NaN },
        };
        TradeOpening opening = Opening(mint, prediction);

        long id = await Store.OpenAsync(opening);

        Assert.Equal("open", await db.ScalarAsync<string>("SELECT status FROM trades WHERE id = @id", ("id", id)));
        Assert.Equal(1, await db.ScalarAsync<int>("SELECT count(*) FROM trades WHERE id = @id AND closed_at IS NULL AND exit_price IS NULL", ("id", id)));

        ClosedTrade closed = Settle(id, opening, 42, 0.000125m, ExitReason.TakeProfit);
        await Store.CloseAsync(closed);

        await using NpgsqlCommand cmd = db.DataSource.CreateCommand(
            "SELECT mint, prediction_id, opened_at, closed_at, entry_price, exit_price, size, duration_sec, pnl, pnl_pct, " +
            "strategy, confidence, exit_reason, status, features_snapshot::text FROM trades WHERE id = @id");
        cmd.Parameters.AddWithValue("id", id);
        await using NpgsqlDataReader row = await cmd.ExecuteReaderAsync();
        Assert.True(await row.ReadAsync());

        Assert.Equal(mint, row.GetString(0));
        Assert.True(row.IsDBNull(1));
        Assert.Equal(T0, row.GetFieldValue<DateTimeOffset>(2));
        Assert.Equal(T0.AddSeconds(42), row.GetFieldValue<DateTimeOffset>(3));
        Assert.Equal(0.0001m, row.GetDecimal(4));
        Assert.Equal(0.000125m, row.GetDecimal(5));
        Assert.Equal(2m, row.GetDecimal(6));
        Assert.Equal(42, row.GetInt32(7));
        Assert.Equal(0.5m, row.GetDecimal(8));
        Assert.Equal(0.25, row.GetDouble(9), 12);
        Assert.Equal("buy", row.GetString(10));
        Assert.Equal(80, row.GetDouble(11));
        Assert.Equal("tp", row.GetString(12));
        Assert.Equal("closed", row.GetString(13));

        using JsonDocument features = JsonDocument.Parse(row.GetString(14));
        Assert.Equal(71.5, features.RootElement.GetProperty("rsi").GetDouble());
        Assert.Equal(JsonValueKind.Null, features.RootElement.GetProperty("volatility").ValueKind);
        Assert.Equal(JsonValueKind.Null, features.RootElement.GetProperty("ema").ValueKind);
    }

    [Fact]
    public async Task An_existing_prediction_id_is_linked()
    {
        string mint = await SeedMintAsync();
        long predictionId = await db.ScalarAsync<long>(
            "INSERT INTO predictions (mint, horizon) VALUES (@m, '1m') RETURNING id", ("m", mint));
        TradeOpening opening = Opening(mint, PolicyFixtures.Buyable() with { Mint = mint, PredictionId = predictionId });

        long id = await Store.OpenAsync(opening);

        Assert.Equal(predictionId, await db.ScalarAsync<long>("SELECT prediction_id FROM trades WHERE id = @id", ("id", id)));
    }

    [Fact]
    public async Task An_unknown_mint_or_prediction_is_rejected_by_the_foreign_keys()
    {
        string mint = await SeedMintAsync();

        await Assert.ThrowsAsync<InvalidOperationException>(() => Store.OpenAsync(Opening("never-seen")));
        await Assert.ThrowsAsync<InvalidOperationException>(() =>
            Store.OpenAsync(Opening(mint, PolicyFixtures.Buyable() with { Mint = mint, PredictionId = long.MaxValue })));
        Assert.Equal(0, await db.ScalarAsync<int>("SELECT count(*) FROM trades WHERE mint = @m", ("m", mint)));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(-1)]
    [InlineData(100.5)]
    public async Task A_confidence_outside_the_check_constraint_is_rejected_before_writing(double confidence)
    {
        string mint = await SeedMintAsync();

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            Store.OpenAsync(Opening(mint, PolicyFixtures.Buyable() with { Mint = mint, Confidence = confidence })));
        Assert.Equal(0, await db.ScalarAsync<int>("SELECT count(*) FROM trades WHERE mint = @m", ("m", mint)));
    }

    [Fact]
    public async Task Closing_an_unknown_or_already_closed_trade_throws()
    {
        string mint = await SeedMintAsync();
        TradeOpening opening = Opening(mint);
        long id = await Store.OpenAsync(opening);
        await Store.CloseAsync(Settle(id, opening, 5, 0.0001m, ExitReason.Signal));

        await Assert.ThrowsAsync<InvalidOperationException>(() => Store.CloseAsync(Settle(id, opening, 9, 0.00005m, ExitReason.StopLoss)));
        await Assert.ThrowsAsync<InvalidOperationException>(() => Store.CloseAsync(Settle(long.MaxValue, opening, 5, 0.0001m, ExitReason.Timeout)));
        // The first close stands.
        Assert.Equal("signal", await db.ScalarAsync<string>("SELECT exit_reason FROM trades WHERE id = @id", ("id", id)));
    }

    [Fact]
    public async Task Closed_trades_are_counted_since_a_cutoff()
    {
        // Far-future close times keep this count clear of the other tests' trades, which close near T0.
        DateTimeOffset cutoff = new(2100, 1, 1, 0, 0, 0, TimeSpan.Zero);
        string mint = await SeedMintAsync();
        long before = await Store.CountClosedSinceAsync(null);

        foreach (int day in new[] { -1, 1, 2 })
        {
            TradeOpening opening = Opening(mint) with { OpenedAt = cutoff.AddDays(-2) };
            long id = await Store.OpenAsync(opening);
            await Store.CloseAsync(ClosedTrade.Settle(
                new OpenPosition(id, opening, cutoff.AddDays(3)), cutoff.AddDays(day), 0.0001m, ExitReason.Timeout));
        }
        await Store.OpenAsync(Opening(mint)); // still open: not counted

        Assert.Equal(2, await Store.CountClosedSinceAsync(cutoff));
        Assert.Equal(before + 3, await Store.CountClosedSinceAsync(null));
    }
}
