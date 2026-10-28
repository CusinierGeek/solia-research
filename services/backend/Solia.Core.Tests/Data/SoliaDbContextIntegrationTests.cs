using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Metadata;
using Npgsql;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Core.Tests.Data;

/// <summary>
/// Integration tests against a real PostgreSQL 16 (Testcontainers) with the schema migrations applied — run
/// in CI. Rows are seeded with raw SQL (the only write path, as in production) and read back through EF:
/// types/precision survive, the mapping matches the live schema, and every write through EF is rejected.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class SoliaDbContextIntegrationTests(SoliaDbFixture db) : IClassFixture<SoliaDbFixture>
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 12, 30, 0, TimeSpan.Zero);

    /// <summary>A fresh mint, present in tokens_history (the FK target), so tests don't share rows.</summary>
    private async Task<string> SeedMintAsync()
    {
        string mint = $"mint-{Guid.NewGuid():N}";
        await db.ScalarAsync<int>(
            "INSERT INTO tokens_history (mint, symbol, name) VALUES (@m, 'SOL', 'Solia token') RETURNING 1", ("m", mint));
        return mint;
    }

    private Task<long> InsertAsync(string sql, params (string Name, object Value)[] parameters) =>
        db.ScalarAsync<long>(sql, parameters);

    [Fact]
    public async Task Full_chain_reads_back_with_correct_types_and_precision()
    {
        string mint = await SeedMintAsync();

        long modelId = await InsertAsync(
            "INSERT INTO models (version, path, is_champion, metrics, n_params) " +
            "VALUES (@v, '/models/v1.pt', TRUE, '{\"auc\":0.91}', 1234567) RETURNING id",
            ("v", $"v-{Guid.NewGuid():N}"));
        long featureId = await InsertAsync(
            "INSERT INTO features (mint, ts, venue, is_migrated, price, market_cap, liquidity, holders, buy_sell_ratio, " +
            "volatility, net_sol_flow, avg_buy_sol, price_acceleration, onchain_signals, raw) VALUES (@m, @ts, 'pumpfun', FALSE, " +
            "0.000000123456789, 123456789.123456789, 42.5, 311, 1.75, 0.0123456789, -1.234567890123, 0.5, -0.0625, " +
            "'{\"dev_sold\":false}', '{\"src\":\"test\"}') RETURNING id",
            ("m", mint),
            ("ts", T0));
        long predictionId = await InsertAsync(
            "INSERT INTO predictions (mint, model_id, ts, horizon, pump_prob, rug_prob, strategy, position_size, " +
            "take_profit, stop_loss, confidence, feature_id) VALUES (@m, @model, @ts, '5m', 0.82, 0.04, 'buy', 2.5, " +
            "0.000000185, 0.0000001, 87.5, @feature) RETURNING id",
            ("m", mint),
            ("model", modelId),
            ("ts", T0),
            ("feature", featureId));
        long tradeId = await InsertAsync(
            "INSERT INTO trades (mint, prediction_id, opened_at, closed_at, entry_price, exit_price, size, duration_sec, " +
            "pnl, pnl_pct, strategy, confidence, exit_reason, features_snapshot, status) " +
            "VALUES (@m, @prediction, @opened, @closed, 0.000000123456789, 0.000000234567891, 2.5, 120, " +
            "0.000123456789, 0.9, 'buy', 87.5, 'tp', '{\"rsi\":71.5}', 'closed') RETURNING id",
            ("m", mint),
            ("prediction", predictionId),
            ("opened", T0),
            ("closed", T0.AddSeconds(120)));

        await using SoliaDbContext ctx = db.NewContext();

        Feature feature = await ctx.Features.SingleAsync(f => f.Id == featureId);
        // NUMERIC must round-trip without drift — assert exact decimal equality.
        Assert.Equal(0.000000123456789m, feature.Price);
        Assert.Equal(123456789.123456789m, feature.MarketCap);
        Assert.Equal(42.5m, feature.Liquidity);
        Assert.Equal(311, feature.Holders);
        Assert.Equal(1.75, feature.BuySellRatio);
        Assert.Equal(-1.234567890123m, feature.NetSolFlow);
        Assert.Equal(0.5m, feature.AvgBuySol);
        Assert.Null(feature.LargeBuyCount);
        Assert.Equal(-0.0625, feature.PriceAcceleration);
        Assert.Null(feature.VolumeBurstRatio);
        Assert.Equal(0.0123456789, feature.Volatility!.Value, 12);
        Assert.Equal(T0, feature.Ts);
        Assert.Equal(TimeSpan.Zero, feature.Ts.Offset); // timestamptz reads back as UTC
        using JsonDocument onchain = JsonDocument.Parse(feature.OnchainSignals!);
        Assert.False(onchain.RootElement.GetProperty("dev_sold").GetBoolean());

        Prediction prediction = await ctx.Predictions
            .Include(p => p.Model)
            .Include(p => p.Feature)
            .SingleAsync(p => p.Id == predictionId);
        Assert.Equal("/models/v1.pt", prediction.Model!.Path);
        Assert.True(prediction.Model.IsChampion);
        Assert.Equal(1_234_567, prediction.Model.NParams);
        Assert.Equal(featureId, prediction.Feature!.Id);
        Assert.Equal("5m", prediction.Horizon);
        Assert.Equal(0.82, prediction.PumpProb);
        Assert.Equal(2.5m, prediction.PositionSize);
        Assert.Equal(0.000000185m, prediction.TakeProfit);

        Trade trade = await ctx.Trades
            .Include(t => t.Prediction)
            .Include(t => t.Token)
            .SingleAsync(t => t.Id == tradeId);
        Assert.Equal(predictionId, trade.Prediction!.Id); // navigations load through the declared FKs
        Assert.Equal("SOL", trade.Token!.Symbol);
        Assert.Equal(0.000000123456789m, trade.EntryPrice);
        Assert.Equal(0.000000234567891m, trade.ExitPrice);
        Assert.Equal(0.000123456789m, trade.Pnl); // 12 fractional digits: below a lamport
        Assert.Equal(120, trade.DurationSec);
        Assert.Equal("closed", trade.Status);
        Assert.Equal(T0, trade.OpenedAt);
        Assert.Equal(T0.AddSeconds(120), trade.ClosedAt);
        using JsonDocument snapshot = JsonDocument.Parse(trade.FeaturesSnapshot!);
        Assert.Equal(71.5, snapshot.RootElement.GetProperty("rsi").GetDouble());
    }

    [Fact]
    public async Task Sql_defaults_read_back_through_the_model()
    {
        string mint = await SeedMintAsync();
        long tradeId = await InsertAsync("INSERT INTO trades (mint) VALUES (@m) RETURNING id", ("m", mint));
        await db.ScalarAsync<int>("INSERT INTO tokens_live (mint) VALUES (@m) RETURNING 1", ("m", mint));

        await using SoliaDbContext ctx = db.NewContext();
        Trade trade = await ctx.Trades.SingleAsync(t => t.Id == tradeId);
        Assert.Equal("open", trade.Status);
        Assert.NotEqual(default, trade.OpenedAt);

        TokenLive live = await ctx.TokensLive.Include(t => t.Token).SingleAsync(t => t.Mint == mint);
        Assert.True(live.IsActive);
        Assert.False(live.IsMigrated);
        Assert.Equal("Solia token", live.Token!.Name);
    }

    [Fact]
    public async Task Every_write_path_through_ef_is_rejected_and_leaves_the_row_unchanged()
    {
        string mint = await SeedMintAsync();
        long tradeId = await InsertAsync("INSERT INTO trades (mint) VALUES (@m) RETURNING id", ("m", mint));

        await using (SoliaDbContext ctx = db.NewContext())
        {
            Trade trade = await ctx.Trades.SingleAsync(t => t.Id == tradeId);
            trade.Status = "closed";
            ctx.Update(trade);
            await Assert.ThrowsAsync<InvalidOperationException>(() => ctx.SaveChangesAsync());

            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                ctx.Trades.Where(t => t.Id == tradeId).ExecuteUpdateAsync(s => s.SetProperty(t => t.Status, "closed")));
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                ctx.Trades.Where(t => t.Id == tradeId).ExecuteDeleteAsync());
            await Assert.ThrowsAsync<InvalidOperationException>(() =>
                ctx.Database.ExecuteSqlAsync($"UPDATE trades SET status = 'closed' WHERE id = {tradeId}"));
            Assert.Throws<InvalidOperationException>(() =>
                ctx.Trades.Where(t => t.Id == tradeId).ExecuteDelete());
        }

        Assert.Equal("open", await db.ScalarAsync<string>("SELECT status FROM trades WHERE id = @id", ("id", tradeId)));
    }

    [Fact]
    public async Task A_non_utc_offset_query_parameter_matches_the_same_instant()
    {
        string mint = await SeedMintAsync();
        DateTimeOffset local = new(2026, 3, 1, 8, 30, 0, TimeSpan.FromHours(-4)); // 12:30Z
        await db.ScalarAsync<int>(
            "INSERT INTO features (mint, ts) VALUES (@m, '2026-03-01T12:30:00Z') RETURNING 1", ("m", mint));

        await using SoliaDbContext ctx = db.NewContext();
        // Npgsql alone rejects a non-zero offset for a timestamptz parameter.
        Feature feature = await ctx.Features.SingleAsync(f => f.Mint == mint && f.Ts == local);
        Assert.Equal(local, feature.Ts);
        Assert.Equal(TimeSpan.Zero, feature.Ts.Offset);
    }

    [Fact]
    public async Task Numeric_beyond_decimal_precision_is_rounded_on_write_and_reads_back()
    {
        string mint = await SeedMintAsync();

        // A 42-digit price, as Python/SQL could produce: unbounded NUMERIC made the EF read throw OverflowException.
        await db.ScalarAsync<int>(
            "INSERT INTO features (mint, ts, price, market_cap) " +
            "VALUES (@m, now(), 0.000000000123456789012345678901234567890123, 123.4567890123456789) RETURNING 1",
            ("m", mint));

        await using SoliaDbContext ctx = db.NewContext();
        Feature feature = await ctx.Features.SingleAsync(f => f.Mint == mint);
        Assert.Equal(0.000000000123456789012346m, feature.Price); // 24 fractional digits
        Assert.Equal(123.456789012346m, feature.MarketCap); // 12 fractional digits
    }

    [Fact]
    public async Task Every_mapped_column_matches_the_sql_schema_type()
    {
        Dictionary<(string Table, string Column), string> schema = [];
        await using (NpgsqlCommand cmd = db.DataSource.CreateCommand(
            "SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a " +
            "JOIN pg_class c ON c.oid = a.attrelid JOIN pg_namespace n ON n.oid = c.relnamespace " +
            "WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped"))
        await using (NpgsqlDataReader reader = await cmd.ExecuteReaderAsync())
        {
            while (await reader.ReadAsync())
            {
                schema[(reader.GetString(0), reader.GetString(1))] = reader.GetString(2);
            }
        }

        await using SoliaDbContext ctx = db.NewContext();
        foreach (IEntityType entity in ctx.Model.GetEntityTypes())
        {
            string table = entity.GetTableName()!;
            foreach (IProperty p in entity.GetProperties())
            {
                string column = p.GetColumnName();
                Assert.True(schema.TryGetValue((table, column), out string? sqlType), $"{table}.{column} is not in the schema");
                Assert.Equal($"{table}.{column}: {sqlType}", $"{table}.{column}: {p.GetColumnType()}");
            }
        }
    }

    [Fact]
    public async Task Every_mapped_foreign_key_exists_in_the_schema()
    {
        await using SoliaDbContext ctx = db.NewContext();
        foreach (IForeignKey fk in ctx.Model.GetEntityTypes().SelectMany(e => e.GetForeignKeys()))
        {
            string table = fk.DeclaringEntityType.GetTableName()!;
            long found = await db.ScalarAsync<long>(
                "SELECT count(*) FROM pg_constraint WHERE contype = 'f' AND conname = @name AND conrelid = @table::regclass",
                ("name", fk.GetConstraintName()!),
                ("table", table));
            Assert.True(found == 1, $"FK {fk.GetConstraintName()} on {table} is not in the schema");
        }
    }
}
