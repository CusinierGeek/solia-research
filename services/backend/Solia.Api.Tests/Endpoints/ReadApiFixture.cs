using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// A real Postgres (migrations applied) seeded once with a known fixture through raw SQL — the production
/// write path — and the real host pointed at it, with a fixed clock and a starting balance of 100. Every
/// expected count, page, win rate and equity value in <see cref="ReadEndpointsIntegrationTests"/> derives
/// from the rows below.
/// </summary>
public sealed class ReadApiFixture : PostgresFixture, IAsyncLifetime
{
    public static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    /// <summary>The host's "now" (for the win-rate windows).</summary>
    public static readonly DateTimeOffset Now = new(2026, 1, 20, 0, 0, 0, TimeSpan.Zero);

    public const decimal StartingBalance = 100m;

    /// <summary>Uses all 24 fractional digits of numeric(28,24).</summary>
    public const decimal TinyPrice = 0.000000123456789012345678m;

    private WebApplicationFactory<Program> _factory = null!;

    public HttpClient Client { get; private set; } = null!;

    async Task IAsyncLifetime.InitializeAsync()
    {
        await InitializeAsync();
        await SeedAsync();
        _factory = TestApiFactory.Create("Production",
            [("SOLIA_DB_CONNECTION", ConnectionString), ("Bot:StartingBalance", "100")],
            builder => builder.ConfigureTestServices(services => services.AddSingleton<TimeProvider>(new FixedClock(Now))));
        Client = _factory.CreateClient();
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        Client.Dispose();
        await _factory.DisposeAsync();
        await DisposeAsync();
    }

    private async Task SeedAsync()
    {
        // Registry: m1..m5 first seen one day apart, newest = m5.
        (string Mint, string? Outcome, int DaysBeforeT0)[] tokens =
            [("m1", "pump", 5), ("m2", "rug", 4), ("m3", "neutral", 3), ("m4", null, 2), ("m5", "pump", 1)];
        foreach ((string mint, string? outcome, int days) in tokens)
        {
            await ExecAsync(
                "INSERT INTO tokens_history (mint, symbol, name, first_seen_at, final_outcome) VALUES (@m, @s, @n, @seen, @o)",
                ("m", mint), ("s", mint.ToUpperInvariant()), ("n", $"Token {mint}"), ("seen", T0.AddDays(-days)),
                ("o", (object?)outcome ?? DBNull.Value));
        }

        // Live set: m1..m3 active (m2 has no market cap), m4 no longer tracked. m1's live symbol differs from the
        // registry's.
        await ExecAsync(
            "INSERT INTO tokens_live (mint, symbol, last_price, last_market_cap, holders, venue, is_migrated, is_active, updated_at) VALUES " +
            "('m1', 'M1-LIVE', @tiny, 1000, 10, 'pumpfun', FALSE, TRUE, @t0), " +
            "('m2', 'M2', 0.5, NULL, 20, 'pumpswap', TRUE, TRUE, @t0p1h), " +
            "('m3', 'M3', 0.25, 5000, NULL, 'pumpfun', FALSE, TRUE, @t0m1h), " +
            "('m4', 'M4', 1, 9999, 99, 'pumpfun', FALSE, FALSE, @t0p2h)",
            ("tiny", TinyPrice), ("t0", T0), ("t0p1h", T0.AddHours(1)), ("t0m1h", T0.AddHours(-1)), ("t0p2h", T0.AddHours(2)));

        // Predictions (minutes after T0): m1 has an older and a newer 1m row plus a model-less 5m one; m2 a 1m and
        // a 30s one at the same instant (inserted in that order); m3 one 1m; m4 (no longer tracked) and m5 (never live) are the newest but never "live".
        await ExecAsync("INSERT INTO models (version, path, is_champion) VALUES ('v1', 'models/v1.pt', TRUE)");
        const string predictionSql =
            "INSERT INTO predictions (mint, model_id, ts, horizon, pump_prob, rug_prob, strategy, position_size, take_profit, stop_loss, confidence) " +
            "VALUES (@m, (SELECT id FROM models WHERE version = @v), @ts, @h, @pump, @rug, @s, @size, @tp, @sl, @conf)";
        (string Mint, int Minutes, string Horizon, string? Model, double? Pump, double? Rug, string Strategy, double? Confidence)[] predictions =
        [
            ("m1", 0, "1m", "v1", 0.2, 0.3, "buy", 50),
            ("m1", 10, "1m", "v1", 0.9, 0.05, "buy", 80),
            ("m1", 5, "5m", null, 0.4, null, "hold", null),
            ("m2", 20, "1m", "v1", null, 0.7, "sell", 60),
            ("m2", 20, "30s", "v1", 0.6, 0.2, "buy", 70),
            ("m3", 15, "1m", "v1", 0.5, 0.1, "buy", 90),
            ("m4", 30, "1m", "v1", 0.99, 0.01, "buy", 99),
            ("m5", 40, "1m", "v1", 0.99, 0.01, "buy", 99),
        ];
        foreach ((string mint, int minutes, string horizon, string? model, double? pump, double? rug, string strategy, double? confidence) in predictions)
        {
            await ExecAsync(predictionSql, ("m", mint), ("v", (object?)model ?? DBNull.Value), ("ts", T0.AddMinutes(minutes)),
                ("h", horizon), ("pump", (object?)pump ?? DBNull.Value), ("rug", (object?)rug ?? DBNull.Value), ("s", strategy),
                ("size", 2.5m), ("tp", TinyPrice), ("sl", 0.0001m), ("conf", (object?)confidence ?? DBNull.Value));
        }

        // Trades: a win, a loss, a break-even (all long ago), a recent win closed exactly 24h before Now, and
        // one open position.
        const string closedSql =
            "INSERT INTO trades (mint, opened_at, closed_at, entry_price, exit_price, size, pnl, strategy, exit_reason, status) " +
            "VALUES (@m, @opened, @closed, @tiny, 0.5, 2.5, @pnl, 'buy', 'tp', 'closed')";
        await ExecAsync(closedSql, ("m", "m1"), ("opened", T0), ("closed", T0.AddHours(1)), ("tiny", TinyPrice), ("pnl", 1.5m));
        await ExecAsync(closedSql, ("m", "m2"), ("opened", T0.AddHours(1)), ("closed", T0.AddHours(2)), ("tiny", TinyPrice), ("pnl", -0.5m));
        await ExecAsync(closedSql, ("m", "m3"), ("opened", T0.AddHours(2)), ("closed", T0.AddHours(3)), ("tiny", TinyPrice), ("pnl", 0m));
        await ExecAsync(closedSql, ("m", "m1"), ("opened", Now.AddDays(-2)), ("closed", Now.AddDays(-1)), ("tiny", TinyPrice),
            ("pnl", 0.123456789012m));
        await ExecAsync("INSERT INTO trades (mint, opened_at, entry_price, size, strategy, status) VALUES ('m5', @o, 0.1, 1, 'buy', 'open')",
            ("o", Now.AddHours(-1)));
        // An inconsistent row (closed, no closed_at, oldest opened): listed in the history, left out of every stat.
        await ExecAsync("INSERT INTO trades (mint, opened_at, pnl, strategy, status) VALUES ('m4', @o, 1, 'buy', 'closed')",
            ("o", T0.AddHours(-1)));

        // Logs: the bot at every level, one minute apart, then a data-pipeline entry with a structured context.
        string[] levels = ["debug", "info", "warn", "error"];
        for (int i = 0; i < levels.Length; i++)
        {
            await ExecAsync("INSERT INTO logs (ts, service, level, message) VALUES (@ts, 'bot', @l, @msg)",
                ("ts", T0.AddMinutes(i)), ("l", levels[i]), ("msg", $"bot {levels[i]}"));
        }

        await ExecAsync("INSERT INTO logs (ts, service, level, message, context) VALUES (@ts, 'data-pipeline', 'info', 'replay done', @c::jsonb)",
            ("ts", T0.AddMinutes(10)), ("c", "{\"events\": 42, \"source\": \"pumpapi\"}"));
    }

    private async Task ExecAsync(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        await cmd.ExecuteNonQueryAsync();
    }

    private sealed class FixedClock(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;
    }
}
