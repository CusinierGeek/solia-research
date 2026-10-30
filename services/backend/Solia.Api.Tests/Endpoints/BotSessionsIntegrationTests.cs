using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Npgsql;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// Two bot sessions and a trade outside any (issue #169), seeded through raw SQL: session 1 (ended, starting balance 10)
/// and the current session 2 (starting balance 50), which resumed after trade 4 — so trades 2 and 4, still open, are
/// orphans of an earlier run while trade 7 is a live position. The configured starting balance is 100.
/// </summary>
public sealed class BotSessionsApiFixture : PostgresFixture, IAsyncLifetime
{
    public static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    private const string SeedSql = """
        INSERT INTO tokens_history (mint) VALUES ('m1');
        INSERT INTO models (id, version, path, is_champion) VALUES (1, 'v1', 'models/v1.pt', FALSE), (2, 'v2', 'models/v2.pt', TRUE);
        INSERT INTO bot_sessions (id, started_at, ended_at, config_hash, options, model_id, model_version, label, resumed_after_trade_id) VALUES
            (1, @t0, @t0 + interval '4 hours', 'h1', '{"startingBalance": 10}', 1, 'v1', 'naive', 0),
            (2, @t0 + interval '4 hours', NULL, 'h2', '{"startingBalance": 50.0, "minPumpProb": 0.25}', 2, 'v2', NULL, 4);
        INSERT INTO trades (id, mint, opened_at, closed_at, size, pnl, exit_reason, status, session_id) VALUES
            (1, 'm1', @t0,                         @t0 + interval '1 hour',  1, 5,    'tp',      'closed', 1),
            (2, 'm1', @t0 + interval '90 minutes', NULL,                     1, NULL, NULL,      'open',   1),
            (3, 'm1', @t0 + interval '1 hour',     @t0 + interval '2 hours', 1, -1,   'sl',      'closed', 1),
            (4, 'm1', @t0 + interval '4 hours',    NULL,                     2, NULL, NULL,      'open',   2),
            (5, 'm1', @t0 + interval '4 hours',    @t0 + interval '5 hours', 1, 2,    'tp',      'closed', 2),
            (6, 'm1', @t0 + interval '5 hours',    @t0 + interval '6 hours', 1, -0.5, 'timeout', 'closed', 2),
            (7, 'm1', @t0 + interval '7 hours',    NULL,                     3, NULL, NULL,      'open',   2),
            (8, 'm1', @t0 + interval '7 hours',    @t0 + interval '8 hours', 1, 100,  'signal',  'closed', NULL);
        """;

    private WebApplicationFactory<Program> _factory = null!;

    public HttpClient Client { get; private set; } = null!;

    async Task IAsyncLifetime.InitializeAsync()
    {
        await InitializeAsync();
        await using (NpgsqlCommand cmd = DataSource.CreateCommand(SeedSql))
        {
            cmd.Parameters.AddWithValue("t0", T0);
            await cmd.ExecuteNonQueryAsync();
        }

        _factory = TestApiFactory.Create("Production",
            [("SOLIA_DB_CONNECTION", ConnectionString), ("Bot:StartingBalance", "100")], _ => { });
        Client = _factory.CreateClient();
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        Client.Dispose();
        await _factory.DisposeAsync();
        await DisposeAsync();
    }
}

/// <summary>The session-scoped reads (issue #169): <c>?session=</c> / <c>?since=</c> on the stats and trades, and <c>/bot/sessions</c>.</summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class BotSessionsIntegrationTests(BotSessionsApiFixture api) : IClassFixture<BotSessionsApiFixture>
{
    private static readonly DateTimeOffset T0 = BotSessionsApiFixture.T0;

    private static readonly string Since = Uri.EscapeDataString(T0.AddHours(5).ToString("O"));

    private async Task<JsonDocument> GetJsonAsync(string url)
    {
        HttpResponseMessage response = await api.Client.GetAsync(url);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return JsonDocument.Parse(await response.Content.ReadAsStringAsync());
    }

    private static (int Wins, int Losses, long? Session, string Exits) WinRate(JsonElement root)
    {
        JsonElement exits = root.GetProperty("exits");
        JsonElement session = root.GetProperty("session");
        return (root.GetProperty("wins").GetInt32(), root.GetProperty("losses").GetInt32(),
            session.ValueKind == JsonValueKind.Null ? null : session.GetInt64(),
            $"{exits.GetProperty("tp")}/{exits.GetProperty("sl")}/{exits.GetProperty("signal")}/{exits.GetProperty("timeout")}");
    }

    private static decimal[] Equities(JsonElement root) =>
        root.GetProperty("points").EnumerateArray().Select(p => p.GetProperty("equity").GetDecimal()).ToArray();

    private static long[] Ids(JsonDocument doc) =>
        doc.RootElement.GetProperty("items").EnumerateArray().Select(i => i.GetProperty("id").GetInt64()).ToArray();

    [Fact]
    public async Task Win_rate_defaults_to_the_current_session()
    {
        using JsonDocument current = await GetJsonAsync("/api/v1/stats/winrate");
        using JsonDocument named = await GetJsonAsync("/api/v1/stats/winrate?session=CURRENT");

        Assert.Equal((1, 1, 2L, "1/0/0/1"), WinRate(current.RootElement));
        Assert.Equal(current.RootElement.GetRawText(), named.RootElement.GetRawText());
    }

    [Fact]
    public async Task Win_rate_reads_one_session_or_every_trade()
    {
        using JsonDocument first = await GetJsonAsync("/api/v1/stats/winrate?session=1");
        using JsonDocument all = await GetJsonAsync("/api/v1/stats/winrate?session=all");

        Assert.Equal((1, 1, 1L, "1/1/0/0"), WinRate(first.RootElement));
        Assert.Equal((3, 2, null, "2/1/1/1"), WinRate(all.RootElement));
    }

    [Fact]
    public async Task Win_rate_since_only_counts_trades_closed_strictly_after_it()
    {
        using JsonDocument doc = await GetJsonAsync($"/api/v1/stats/winrate?since={Since}");

        // Trade 5 closed exactly at since: out.
        Assert.Equal((0, 1, 2L, "0/0/0/1"), WinRate(doc.RootElement));
    }

    [Fact]
    public async Task The_equity_curve_restarts_from_each_sessions_own_starting_balance()
    {
        using JsonDocument current = await GetJsonAsync("/api/v1/stats/performance");
        using JsonDocument first = await GetJsonAsync("/api/v1/stats/performance?session=1");
        using JsonDocument all = await GetJsonAsync("/api/v1/stats/performance?session=all");

        Assert.Equal([52m, 51.5m], Equities(current.RootElement));
        Assert.Equal(1.5m, current.RootElement.GetProperty("total_pnl").GetDecimal());
        Assert.Equal(2, current.RootElement.GetProperty("session").GetInt64());
        Assert.Equal([15m, 14m], Equities(first.RootElement));
        // Every trade: the configured balance (100), in close order across sessions.
        Assert.Equal([105m, 104m, 106m, 105.5m, 205.5m], Equities(all.RootElement));
        Assert.Equal(JsonValueKind.Null, all.RootElement.GetProperty("session").ValueKind);
    }

    [Fact]
    public async Task The_equity_curve_since_starts_after_it()
    {
        using JsonDocument doc = await GetJsonAsync($"/api/v1/stats/performance?since={Since}");

        Assert.Equal([49.5m], Equities(doc.RootElement));
        Assert.Equal(-0.5m, doc.RootElement.GetProperty("total_pnl").GetDecimal());
    }

    [Fact]
    public async Task Trade_history_defaults_to_the_current_session()
    {
        using JsonDocument current = await GetJsonAsync("/api/v1/trades/history");
        using JsonDocument first = await GetJsonAsync("/api/v1/trades/history?session=1&status=open");
        using JsonDocument all = await GetJsonAsync("/api/v1/trades/history?session=all");

        Assert.Equal([7L, 6L, 5L, 4L], Ids(current));
        Assert.Equal([2L], Ids(first));
        Assert.Equal(8, all.RootElement.GetProperty("total").GetInt64());
    }

    [Fact]
    public async Task Sessions_are_listed_newest_first_with_orphans_out_of_the_open_positions()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/bot/sessions");
        JsonElement[] sessions = doc.RootElement.EnumerateArray().ToArray();

        Assert.Equal([2L, 1L], sessions.Select(s => s.GetProperty("id").GetInt64()));
        JsonElement current = sessions[0];
        Assert.True(current.GetProperty("is_current").GetBoolean());
        Assert.Equal("v2", current.GetProperty("model_version").GetString());
        Assert.Equal(0.25, current.GetProperty("options").GetProperty("minPumpProb").GetDouble());
        Assert.Equal(4, current.GetProperty("trades").GetInt64());
        Assert.Equal(2, current.GetProperty("closed").GetInt64());
        Assert.Equal(1, current.GetProperty("open").GetInt64());     // trade 7
        Assert.Equal(1, current.GetProperty("orphaned").GetInt64()); // trade 4, opened before the restart
        Assert.Equal(1.5m, current.GetProperty("total_pnl").GetDecimal());
        Assert.Equal(3m, current.GetProperty("committed").GetDecimal());

        JsonElement first = sessions[1];
        Assert.False(first.GetProperty("is_current").GetBoolean());
        Assert.Equal("naive", first.GetProperty("label").GetString());
        Assert.Equal(T0.AddHours(4), first.GetProperty("ended_at").GetDateTimeOffset());
        Assert.Equal(0, first.GetProperty("open").GetInt64());
        Assert.Equal(1, first.GetProperty("orphaned").GetInt64());
        Assert.Equal(4m, first.GetProperty("total_pnl").GetDecimal());
        Assert.Equal(0m, first.GetProperty("committed").GetDecimal());
    }

    [Theory]
    [InlineData("/api/v1/stats/winrate?session=99")]
    [InlineData("/api/v1/stats/performance?session=99")]
    [InlineData("/api/v1/trades/history?session=99")]
    public async Task An_unknown_session_is_not_found(string url)
    {
        HttpResponseMessage response = await api.Client.GetAsync(url);

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
        Assert.Contains("not_found", await response.Content.ReadAsStringAsync());
    }

    [Theory]
    [InlineData("abc")]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("1.5")]
    public async Task A_malformed_session_is_rejected(string session)
    {
        HttpResponseMessage response = await api.Client.GetAsync($"/api/v1/stats/winrate?session={session}");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        Assert.Contains("session", await response.Content.ReadAsStringAsync());
    }
}
