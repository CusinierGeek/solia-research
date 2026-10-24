using System.Net;
using System.Text.Json;
using static Solia.Api.Tests.Endpoints.ReadApiFixture;

namespace Solia.Api.Tests.Endpoints;

/// <summary>
/// The read endpoints end to end — real host, real Postgres (Testcontainers), the known fixture seeded by
/// <see cref="ReadApiFixture"/>: documented (snake_case) shapes, ordering, filters, pagination boundaries,
/// win-rate/equity math and decimal precision. Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public partial class ReadEndpointsIntegrationTests(ReadApiFixture api) : IClassFixture<ReadApiFixture>
{
    private async Task<JsonDocument> GetJsonAsync(string url)
    {
        HttpResponseMessage response = await api.Client.GetAsync(url);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        return JsonDocument.Parse(await response.Content.ReadAsStringAsync());
    }

    private static string[] Strings(JsonElement array, string property) =>
        array.EnumerateArray().Select(e => e.GetProperty(property).GetString()!).ToArray();

    private static void AssertPage(JsonElement root, int page, int pageSize, long total)
    {
        Assert.Equal(page, root.GetProperty("page").GetInt32());
        Assert.Equal(pageSize, root.GetProperty("page_size").GetInt32());
        Assert.Equal(total, root.GetProperty("total").GetInt64());
    }

    [Fact]
    public async Task Live_tokens_are_the_active_ones_newest_update_first_with_the_snapshot_shape()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/tokens/live");
        JsonElement items = doc.RootElement;

        Assert.Equal(["m2", "m1", "m3"], Strings(items, "mint"));
        JsonElement m1 = items[1];
        Assert.Equal(TinyPrice, m1.GetProperty("last_price").GetDecimal());
        Assert.Equal(1000m, m1.GetProperty("last_market_cap").GetDecimal());
        Assert.Equal(10, m1.GetProperty("holders").GetInt32());
        Assert.Equal("pumpfun", m1.GetProperty("venue").GetString());
        Assert.False(m1.GetProperty("is_migrated").GetBoolean());
        Assert.Equal(T0, m1.GetProperty("updated_at").GetDateTimeOffset());
        Assert.Equal(JsonValueKind.Null, items[0].GetProperty("last_market_cap").ValueKind);
    }

    [Fact]
    public async Task Live_tokens_sort_descending_with_missing_values_last_and_honour_the_limit()
    {
        using JsonDocument byCap = await GetJsonAsync("/api/v1/tokens/live?sort=market_cap");
        using JsonDocument limited = await GetJsonAsync("/api/v1/tokens/live?limit=1");

        Assert.Equal(["m3", "m1", "m2"], Strings(byCap.RootElement, "mint"));
        Assert.Equal(["m2"], Strings(limited.RootElement, "mint"));
    }

    [Fact]
    public async Task Token_history_pages_newest_first_with_the_total()
    {
        using JsonDocument first = await GetJsonAsync("/api/v1/tokens/history");
        using JsonDocument second = await GetJsonAsync("/api/v1/tokens/history?page=2&pageSize=2");
        using JsonDocument last = await GetJsonAsync("/api/v1/tokens/history?page=3&pageSize=2");
        using JsonDocument beyond = await GetJsonAsync("/api/v1/tokens/history?page=4&pageSize=2");

        Assert.Equal(["m5", "m4", "m3", "m2", "m1"], Strings(first.RootElement.GetProperty("items"), "mint"));
        AssertPage(first.RootElement, 1, 50, 5);
        Assert.Equal(["m3", "m2"], Strings(second.RootElement.GetProperty("items"), "mint"));
        AssertPage(second.RootElement, 2, 2, 5);
        Assert.Equal(["m1"], Strings(last.RootElement.GetProperty("items"), "mint"));
        Assert.Empty(beyond.RootElement.GetProperty("items").EnumerateArray());
        AssertPage(beyond.RootElement, 4, 2, 5);

        JsonElement m5 = first.RootElement.GetProperty("items")[0];
        Assert.Equal("pump", m5.GetProperty("final_outcome").GetString());
        Assert.Equal(T0.AddDays(-1), m5.GetProperty("first_seen_at").GetDateTimeOffset());
        Assert.Equal("M5", m5.GetProperty("symbol").GetString());
    }

    [Fact]
    public async Task Token_history_filters_by_outcome_and_first_seen_range()
    {
        using JsonDocument pumps = await GetJsonAsync("/api/v1/tokens/history?outcome=pump");
        using JsonDocument range = await GetJsonAsync(
            $"/api/v1/tokens/history?from={Iso(T0.AddDays(-3))}&to={Iso(T0.AddDays(-2))}");

        Assert.Equal(["m5", "m1"], Strings(pumps.RootElement.GetProperty("items"), "mint"));
        AssertPage(pumps.RootElement, 1, 50, 2);
        Assert.Equal(["m4", "m3"], Strings(range.RootElement.GetProperty("items"), "mint"));
    }

    [Fact]
    public async Task Trade_history_lists_most_recently_opened_first_with_precise_amounts()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/trades/history");
        JsonElement items = doc.RootElement.GetProperty("items");

        Assert.Equal(["m5", "m1", "m3", "m2", "m1", "m4"], Strings(items, "mint"));
        AssertPage(doc.RootElement, 1, 50, 6);
        JsonElement recentWin = items[1];
        Assert.Equal("closed", recentWin.GetProperty("status").GetString());
        Assert.Equal(TinyPrice, recentWin.GetProperty("entry_price").GetDecimal());
        Assert.Equal(0.123456789012m, recentWin.GetProperty("pnl").GetDecimal());
        Assert.Equal(Now.AddDays(-1), recentWin.GetProperty("closed_at").GetDateTimeOffset());
        Assert.Equal("tp", recentWin.GetProperty("exit_reason").GetString());
        Assert.Equal(JsonValueKind.Null, items[0].GetProperty("closed_at").ValueKind);
    }

    [Fact]
    public async Task Trade_history_filters_by_status_and_opened_range_and_pages()
    {
        using JsonDocument open = await GetJsonAsync("/api/v1/trades/history?status=open");
        using JsonDocument closedPage2 = await GetJsonAsync("/api/v1/trades/history?status=closed&page=2&pageSize=2");
        using JsonDocument range = await GetJsonAsync(
            $"/api/v1/trades/history?from={Iso(T0.AddMinutes(30))}&to={Iso(T0.AddMinutes(90))}");

        Assert.Equal(["m5"], Strings(open.RootElement.GetProperty("items"), "mint"));
        AssertPage(open.RootElement, 1, 50, 1);
        Assert.Equal(["m2", "m1"], Strings(closedPage2.RootElement.GetProperty("items"), "mint"));
        AssertPage(closedPage2.RootElement, 2, 2, 5);
        Assert.Equal(["m2"], Strings(range.RootElement.GetProperty("items"), "mint"));
    }

    [Fact]
    public async Task Win_rate_is_wins_over_every_closed_trade_like_the_backtest()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/stats/winrate");
        using JsonDocument blank = await GetJsonAsync("/api/v1/stats/winrate?window=");
        JsonElement root = doc.RootElement;

        // The open position and the closed row without closed_at are out; the break-even stays in the denominator.
        Assert.Equal(2, root.GetProperty("wins").GetInt32());
        Assert.Equal(1, root.GetProperty("losses").GetInt32());
        Assert.Equal(1, root.GetProperty("break_even").GetInt32());
        Assert.Equal(0.5, root.GetProperty("winrate").GetDouble(), precision: 12);
        Assert.Equal("all", root.GetProperty("window").GetString());
        Assert.Equal(root.GetRawText(), blank.RootElement.GetRawText());
    }

    [Fact]
    public async Task Win_rate_window_only_counts_trades_closed_strictly_within_it()
    {
        using JsonDocument week = await GetJsonAsync("/api/v1/stats/winrate?window=7d");
        using JsonDocument day = await GetJsonAsync("/api/v1/stats/winrate?window=24h");

        Assert.Equal(1, week.RootElement.GetProperty("wins").GetInt32());
        Assert.Equal(0, week.RootElement.GetProperty("losses").GetInt32());
        Assert.Equal(0, week.RootElement.GetProperty("break_even").GetInt32());
        Assert.Equal(1.0, week.RootElement.GetProperty("winrate").GetDouble());
        Assert.Equal("7d", week.RootElement.GetProperty("window").GetString());

        // The recent win closed exactly 24h before Now: on the bound, so outside it (the bot's IClosedTradeCounter "since").
        Assert.Equal(0, day.RootElement.GetProperty("wins").GetInt32());
        Assert.Equal(0.0, day.RootElement.GetProperty("winrate").GetDouble());
    }

    [Fact]
    public async Task Performance_is_the_equity_curve_from_the_starting_balance_in_close_order()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/stats/performance");
        JsonElement points = doc.RootElement.GetProperty("points");

        (DateTimeOffset Ts, decimal Equity, decimal Pnl)[] actual = points.EnumerateArray()
            .Select(p => (p.GetProperty("ts").GetDateTimeOffset(), p.GetProperty("equity").GetDecimal(),
                p.GetProperty("pnl").GetDecimal()))
            .ToArray();
        Assert.Equal(
            [
                (T0.AddHours(1), StartingBalance + 1.5m, 1.5m),
                (T0.AddHours(2), StartingBalance + 1.0m, -0.5m),
                (T0.AddHours(3), StartingBalance + 1.0m, 0m),
                (Now.AddDays(-1), StartingBalance + 1.123456789012m, 0.123456789012m),
            ],
            actual);
        Assert.Equal(1.123456789012m, doc.RootElement.GetProperty("total_pnl").GetDecimal());
    }

    [Fact]
    public async Task Performance_limit_keeps_the_latest_points_with_absolute_equity_and_the_full_total()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/stats/performance?limit=2");
        JsonElement points = doc.RootElement.GetProperty("points");

        (DateTimeOffset Ts, decimal Equity)[] actual = points.EnumerateArray()
            .Select(p => (p.GetProperty("ts").GetDateTimeOffset(), p.GetProperty("equity").GetDecimal()))
            .ToArray();
        Assert.Equal(
            [(T0.AddHours(3), StartingBalance + 1.0m), (Now.AddDays(-1), StartingBalance + 1.123456789012m)],
            actual);
        Assert.Equal(1.123456789012m, doc.RootElement.GetProperty("total_pnl").GetDecimal());
    }

    [Fact]
    public async Task Blank_filter_falls_back_to_the_default()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/trades/history?status=&page=1");

        AssertPage(doc.RootElement, 1, 50, 6);
    }

    [Fact]
    public async Task Bot_logs_are_newest_first_filtered_by_service_and_minimum_level()
    {
        using JsonDocument all = await GetJsonAsync("/api/v1/logs/bot?service=bot");
        using JsonDocument warnAndUp = await GetJsonAsync("/api/v1/logs/bot?service=bot&level=warn");
        using JsonDocument limited = await GetJsonAsync("/api/v1/logs/bot?service=bot&limit=2");

        Assert.Equal(["error", "warn", "info", "debug"], Strings(all.RootElement, "level"));
        Assert.Equal(["error", "warn"], Strings(warnAndUp.RootElement, "level"));
        Assert.Equal(["bot error", "bot warn"], Strings(limited.RootElement, "message"));
        JsonElement newest = all.RootElement[0];
        Assert.Equal("bot", newest.GetProperty("service").GetString());
        Assert.Equal(T0.AddMinutes(3), newest.GetProperty("ts").GetDateTimeOffset());
        Assert.Equal(JsonValueKind.Null, newest.GetProperty("context").ValueKind);
    }

    [Fact]
    public async Task Log_context_is_returned_as_structured_json()
    {
        using JsonDocument doc = await GetJsonAsync("/api/v1/logs/bot?service=data-pipeline");

        JsonElement context = Assert.Single(doc.RootElement.EnumerateArray()).GetProperty("context");
        Assert.Equal(42, context.GetProperty("events").GetInt32());
        Assert.Equal("pumpapi", context.GetProperty("source").GetString());
    }

    [Fact]
    public async Task Invalid_query_against_a_live_database_is_still_a_400_envelope()
    {
        HttpResponseMessage response = await api.Client.GetAsync("/api/v1/tokens/history?outcome=moon");

        Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("invalid_request", doc.RootElement.GetProperty("error").GetProperty("code").GetString());
    }

    private static string Iso(DateTimeOffset value) => Uri.EscapeDataString(value.ToString("O"));
}
