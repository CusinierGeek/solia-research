using Solia.Dashboard.Api;

namespace Solia.Dashboard.Tests.Api;

public class QueryValuesTests
{
    [Fact]
    public void Every_enum_value_has_its_documented_spelling()
    {
        Assert.Equal(["updated_at", "market_cap", "liquidity", "holders", "trending_score", "created_at"],
            Enum.GetValues<TokenSort>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["pump", "rug", "neutral"], Enum.GetValues<TokenOutcome>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["30s", "1m", "5m", "15m", "1h"], Enum.GetValues<PredictionHorizon>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["ts", "pump_prob", "rug_prob", "confidence"], Enum.GetValues<PredictionSort>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["buy", "sell", "hold"], Enum.GetValues<TradeStrategy>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["open", "closed"], Enum.GetValues<TradeStatus>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["all", "24h", "7d", "30d"], Enum.GetValues<StatsWindow>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["debug", "info", "warn", "error"], Enum.GetValues<BotLogLevel>().Select(v => QueryValues.Format(v)));
        Assert.Equal(["data-pipeline", "ai-service", "backend", "bot"], Enum.GetValues<LogService>().Select(v => QueryValues.Format(v)));
    }

    [Fact]
    public void An_undefined_enum_value_is_rejected() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => QueryValues.Format((TradeStatus)42));

    [Fact]
    public void Timestamps_are_sent_as_utc_with_a_z_suffix()
    {
        var local = new DateTimeOffset(2026, 10, 26, 14, 30, 5, 123, TimeSpan.FromHours(-4));

        Assert.Equal("2026-10-26T18:30:05.123Z", QueryValues.Format(local));
        Assert.Equal("2026-10-26T18:30:05Z", QueryValues.Format(new DateTimeOffset(2026, 10, 26, 18, 30, 5, TimeSpan.Zero)));
    }

    [Fact]
    public void Null_and_blank_values_are_omitted_and_the_rest_escaped()
    {
        string url = QueryValues.Build("predictions/history",
            [("mint", "So1 a&b"), ("horizon", null), ("strategy", ""), ("from", "   "), ("page", "2")]);

        Assert.Equal("predictions/history?mint=So1%20a%26b&page=2", url);
    }

    [Fact]
    public void A_query_without_values_leaves_the_path_bare() =>
        Assert.Equal("tokens/live", QueryValues.Build("tokens/live", new LiveTokensQuery().ToParameters()));
}
