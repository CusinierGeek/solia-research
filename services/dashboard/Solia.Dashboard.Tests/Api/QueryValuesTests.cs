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
    public void A_session_selection_is_current_all_or_a_positive_id()
    {
        Assert.Equal(["current", "all", "7"], new[] { SessionSelection.Current, SessionSelection.All, SessionSelection.Of(7) }.Select(s => s.Wire));
        Assert.Equal(SessionSelection.Of(7), SessionSelection.Parse(" 7 "));
        Assert.Equal(7, SessionSelection.Parse("7")!.Id);
        Assert.Same(SessionSelection.All, SessionSelection.Parse("ALL"));
        Assert.Same(SessionSelection.Current, SessionSelection.Parse("current"));
        Assert.Null(SessionSelection.All.Id);
        Assert.All(new[] { null, "", "0", "-3", "1.5", "x" }, v => Assert.Null(SessionSelection.Parse(v)));
        Assert.Throws<ArgumentOutOfRangeException>(() => SessionSelection.Of(0));
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

    [Theory]
    [InlineData("5m", PredictionHorizon.Minutes5)]
    [InlineData(" 1H ", PredictionHorizon.Hour1)]
    [InlineData("2d", null)]
    [InlineData("", null)]
    [InlineData(null, null)]
    public void Wire_spellings_parse_back_case_insensitively(string? value, PredictionHorizon? expected) =>
        Assert.Equal(expected, QueryValues.Parse<PredictionHorizon>(value));

    [Fact]
    public void Timestamps_parse_back_as_utc()
    {
        var expected = new DateTimeOffset(2026, 10, 26, 18, 30, 5, TimeSpan.Zero);

        Assert.Equal(expected, QueryValues.ParseTimestamp("2026-10-26T18:30:05Z"));
        Assert.Equal(expected, QueryValues.ParseTimestamp("2026-10-26T14:30:05-04:00"));
        Assert.Equal(expected, QueryValues.ParseTimestamp("2026-10-26T18:30:05")); // no zone: UTC
        Assert.Equal(TimeSpan.Zero, QueryValues.ParseTimestamp("2026-10-26T14:30:05-04:00")!.Value.Offset);
        Assert.Equal(expected, QueryValues.ParseTimestamp("2026-10-26T18:30:05.000Z"));
        Assert.Equal(expected.AddSeconds(-5), QueryValues.ParseTimestamp("2026-10-26T18:30")); // datetime-local, no seconds
        Assert.Equal(new DateTimeOffset(2026, 10, 26, 0, 0, 0, TimeSpan.Zero), QueryValues.ParseTimestamp("2026-10-26"));
        Assert.Null(QueryValues.ParseTimestamp("yesterday"));
        Assert.Null(QueryValues.ParseTimestamp(null));
    }

    [Theory]
    [InlineData("1/2/2026")]
    [InlineData("Jan 2026")]
    [InlineData("26 Oct 2026 18:30")]
    [InlineData("2026-10-26 18:30:05")]
    [InlineData("20261026T183005Z")]
    [InlineData("2026-10")]
    [InlineData("")]
    public void Non_iso_timestamps_are_rejected(string value) =>
        Assert.Null(QueryValues.ParseTimestamp(value));
}
