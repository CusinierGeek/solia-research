using Solia.Dashboard.Api;
using Solia.Dashboard.Predictions;

namespace Solia.Dashboard.Tests.Predictions;

public class PredictionHistoryStateTests
{
    private const string Base = "http://localhost/predictions/history";

    private static readonly DateTimeOffset From = new(2026, 10, 1, 0, 0, 0, TimeSpan.Zero);
    private static readonly DateTimeOffset To = new(2026, 10, 2, 12, 30, 0, TimeSpan.Zero);

    [Fact]
    public void No_query_string_is_the_default_view()
    {
        PredictionHistoryState state = PredictionHistoryState.FromUri(Base);

        Assert.Equal(PredictionHistoryState.Default, state);
        Assert.All(state.ToQueryParameters().Values, Assert.Null);
    }

    [Fact]
    public void Every_parameter_is_read_from_the_url()
    {
        PredictionHistoryState state = PredictionHistoryState.FromUri(
            $"{Base}?mint=So1abc&horizon=5M&strategy=buy&from=2026-10-01T00:00:00Z&to=2026-10-02T12:30:00Z&page=3&pageSize=25");

        Assert.Equal(new PredictionHistoryFilter("So1abc", PredictionHorizon.Minutes5, TradeStrategy.Buy, From, To), state.Filter);
        Assert.Equal(3, state.Page);
        Assert.Equal(25, state.PageSize);
    }

    [Fact]
    public void Unparseable_values_are_ignored()
    {
        PredictionHistoryState state = PredictionHistoryState.FromUri(
            $"{Base}?mint=%20%20&horizon=2d&strategy=yolo&from=yesterday&page=-1&pageSize=abc");

        Assert.Equal(PredictionHistoryState.Default, state);
    }

    [Fact]
    public void Url_parameters_round_trip_and_omit_defaults()
    {
        var state = new PredictionHistoryState(
            new PredictionHistoryFilter("So1abc", PredictionHorizon.Hour1, TradeStrategy.Sell, From, To), Page: 4, PageSize: 100);

        IReadOnlyDictionary<string, object?> parameters = state.ToQueryParameters();

        Assert.Equal("1h", parameters["horizon"]);
        Assert.Equal("sell", parameters["strategy"]);
        Assert.Equal("2026-10-01T00:00:00Z", parameters["from"]);
        Assert.Equal("4", parameters["page"]);
        Assert.Equal("100", parameters["pageSize"]);
        string query = string.Join('&', parameters.Select(p => $"{p.Key}={Uri.EscapeDataString(p.Value!.ToString()!)}"));
        Assert.Equal(state, PredictionHistoryState.FromUri($"{Base}?{query}"));

        IReadOnlyDictionary<string, object?> defaults = (state with { Page = 1, PageSize = 50 }).ToQueryParameters();
        Assert.Null(defaults["page"]);
        Assert.Null(defaults["pageSize"]);
    }

    [Theory]
    [InlineData("201", 200)]
    [InlineData("100000", 200)]
    [InlineData("200", 200)]
    [InlineData("1", 1)]
    public void A_page_size_over_the_api_maximum_is_capped(string pageSize, int expected) =>
        Assert.Equal(expected, PredictionHistoryState.FromUri($"{Base}?pageSize={pageSize}").PageSize);

    [Fact]
    public void Url_parameters_use_the_api_query_names_and_spellings()
    {
        var state = new PredictionHistoryState(
            new PredictionHistoryFilter("So1abc", PredictionHorizon.Minutes15, TradeStrategy.Hold, From, To), Page: 2, PageSize: 25);

        IReadOnlyDictionary<string, object?> parameters = state.ToQueryParameters();

        Assert.Equal(
            state.ToQuery().ToParameters().Select(p => (p.Name, (object?)p.Value)),
            parameters.Select(p => (p.Key, p.Value)));
    }

    [Fact]
    public void The_api_query_carries_the_filters_and_the_page()
    {
        var state = new PredictionHistoryState(new PredictionHistoryFilter("m", PredictionHorizon.Minute1, null, From, null), 2);

        Assert.Equal(new PredictionHistoryQuery
        {
            Mint = "m",
            Horizon = PredictionHorizon.Minute1,
            From = From,
            Page = 2,
            PageSize = 50,
        }, state.ToQuery());
    }

    [Fact]
    public void The_mint_is_trimmed_and_blank_is_no_filter_however_it_is_set()
    {
        Assert.Equal("abc", new PredictionHistoryFilter("  abc ").Mint);
        Assert.Null(new PredictionHistoryFilter(" ").Mint);
        Assert.Equal("xyz", (PredictionHistoryFilter.None with { Mint = " xyz" }).Mint);
        Assert.Equal(PredictionHistoryFilter.None, PredictionHistoryFilter.None with { Mint = "" });
    }
}
