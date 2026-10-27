using Solia.Dashboard.Api;
using Solia.Dashboard.Trades;

namespace Solia.Dashboard.Tests.Trades;

public class TradeHistoryStateTests
{
    private const string Base = "http://localhost/trades";

    private static readonly DateTimeOffset From = new(2026, 10, 1, 0, 0, 0, TimeSpan.Zero);
    private static readonly DateTimeOffset To = new(2026, 10, 2, 12, 30, 0, TimeSpan.Zero);

    [Fact]
    public void No_query_string_is_the_default_live_view()
    {
        TradeHistoryState state = TradeHistoryState.FromUri(Base);

        Assert.Equal(TradeHistoryState.Default, state);
        Assert.True(state.IsLive);
        Assert.All(state.ToQueryParameters().Values, Assert.Null);
    }

    [Fact]
    public void Every_parameter_is_read_from_the_url()
    {
        TradeHistoryState state = TradeHistoryState.FromUri(
            $"{Base}?status=CLOSED&from=2026-10-01T00:00:00Z&to=2026-10-02T12:30:00Z&page=3&pageSize=25");

        Assert.Equal(new TradeHistoryFilter(TradeStatus.Closed, From, To), state.Filter);
        Assert.Equal(3, state.Page);
        Assert.Equal(25, state.PageSize);
    }

    [Fact]
    public void Unparseable_values_are_ignored()
    {
        TradeHistoryState state = TradeHistoryState.FromUri($"{Base}?status=pending&from=yesterday&to=1/2/2026&page=-1&pageSize=abc");

        Assert.Equal(TradeHistoryState.Default, state);
    }

    [Fact]
    public void Url_parameters_round_trip_and_omit_defaults()
    {
        var state = new TradeHistoryState(new TradeHistoryFilter(TradeStatus.Open, From, To), Page: 4, PageSize: 100);

        IReadOnlyDictionary<string, object?> parameters = state.ToQueryParameters();

        Assert.Equal("open", parameters["status"]);
        Assert.Equal("2026-10-01T00:00:00Z", parameters["from"]);
        Assert.Equal("4", parameters["page"]);
        Assert.Equal("100", parameters["pageSize"]);
        string query = string.Join('&', parameters.Select(p => $"{p.Key}={Uri.EscapeDataString((string)p.Value!)}"));
        Assert.Equal(state, TradeHistoryState.FromUri($"{Base}?{query}"));
    }

    [Fact]
    public void The_query_carries_the_filters_and_the_page()
    {
        var state = new TradeHistoryState(new TradeHistoryFilter(TradeStatus.Closed, From), Page: 2, PageSize: 25);

        Assert.Equal(
            new TradeHistoryQuery { Status = TradeStatus.Closed, From = From, Page = 2, PageSize = 25 },
            state.ToQuery());
    }

    [Theory]
    [InlineData("", true)]
    [InlineData("?pageSize=100", true)]
    [InlineData("?page=2", false)]
    [InlineData("?status=open", false)]
    [InlineData("?from=2026-10-01", false)]
    [InlineData("?to=2026-10-01", false)]
    public void Only_the_first_unfiltered_page_is_live(string query, bool live) =>
        Assert.Equal(live, TradeHistoryState.FromUri(Base + query).IsLive);
}
