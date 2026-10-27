using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.Primitives;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Tables;

namespace Solia.Dashboard.Trades;

/// <summary>The trade history filters (null = not filtered); <see cref="From"/>/<see cref="To"/> bound <c>opened_at</c>.</summary>
public sealed record TradeHistoryFilter(TradeStatus? Status = null, DateTimeOffset? From = null, DateTimeOffset? To = null)
{
    public static readonly TradeHistoryFilter None = new();
}

/// <summary>
/// What the trade history page shows — filters and page — as it lives in the URL query string, with the API's own
/// parameter names and spellings. Parsed like <see cref="Predictions.PredictionHistoryState"/>: a value that doesn't
/// parse is ignored, an oversized page size is capped, anything else out of range is left for the backend to reject.
/// </summary>
public sealed record TradeHistoryState(TradeHistoryFilter Filter, int Page = 1, int PageSize = Paging.DefaultPageSize)
{
    public static readonly TradeHistoryState Default = new(TradeHistoryFilter.None);

    /// <summary>
    /// Whether the page shows the newest trades unfiltered — the only view live events can be applied to as they
    /// come (an opened trade goes on top, a closed one is updated in place) without re-checking the filters.
    /// </summary>
    public bool IsLive => Page == 1 && Filter == TradeHistoryFilter.None;

    public static TradeHistoryState FromUri(string uri)
    {
        Dictionary<string, StringValues> query = QueryHelpers.ParseQuery(new Uri(uri).Query);
        string? Get(string name) => query.TryGetValue(name, out StringValues v) ? v.ToString() : null;

        var filter = new TradeHistoryFilter(
            QueryValues.Parse<TradeStatus>(Get("status")),
            QueryValues.ParseTimestamp(Get("from")),
            QueryValues.ParseTimestamp(Get("to")));
        (int page, int pageSize) = Paging.FromQuery(Get("page"), Get("pageSize"));
        return new(filter, page, pageSize);
    }

    /// <summary>The query-string parameters, as the API query sends them; defaults and absent filters are null (removed).</summary>
    public IReadOnlyDictionary<string, object?> ToQueryParameters() =>
        (ToQuery() with
        {
            Page = Page > 1 ? Page : null,
            PageSize = PageSize != Paging.DefaultPageSize ? PageSize : null,
        })
        .ToParameters()
        .ToDictionary(p => p.Name, p => (object?)p.Value);

    public TradeHistoryQuery ToQuery() => new()
    {
        Status = Filter.Status,
        From = Filter.From,
        To = Filter.To,
        Page = Page,
        PageSize = PageSize,
    };
}
