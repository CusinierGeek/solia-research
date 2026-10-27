using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.Primitives;
using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Tables;

namespace Solia.Dashboard.Predictions;

/// <summary>The prediction history filters (null = not filtered); <see cref="Mint"/> is trimmed, blank is null.</summary>
public sealed record PredictionHistoryFilter(
    string? Mint = null,
    PredictionHorizon? Horizon = null,
    TradeStrategy? Strategy = null,
    DateTimeOffset? From = null,
    DateTimeOffset? To = null)
{
    public static readonly PredictionHistoryFilter None = new();

    private readonly string? _mint = Normalize(Mint);

    public string? Mint { get => _mint; init => _mint = Normalize(value); }

    private static string? Normalize(string? mint) => string.IsNullOrWhiteSpace(mint) ? null : mint.Trim();
}

/// <summary>
/// What the prediction history page shows — filters and page — as it lives in the URL query string, so a view is
/// shareable and survives a refresh. Uses the API's own parameter names and spellings; a value that doesn't parse
/// is ignored (treated as absent) rather than failing the page, a page size over the API's maximum is capped to it,
/// and anything else out of range (a page past the depth cap, <c>from &gt; to</c>) is left for the backend to reject
/// with a message.
/// </summary>
public sealed record PredictionHistoryState(PredictionHistoryFilter Filter, int Page = 1, int PageSize = Paging.DefaultPageSize)
{
    public static readonly PredictionHistoryState Default = new(PredictionHistoryFilter.None);

    public static PredictionHistoryState FromUri(string uri)
    {
        Dictionary<string, StringValues> query = QueryHelpers.ParseQuery(new Uri(uri).Query);
        string? Get(string name) => query.TryGetValue(name, out StringValues v) ? v.ToString() : null;

        var filter = new PredictionHistoryFilter(
            Get("mint"),
            QueryValues.Parse<PredictionHorizon>(Get("horizon")),
            QueryValues.Parse<TradeStrategy>(Get("strategy")),
            QueryValues.ParseTimestamp(Get("from")),
            QueryValues.ParseTimestamp(Get("to")));
        (int page, int pageSize) = Paging.FromQuery(Get("page"), Get("pageSize"));
        return new(filter, page, pageSize);
    }

    /// <summary>
    /// The query-string parameters, named and spelled as the API query sends them (<see cref="ToQuery"/>); defaults
    /// (page 1, the default page size) and absent filters are null, i.e. removed.
    /// </summary>
    public IReadOnlyDictionary<string, object?> ToQueryParameters() =>
        (ToQuery() with
        {
            Page = Page > 1 ? Page : null,
            PageSize = PageSize != Paging.DefaultPageSize ? PageSize : null,
        })
        .ToParameters()
        .ToDictionary(p => p.Name, p => (object?)p.Value);

    public PredictionHistoryQuery ToQuery() => new()
    {
        Mint = Filter.Mint,
        Horizon = Filter.Horizon,
        Strategy = Filter.Strategy,
        From = Filter.From,
        To = Filter.To,
        Page = Page,
        PageSize = PageSize,
    };
}
