using System.Globalization;
using System.Reflection;
using System.Text.Json.Serialization;

namespace Solia.Dashboard.Api;

// Strongly-typed query objects of the read endpoints (docs/API-CONTRACTS.md). Every property is optional: null
// (or a blank string) is omitted from the query string, so the backend applies its documented default. Ranges are
// not checked here — the backend owns them and answers 400 invalid_request, surfaced as a BackendErrorException.
// Every enum member carries its wire spelling in [JsonStringEnumMemberName].

/// <summary><c>GET /tokens/live</c> sort key (always descending).</summary>
public enum TokenSort
{
    [JsonStringEnumMemberName("updated_at")] UpdatedAt,
    [JsonStringEnumMemberName("market_cap")] MarketCap,
    [JsonStringEnumMemberName("liquidity")] Liquidity,
    [JsonStringEnumMemberName("holders")] Holders,
    [JsonStringEnumMemberName("trending_score")] TrendingScore,
    [JsonStringEnumMemberName("created_at")] CreatedAt,
}

/// <summary>A token's final outcome (<c>GET /tokens/history?outcome=</c>).</summary>
public enum TokenOutcome
{
    [JsonStringEnumMemberName("pump")] Pump,
    [JsonStringEnumMemberName("rug")] Rug,
    [JsonStringEnumMemberName("neutral")] Neutral,
}

/// <summary>A prediction horizon.</summary>
public enum PredictionHorizon
{
    [JsonStringEnumMemberName("30s")] Seconds30,
    [JsonStringEnumMemberName("1m")] Minute1,
    [JsonStringEnumMemberName("5m")] Minutes5,
    [JsonStringEnumMemberName("15m")] Minutes15,
    [JsonStringEnumMemberName("1h")] Hour1,
}

/// <summary><c>GET /predictions/live</c> sort key (always descending).</summary>
public enum PredictionSort
{
    [JsonStringEnumMemberName("ts")] Ts,
    [JsonStringEnumMemberName("pump_prob")] PumpProb,
    [JsonStringEnumMemberName("rug_prob")] RugProb,
    [JsonStringEnumMemberName("confidence")] Confidence,
}

/// <summary>A prediction's recommended strategy.</summary>
public enum TradeStrategy
{
    [JsonStringEnumMemberName("buy")] Buy,
    [JsonStringEnumMemberName("sell")] Sell,
    [JsonStringEnumMemberName("hold")] Hold,
}

/// <summary>A paper trade's status.</summary>
public enum TradeStatus
{
    [JsonStringEnumMemberName("open")] Open,
    [JsonStringEnumMemberName("closed")] Closed,
}

/// <summary>The <c>/stats/winrate</c> look-back window.</summary>
public enum StatsWindow
{
    [JsonStringEnumMemberName("all")] All,
    [JsonStringEnumMemberName("24h")] Last24Hours,
    [JsonStringEnumMemberName("7d")] Last7Days,
    [JsonStringEnumMemberName("30d")] Last30Days,
}

/// <summary>Minimum log severity (<c>debug</c> &lt; <c>info</c> &lt; <c>warn</c> &lt; <c>error</c>).</summary>
public enum BotLogLevel
{
    [JsonStringEnumMemberName("debug")] Debug,
    [JsonStringEnumMemberName("info")] Info,
    [JsonStringEnumMemberName("warn")] Warn,
    [JsonStringEnumMemberName("error")] Error,
}

/// <summary>The service a log row comes from.</summary>
public enum LogService
{
    [JsonStringEnumMemberName("data-pipeline")] DataPipeline,
    [JsonStringEnumMemberName("ai-service")] AiService,
    [JsonStringEnumMemberName("backend")] Backend,
    [JsonStringEnumMemberName("bot")] Bot,
}

/// <summary>The shared paging parameters (<c>page</c> 1-based, <c>pageSize</c> 1–200).</summary>
public abstract record PagedQuery
{
    public int? Page { get; init; }

    public int? PageSize { get; init; }

    internal IEnumerable<(string Name, string? Value)> PagingParameters()
    {
        yield return ("page", QueryValues.Format(Page));
        yield return ("pageSize", QueryValues.Format(PageSize));
    }
}

/// <summary><c>GET /tokens/live</c>.</summary>
public sealed record LiveTokensQuery
{
    public int? Limit { get; init; }

    public TokenSort? Sort { get; init; }

    internal IEnumerable<(string Name, string? Value)> ToParameters() =>
    [
        ("limit", QueryValues.Format(Limit)),
        ("sort", Sort is { } s ? QueryValues.Format(s) : null),
    ];
}

/// <summary><c>GET /tokens/history</c>: <see cref="From"/>/<see cref="To"/> bound <c>first_seen_at</c>.</summary>
public sealed record TokenHistoryQuery : PagedQuery
{
    public DateTimeOffset? From { get; init; }

    public DateTimeOffset? To { get; init; }

    public TokenOutcome? Outcome { get; init; }

    internal IEnumerable<(string Name, string? Value)> ToParameters() =>
    [
        ("from", QueryValues.Format(From)),
        ("to", QueryValues.Format(To)),
        ("outcome", Outcome is { } o ? QueryValues.Format(o) : null),
        .. PagingParameters(),
    ];
}

/// <summary><c>GET /predictions/live</c>.</summary>
public sealed record LivePredictionsQuery
{
    public int? Limit { get; init; }

    public PredictionHorizon? Horizon { get; init; }

    public PredictionSort? Sort { get; init; }

    internal IEnumerable<(string Name, string? Value)> ToParameters() =>
    [
        ("limit", QueryValues.Format(Limit)),
        ("horizon", Horizon is { } h ? QueryValues.Format(h) : null),
        ("sort", Sort is { } s ? QueryValues.Format(s) : null),
    ];
}

/// <summary><c>GET /predictions/history</c>: <see cref="Mint"/> is exact and case-sensitive; <see cref="From"/>/<see cref="To"/> bound <c>ts</c>.</summary>
public sealed record PredictionHistoryQuery : PagedQuery
{
    public string? Mint { get; init; }

    public PredictionHorizon? Horizon { get; init; }

    public TradeStrategy? Strategy { get; init; }

    public DateTimeOffset? From { get; init; }

    public DateTimeOffset? To { get; init; }

    internal IEnumerable<(string Name, string? Value)> ToParameters() =>
    [
        ("mint", Mint?.Trim()),
        ("horizon", Horizon is { } h ? QueryValues.Format(h) : null),
        ("strategy", Strategy is { } s ? QueryValues.Format(s) : null),
        ("from", QueryValues.Format(From)),
        ("to", QueryValues.Format(To)),
        .. PagingParameters(),
    ];
}

/// <summary><c>GET /trades/history</c>: <see cref="From"/>/<see cref="To"/> bound <c>opened_at</c>.</summary>
public sealed record TradeHistoryQuery : PagedQuery
{
    public TradeStatus? Status { get; init; }

    public DateTimeOffset? From { get; init; }

    public DateTimeOffset? To { get; init; }

    internal IEnumerable<(string Name, string? Value)> ToParameters() =>
    [
        ("status", Status is { } s ? QueryValues.Format(s) : null),
        ("from", QueryValues.Format(From)),
        ("to", QueryValues.Format(To)),
        .. PagingParameters(),
    ];
}

/// <summary><c>GET /logs/bot</c>: <see cref="Level"/> is a minimum severity.</summary>
public sealed record BotLogsQuery
{
    public BotLogLevel? Level { get; init; }

    public LogService? Service { get; init; }

    public int? Limit { get; init; }

    internal IEnumerable<(string Name, string? Value)> ToParameters() =>
    [
        ("level", Level is { } l ? QueryValues.Format(l) : null),
        ("service", Service is { } s ? QueryValues.Format(s) : null),
        ("limit", QueryValues.Format(Limit)),
    ];
}

/// <summary>The documented wire spelling of every query value, and the query-string builder.</summary>
internal static class QueryValues
{
    public static string? Format(int? value) => value?.ToString(CultureInfo.InvariantCulture);

    /// <summary>ISO-8601 in UTC with a <c>Z</c> suffix (no <c>+</c> to mis-decode as a space).</summary>
    public static string? Format(DateTimeOffset? value) =>
        value?.UtcDateTime.ToString("yyyy-MM-dd'T'HH:mm:ss.FFFFFFF'Z'", CultureInfo.InvariantCulture);

    /// <summary>The member's documented wire spelling, its <see cref="JsonStringEnumMemberNameAttribute"/>.</summary>
    public static string Format<T>(T value) where T : struct, Enum =>
        WireNames<T>.ByValue.TryGetValue(value, out string? name) ? name : throw Unknown(value);

    /// <summary><paramref name="path"/> plus the non-blank parameters, escaped, in the given order.</summary>
    public static string Build(string path, IEnumerable<(string Name, string? Value)> parameters)
    {
        string query = string.Join('&', parameters
            .Where(p => !string.IsNullOrWhiteSpace(p.Value))
            .Select(p => $"{p.Name}={Uri.EscapeDataString(p.Value!)}"));
        return query.Length == 0 ? path : $"{path}?{query}";
    }

    private static ArgumentOutOfRangeException Unknown<T>(T value) where T : struct, Enum =>
        new(nameof(value), value, $"Unknown {typeof(T).Name} value.");

    private static class WireNames<T> where T : struct, Enum
    {
        public static readonly IReadOnlyDictionary<T, string> ByValue = typeof(T)
            .GetFields(BindingFlags.Public | BindingFlags.Static)
            .ToDictionary(
                f => (T)f.GetValue(null)!,
                f => f.GetCustomAttribute<JsonStringEnumMemberNameAttribute>()?.Name
                    ?? throw new InvalidOperationException($"{typeof(T).Name}.{f.Name} has no [JsonStringEnumMemberName]."));
    }
}
