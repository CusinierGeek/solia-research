using System.Globalization;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.Primitives;
using Solia.Dashboard.Api;

namespace Solia.Dashboard.Logs;

/// <summary>
/// What the bot logs page shows, as it lives in the URL query string with the API's own parameter names and
/// spellings: a minimum <see cref="Level"/> and a <see cref="Service"/> (null = not filtered), and how many entries
/// to keep (<see cref="Limit"/>, null = the API's default, which it is normalised to). Parsed like the other views' state: a value that doesn't
/// parse is ignored, a limit over the API's maximum is capped to it. <see cref="Matches"/> applies the same rules as
/// <c>GET /logs/bot</c>, so a pushed entry is kept exactly when a re-fetch would have returned it.
/// </summary>
public sealed record BotLogsFilter(BotLogLevel? Level = null, LogService? Service = null, int? Limit = null)
{
    /// <summary>The API's default <c>limit</c>.</summary>
    public const int DefaultLimit = 100;

    /// <summary>The largest <c>limit</c> the API serves.</summary>
    public const int MaxLimit = 1000;

    // The default limit is held as null, so a filter equals itself however its limit was spelled.
    private readonly int? _limit = Normalize(Limit);

    public static readonly BotLogsFilter None = new();

    public int? Limit { get => _limit; init => _limit = Normalize(value); }

    /// <summary>The limits the filter bar offers.</summary>
    public static readonly IReadOnlyList<int> Limits = [50, DefaultLimit, 250, 500, MaxLimit];

    /// <summary>How many entries are fetched and kept on screen.</summary>
    public int EffectiveLimit => Limit ?? DefaultLimit;

    public static BotLogsFilter FromUri(string uri)
    {
        Dictionary<string, StringValues> query = QueryHelpers.ParseQuery(new Uri(uri).Query);
        string? Get(string name) => query.TryGetValue(name, out StringValues v) ? v.ToString() : null;

        return new(QueryValues.Parse<BotLogLevel>(Get("level")), QueryValues.Parse<LogService>(Get("service")),
            ParseLimit(Get("limit")));
    }

    // Digits too many for an int are still a limit over the maximum.
    private static int? ParseLimit(string? value) =>
        int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int n)
            ? n > 0 ? Math.Min(n, MaxLimit) : null
            : !string.IsNullOrEmpty(value) && value.All(char.IsAsciiDigit) ? MaxLimit : null;

    private static int? Normalize(int? limit) => limit == DefaultLimit ? null : limit;

    /// <summary>The query-string parameters; absent filters and the default limit are null (removed).</summary>
    public IReadOnlyDictionary<string, object?> ToQueryParameters() =>
        ToQuery()
        .ToParameters()
        .ToDictionary(p => p.Name, p => (object?)p.Value);

    public BotLogsQuery ToQuery() => new() { Level = Level, Service = Service, Limit = Limit };

    /// <summary>
    /// Whether <paramref name="log"/> passes the filters: its service is the one asked for, and its level is at
    /// least the minimum (<c>debug</c> &lt; <c>info</c> &lt; <c>warn</c> &lt; <c>error</c>; a level outside that
    /// scale never meets a minimum).
    /// </summary>
    public bool Matches(LogDto log) =>
        (Service is not { } service || log.Service == QueryValues.Format(service))
        && (Level is not { } min || (QueryValues.Parse<BotLogLevel>(log.Level) is { } level && level >= min));
}
