using Solia.Dashboard.Api;

namespace Solia.Dashboard.Live;

/// <summary>One server → client event of the backend's live hub, typed by its payload.</summary>
public sealed class LiveEvent<T>
{
    internal LiveEvent(string name) => Name = name;

    /// <summary>The hub method name (docs/API-CONTRACTS.md, "Live hub").</summary>
    public string Name { get; }

    public override string ToString() => Name;
}

/// <summary>
/// The live hub's contract as the dashboard sees it: its path under <c>Backend:BaseUrl</c> and its events, each
/// carrying the same DTO as the matching REST item.
/// </summary>
public static class LiveFeedEvents
{
    /// <summary>Where the backend maps the hub, relative to <c>Backend:BaseUrl</c> (outside <c>/api/v1</c>).</summary>
    public const string HubPath = "hubs/live";

    /// <summary>A prediction was recorded; payload = a <c>/predictions/history</c> item.</summary>
    public static readonly LiveEvent<PredictionRecordDto> PredictionCreated = new("prediction_created");

    /// <summary>A paper trade was opened; payload = a <c>/trades/history</c> item.</summary>
    public static readonly LiveEvent<TradeDto> TradeOpened = new("trade_opened");

    /// <summary>A paper trade was closed; payload = a <c>/trades/history</c> item.</summary>
    public static readonly LiveEvent<TradeDto> TradeClosed = new("trade_closed");

    /// <summary>A non-debug log row was written by any service; payload = a <c>/logs/bot</c> item.</summary>
    public static readonly LiveEvent<LogDto> LogWritten = new("log_written");

    /// <summary>Sent after every <see cref="TradeClosed"/>: re-fetch <c>/stats/winrate</c> and <c>/stats/performance</c>.</summary>
    public static readonly LiveEvent<StatsChangedDto> StatsChanged = new("stats_changed");
}
