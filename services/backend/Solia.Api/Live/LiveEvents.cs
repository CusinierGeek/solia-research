namespace Solia.Api.Live;

/// <summary>
/// The live hub's contract (docs/API-CONTRACTS.md, "Live hub"): its path and its server → client event names.
/// The four row events are also the <c>&lt;event&gt;</c> of the <c>&lt;event&gt;:&lt;id&gt;</c> payload the
/// 0004 migration's triggers NOTIFY on <see cref="Channel"/>.
/// </summary>
public static class LiveEvents
{
    /// <summary>Where the hub is mapped — outside <c>/api/v1</c>, like the health probes.</summary>
    public const string Path = "/hubs/live";

    /// <summary>The Postgres NOTIFY channel the triggers raise and the backend LISTENs on.</summary>
    public const string Channel = "solia_live";

    /// <summary>A <c>predictions</c> row was inserted; payload = a <c>/predictions/history</c> item.</summary>
    public const string PredictionCreated = "prediction_created";

    /// <summary>A <c>trades</c> row was inserted open; payload = a <c>/trades/history</c> item.</summary>
    public const string TradeOpened = "trade_opened";

    /// <summary>A <c>trades</c> row was closed; payload = a <c>/trades/history</c> item.</summary>
    public const string TradeClosed = "trade_closed";

    /// <summary>A <c>logs</c> row was inserted (any service); payload = a <c>/logs/bot</c> item.</summary>
    public const string LogWritten = "log_written";

    /// <summary>Sent right after every <see cref="TradeClosed"/>: the <c>/stats/*</c> figures moved.</summary>
    public const string StatsChanged = "stats_changed";

    /// <summary>The events a NOTIFY can carry (every event but the derived <see cref="StatsChanged"/>).</summary>
    public static readonly string[] RowEvents = [PredictionCreated, TradeOpened, TradeClosed, LogWritten];
}
