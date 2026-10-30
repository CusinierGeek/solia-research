using System.Text.Json;
using System.Text.Json.Serialization;

namespace Solia.Api.Endpoints;

// Response DTOs of the public read endpoints (docs/API-CONTRACTS.md), deliberately separate from the EF
// entities so the wire shape never follows a mapping change. Serialized snake_case by the host's JSON
// options; money/prices stay decimal (exact), probabilities/ratios double, timestamps ISO-8601 UTC.

/// <summary>A currently tracked token with its latest snapshot (<c>GET /tokens/live</c>).</summary>
public sealed record LiveTokenDto(
    string Mint,
    string? Symbol,
    string? Name,
    DateTimeOffset? CreatedAt,
    decimal? LastPrice,
    decimal? LastMarketCap,
    decimal? LastLiquidity,
    int? Holders,
    double? TrendingScore,
    string? Venue,
    bool IsMigrated,
    DateTimeOffset UpdatedAt);

/// <summary>A registry entry (<c>GET /tokens/history</c>); the raw source metadata is not exposed.</summary>
public sealed record TokenHistoryDto(
    string Mint,
    string? Symbol,
    string? Name,
    DateTimeOffset? CreatedAt,
    DateTimeOffset FirstSeenAt,
    DateTimeOffset? LastSeenAt,
    string? FinalOutcome);

/// <summary>
/// A recorded prediction (<c>GET /predictions/live</c>, <c>GET /predictions/history</c>);
/// <paramref name="ModelVersion"/> is null when the row has no model.
/// </summary>
public sealed record PredictionRecordDto(
    long Id,
    string Mint,
    string? Symbol,
    DateTimeOffset Ts,
    string Horizon,
    double? PumpProb,
    double? RugProb,
    string? Strategy,
    double? Confidence,
    decimal? PositionSize,
    decimal? TakeProfit,
    decimal? StopLoss,
    string? ModelVersion);

/// <summary>A paper trade (<c>GET /trades/history</c>); the entry features snapshot is not exposed.</summary>
public sealed record TradeDto(
    long Id,
    string Mint,
    long? PredictionId,
    DateTimeOffset OpenedAt,
    DateTimeOffset? ClosedAt,
    decimal? EntryPrice,
    decimal? ExitPrice,
    decimal? Size,
    int? DurationSec,
    decimal? Pnl,
    double? PnlPct,
    string? Strategy,
    double? Confidence,
    string? ExitReason,
    string Status);

/// <summary>
/// <c>GET /stats/winrate</c>: <c>{ winrate, wins, losses, break_even, window, session, exits }</c> — <c>session</c> is
/// the bot session read (null: every trade).
/// </summary>
public sealed record WinRateResponse(
    double Winrate, int Wins, int Losses, int BreakEven, string Window, long? Session = null, ExitBreakdown? Exits = null);

/// <summary>How the closed trades of a <c>/stats/winrate</c> scope exited, by <c>exit_reason</c>.</summary>
public sealed record ExitBreakdown(int Tp, int Sl, int Signal, int Timeout);

/// <summary>One equity-curve point: the equity right after the trade closed at <paramref name="Ts"/>, and its P&amp;L.</summary>
public sealed record EquityPoint(DateTimeOffset Ts, decimal Equity, decimal Pnl);

/// <summary>
/// <c>GET /stats/performance</c>: <c>{ points: [{ ts, equity, pnl }], total_pnl, session }</c> — <c>session</c> is the
/// bot session read (null: every trade).
/// </summary>
public sealed record PerformanceResponse(IReadOnlyList<EquityPoint> Points, decimal TotalPnl, long? Session = null);

/// <summary>
/// The live hub's <c>stats_changed</c>: a signal, not the figures — the trade whose close moved
/// <c>/stats/winrate</c> and <c>/stats/performance</c>, which clients re-fetch.
/// </summary>
public sealed record StatsChangedDto(long TradeId, DateTimeOffset? ClosedAt);

/// <summary>An operational log entry (<c>GET /logs/bot</c>); <paramref name="Context"/> is the structured JSON as-is.</summary>
public sealed record LogDto(long Id, DateTimeOffset Ts, string Service, string Level, string Message, RawJson? Context);

/// <summary>
/// A JSON value already serialized — a <c>jsonb</c> column as Postgres returns it — written to the response
/// verbatim instead of being parsed and re-serialized per row.
/// </summary>
[JsonConverter(typeof(RawJsonConverter))]
public readonly record struct RawJson(string Json);

internal sealed class RawJsonConverter : JsonConverter<RawJson>
{
    public override RawJson Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var document = JsonDocument.ParseValue(ref reader);
        return new RawJson(document.RootElement.GetRawText());
    }

    // Postgres only ever returns valid jsonb text, so the writer's re-validation is skipped.
    public override void Write(Utf8JsonWriter writer, RawJson value, JsonSerializerOptions options) =>
        writer.WriteRawValue(value.Json, skipInputValidation: true);
}

/// <summary>
/// A paper bot session (<c>GET /bot/sessions</c>): one configuration + champion. <c>open</c> / <c>committed</c> count the
/// positions the bot still holds; <c>orphaned</c> the open trades an earlier bot run left behind.
/// </summary>
public sealed record BotSessionDto(
    long Id,
    DateTimeOffset StartedAt,
    DateTimeOffset? EndedAt,
    string? Label,
    long? ModelId,
    string? ModelVersion,
    RawJson? Options,
    bool IsCurrent,
    long Trades,
    long Closed,
    long Open,
    long Orphaned,
    decimal TotalPnl,
    decimal Committed);
