using System.Text.Json;

namespace Solia.Dashboard.Api;

// Response DTOs of the backend's read endpoints (docs/API-CONTRACTS.md), owned by the dashboard: the backend's
// own records live in Solia.Api, which would drag ASP.NET Core and EF into the UI. Their wire shape is pinned by
// JSON fixture tests taken from the contract. Deserialized snake_case (SoliaApiJson.Options); money/prices are
// decimal (exact), probabilities/ratios double, timestamps UTC DateTimeOffset. Enum-like fields stay strings so a
// value the backend adds later still deserializes.

/// <summary>One page of <typeparamref name="T"/> plus the total row count matching the filters.</summary>
public sealed record PagedResult<T>(IReadOnlyList<T> Items, int Page, int PageSize, long Total);

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

/// <summary>A registry entry (<c>GET /tokens/history</c>).</summary>
public sealed record TokenHistoryDto(
    string Mint,
    string? Symbol,
    string? Name,
    DateTimeOffset? CreatedAt,
    DateTimeOffset FirstSeenAt,
    DateTimeOffset? LastSeenAt,
    string? FinalOutcome);

/// <summary>
/// A recorded prediction (<c>GET /predictions/live</c>, <c>GET /predictions/history</c>, the hub's
/// <c>prediction_created</c>); <paramref name="ModelVersion"/> is null when the row has no model.
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

/// <summary>A paper trade (<c>GET /trades/history</c>, the hub's <c>trade_opened</c> / <c>trade_closed</c>).</summary>
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

/// <summary><c>GET /stats/winrate</c>.</summary>
public sealed record WinRateResponse(double Winrate, int Wins, int Losses, int BreakEven, string Window);

/// <summary>One equity-curve point: the equity right after the trade closed at <paramref name="Ts"/>, and its P&amp;L.</summary>
public sealed record EquityPoint(DateTimeOffset Ts, decimal Equity, decimal Pnl);

/// <summary><c>GET /stats/performance</c>.</summary>
public sealed record PerformanceResponse(IReadOnlyList<EquityPoint> Points, decimal TotalPnl);

/// <summary>
/// The hub's <c>stats_changed</c> signal, sent right after every <c>trade_closed</c>: the <c>/stats/*</c> figures
/// moved — re-fetch them.
/// </summary>
public sealed record StatsChangedDto(long TradeId, DateTimeOffset? ClosedAt);

/// <summary>An operational log entry (<c>GET /logs/bot</c>); <paramref name="Context"/> is the structured JSON as-is.</summary>
public sealed record LogDto(long Id, DateTimeOffset Ts, string Service, string Level, string Message, JsonElement? Context);

/// <summary>
/// <c>GET /model/status</c>: the serving champion (null when the ai-service is down or has none yet; then
/// <paramref name="Metrics"/> is empty) plus the retrain progress.
/// </summary>
public sealed record ModelStatusResponse(
    string? ChampionVersion,
    DateTimeOffset? CreatedAt,
    IReadOnlyDictionary<string, JsonElement> Metrics,
    long TradesSinceLastTrain,
    int RetrainThreshold);

/// <summary>
/// The documented error envelope: <c>{ "error": { "code": "...", "message": "..." } }</c>. Its members are optional
/// so a partial envelope still yields what it carries.
/// </summary>
internal sealed record ApiErrorEnvelope(ApiErrorBody? Error = null);

internal sealed record ApiErrorBody(string? Code = null, string? Message = null);

/// <summary>
/// The backend's JSON conventions: snake_case properties, web defaults otherwise. Strict about the contract: every
/// constructor parameter without a default must be present (the backend writes nulls rather than omitting them),
/// and a null for a non-nullable reference is rejected, so contract drift is an invalid_response, not a silent
/// default.
/// </summary>
internal static class SoliaApiJson
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web)
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        RespectNullableAnnotations = true,
        RespectRequiredConstructorParameters = true,
    };
}
