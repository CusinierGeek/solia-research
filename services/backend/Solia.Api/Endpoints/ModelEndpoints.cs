using System.Text.Json;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Solia.Core.Ai;
using Solia.Core.Configuration;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary>
/// <c>GET /model/status</c> and <c>POST /model/retrain</c>, proxied to the ai-service. Only the proxy surface
/// lives here — the retrain orchestrator itself is the ai-service's (Sprint 7).
/// </summary>
public static class ModelEndpoints
{
    public static RouteGroupBuilder MapModelEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/model/status", GetStatusAsync);
        group.MapPost("/model/retrain", RetrainAsync);
        return group;
    }

    /// <summary>
    /// The ai-service's champion and metrics, plus the retrain progress. The ai-service call and the local trade
    /// count run concurrently. Progress is one consistent pair: the ai-service's own count and threshold when it
    /// reports both, else the local count (<see cref="CountTradesSinceLastTrainAsync"/>) against
    /// <see cref="AiServiceOptions.RetrainThreshold"/>. An ai-service failure (down, no champion yet) does not
    /// fail the call: the champion fields come back null / empty and the progress is the local pair.
    /// </summary>
    internal static async Task<ModelStatusResponse> GetStatusAsync(
        SoliaDbContext db,
        ISoliaAiClient ai,
        IOptions<AiServiceOptions> options,
        ILoggerFactory loggers,
        CancellationToken cancellationToken)
    {
        Task<AiModelStatus?> statusCall = TryGetChampionStatusAsync(ai, loggers, cancellationToken);
        long localCount = await CountTradesSinceLastTrainAsync(db, cancellationToken);
        AiModelStatus? status = await statusCall;

        (long tradesSince, int threshold) = status is { TradesSinceLastTrain: { } upstreamCount, RetrainThreshold: { } upstreamThreshold }
            ? (upstreamCount, upstreamThreshold)
            : (localCount, options.Value.RetrainThreshold);

        return new ModelStatusResponse(
            status?.ChampionVersion,
            status?.CreatedAt,
            status?.Metrics ?? new Dictionary<string, JsonElement>(),
            tradesSince,
            threshold);
    }

    /// <summary>The ai-service's status, or null (logged with the upstream details) when it cannot report one.</summary>
    private static async Task<AiModelStatus?> TryGetChampionStatusAsync(
        ISoliaAiClient ai, ILoggerFactory loggers, CancellationToken cancellationToken)
    {
        try
        {
            return await ai.GetModelStatusAsync(cancellationToken);
        }
        catch (AiServiceException ex)
        {
            loggers.CreateLogger(typeof(ModelEndpoints)).LogWarning(ex,
                "model status without the champion: ai-service {Kind} (status {UpstreamStatus}, code {UpstreamCode}): {UpstreamMessage}",
                ex.Kind, ex.UpstreamStatusCode, ex.UpstreamCode, ex.UpstreamMessage);
            return null;
        }
    }

    /// <summary>
    /// <see cref="ClosedTrades">Closed trades</see> closed strictly after the latest <em>finished</em> training run
    /// started (its data cut-off), or every closed trade before any run finished. A run still in flight — or one
    /// that crashed and never wrote <c>finished_at</c> — trained no model, so it does not reset the count; a
    /// finished one does whatever its decision, since a rejected candidate still consumed those trades.
    /// </summary>
    internal static async Task<long> CountTradesSinceLastTrainAsync(SoliaDbContext db, CancellationToken cancellationToken)
    {
        DateTimeOffset? lastTrain = await db.TrainingEvents.AsNoTracking()
            .Where(e => e.FinishedAt != null)
            .MaxAsync(e => (DateTimeOffset?)e.StartedAt, cancellationToken);

        IQueryable<Trade> closed = db.Trades.AsNoTracking().Closed();
        if (lastTrain is { } since)
        {
            closed = closed.Where(t => t.ClosedAt > since);
        }

        return await closed.LongCountAsync(cancellationToken);
    }

    /// <summary>Kicks off a manual retrain on the ai-service: <c>202 Accepted</c> + the run's <c>training_event_id</c>.</summary>
    internal static async Task<IResult> RetrainAsync(ISoliaAiClient ai, CancellationToken cancellationToken)
    {
        AiRetrainResponse response = await ai.TriggerRetrainAsync(cancellationToken);
        return TypedResults.Accepted((string?)null, new RetrainResponse(response.TrainingEventId));
    }
}
