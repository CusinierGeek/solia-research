using System.Data.Common;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Options;
using Solia.Api.Endpoints;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Intake;
using Solia.Bot.Policy;
using Solia.Core.Ai;
using Solia.Core.Configuration;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Bot;

/// <summary>A <c>features</c> row as the feed reads it; <see cref="Tracked"/> is its token's <c>tokens_live.is_active</c>.</summary>
internal sealed record FeedRow(long Id, string Mint, DateTimeOffset Ts, decimal? Price, bool Tracked);

/// <summary>
/// The bot's live feed: polls the <c>features</c> snapshots the data-pipeline writes and turns them into
/// <see cref="BotEvent"/>s — "derive, don't fetch": no market call, only the free stream's aggregates.
/// Two loops run side by side for as long as the stream is read:
/// <list type="number">
///   <item>the <em>poll</em> loop reads the new rows by id (<see cref="FeatureCursor"/>: an id that commits
///   late is still read) and emits every priced row of a <em>tracked</em> token (<c>tokens_live.is_active</c>)
///   as a <see cref="BotEvent.Price"/> tick at its snapshot time, then queues the token for scoring;</item>
///   <item>the <em>scoring</em> loop takes the queued tokens one at a time, oldest-waiting first
///   (<see cref="ScoringQueue"/>), scores each on its feature window <em>up to the queued row</em> — the
///   window <c>POST /predict</c> sends, never a newer row — through the ai-service, persists the
///   <see cref="BotHostOptions.Horizon"/> prediction to <c>predictions</c> and emits it as a
///   <see cref="BotEvent.Prediction"/> at that row's time. A slow ai-service never holds the ticks back.</item>
/// </list>
/// Both go through a <see cref="ReorderBuffer"/> of <see cref="BotHostOptions.ReorderWindow"/>, so the events
/// reach the engine in time order. Rows older than <see cref="BotOptions.MaxMarkAgeSeconds"/> are never
/// emitted nor scored: a stale mark can't support an entry (so a restart or a historical replay into
/// <c>features</c> isn't traded). A failed scoring (ai-service down, window too short, malformed answer, DB
/// error, any unexpected error) skips that token — an unavailable ai-service pauses scoring for a poll
/// interval — and a failed read is retried on the next poll: an outage pauses the feed, it never stops the
/// bot. The stream stays open until cancelled. <strong>Paper only.</strong>
/// </summary>
public sealed class DbBotEventSource(
    IServiceScopeFactory scopes,
    ISoliaAiClient ai,
    PostgresPredictionStore predictions,
    IOptions<BotHostOptions> hostOptions,
    IOptions<BotOptions> botOptions,
    IOptions<AiServiceOptions> aiOptions,
    TimeProvider time,
    ILogger<DbBotEventSource> logger) : IBotEventSource
{
    private readonly BotHostOptions _host = hostOptions.Value;

    /// <summary>One read of the stream's state: the buffer is shared by both loops, under its own lock.</summary>
    private sealed class Feed(ReorderBuffer buffer, ChannelWriter<BotEvent> output)
    {
        public ScoringQueue Scoring { get; } = new();

        public void Emit(IEnumerable<BotEvent> events)
        {
            lock (buffer)
            {
                foreach (BotEvent botEvent in events)
                {
                    buffer.Add(botEvent);
                }

                foreach (BotEvent due in buffer.Release())
                {
                    output.TryWrite(due);
                }
            }
        }
    }

    public async IAsyncEnumerable<BotEvent> ReadAsync([EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        Channel<BotEvent> output = Channel.CreateUnbounded<BotEvent>(new UnboundedChannelOptions { SingleReader = true });
        var feed = new Feed(new ReorderBuffer(time, _host.ReorderWindow), output.Writer);
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        Task loops = RunLoopsAsync(feed, output.Writer, stop);
        try
        {
            await foreach (BotEvent botEvent in output.Reader.ReadAllAsync(cancellationToken))
            {
                yield return botEvent;
            }
        }
        finally
        {
            await stop.CancelAsync();
            await loops;
        }
    }

    /// <summary>Runs both loops until cancelled; an unexpected failure of either ends the stream with it.</summary>
    private async Task RunLoopsAsync(Feed feed, ChannelWriter<BotEvent> output, CancellationTokenSource stop)
    {
        Task poll = Task.Run(() => PollLoopAsync(feed, stop.Token));
        Task score = Task.Run(() => ScoreLoopAsync(feed, stop.Token));
        Exception? failure = null;
        try
        {
            await await Task.WhenAny(poll, score);
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            // Normal end of the stream.
        }
        catch (Exception ex)
        {
            failure = ex;
        }

        await stop.CancelAsync();
        try
        {
            await Task.WhenAll(poll, score);
        }
        catch (Exception) when (stop.IsCancellationRequested)
        {
            // The other loop's cancellation, or the failure already captured.
        }

        output.TryComplete(failure);
    }

    private async Task PollLoopAsync(Feed feed, CancellationToken cancellationToken)
    {
        var cursor = new FeatureCursor(time, _host.MaxCommitDelay, logger);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            List<BotEvent> ticks = [];
            List<FeedRow> toScore = [];
            bool backlog = false;
            try
            {
                List<FeedRow> rows = await ReadRowsAsync(cursor, cancellationToken);
                backlog = rows.Count == _host.MaxTicksPerPoll;
                DateTimeOffset since = time.GetUtcNow().AddSeconds(-botOptions.Value.MaxMarkAgeSeconds);
                foreach (FeedRow row in rows)
                {
                    if (!cursor.TryAccept(row.Id) || !row.Tracked || row.Ts < since)
                    {
                        continue;
                    }

                    if (row.Price is decimal price)
                    {
                        ticks.Add(new BotEvent.Price(new PriceTick { Mint = row.Mint, Ts = row.Ts, Price = price }));
                    }

                    toScore.Add(row);
                }

                cursor.ExpireGaps();
            }
            catch (DbException ex) when (!cancellationToken.IsCancellationRequested)
            {
                logger.LogWarning(ex, "bot feed: reading features failed, retrying on the next poll");
            }

            // Ticks first, so a prediction always follows the tick it was scored on. Emitting with no new tick
            // still releases the events that waited out the reorder window.
            feed.Emit(ticks);
            toScore.ForEach(feed.Scoring.Mark);

            if (!backlog)
            {
                await Task.Delay(_host.PollInterval, time, cancellationToken);
            }
        }
    }

    /// <summary>
    /// The next rows by id: the ones after the cursor plus the late commits it still waits for. The first read
    /// starts at the oldest fresh row; later reads aren't filtered in SQL so every committed id is seen.
    /// </summary>
    private async Task<List<FeedRow>> ReadRowsAsync(FeatureCursor cursor, CancellationToken cancellationToken)
    {
        await using AsyncServiceScope scope = scopes.CreateAsyncScope();
        SoliaDbContext db = scope.ServiceProvider.GetRequiredService<SoliaDbContext>();
        IQueryable<Feature> query = db.Features.AsNoTracking();
        if (cursor.MaxSeen is long after)
        {
            long[] gaps = [.. cursor.Gaps];
            query = query.Where(f => f.Id > after || gaps.Contains(f.Id));
        }
        else
        {
            DateTimeOffset since = time.GetUtcNow().AddSeconds(-botOptions.Value.MaxMarkAgeSeconds);
            query = query.Where(f => f.Ts >= since);
        }

        List<FeedRow> rows = await query
            .OrderBy(f => f.Id)
            .Take(_host.MaxTicksPerPoll)
            .Select(f => new FeedRow(f.Id, f.Mint, f.Ts, f.Price, db.TokensLive.Any(l => l.Mint == f.Mint && l.IsActive)))
            .ToListAsync(cancellationToken);
        if (rows.Count == 0 && cursor.MaxSeen is null)
        {
            cursor.Seed(await db.Features.MaxAsync(f => (long?)f.Id, cancellationToken) ?? 0);
        }

        return rows;
    }

    private async Task ScoreLoopAsync(Feed feed, CancellationToken cancellationToken)
    {
        while (true)
        {
            FeedRow row = await feed.Scoring.TakeAsync(cancellationToken);
            if (row.Ts < time.GetUtcNow().AddSeconds(-botOptions.Value.MaxMarkAgeSeconds))
            {
                continue; // went stale while queued: no entry could be taken on it
            }

            ModelPrediction? prediction;
            try
            {
                prediction = await ScoreAsync(row, cancellationToken);
            }
            catch (AiServiceException ex) when (ex.Kind == AiServiceErrorKind.Unavailable)
            {
                // Down or circuit open: the other tokens would fail the same way — keep this one and pause.
                logger.LogWarning("bot feed: ai-service unavailable, scoring paused for a poll interval: {Message}", ex.Message);
                feed.Scoring.Mark(row);
                await Task.Delay(_host.PollInterval, time, cancellationToken);
                continue;
            }
            catch (AiServiceException ex)
            {
                logger.LogInformation("bot feed: {Mint} not scored ({Kind}): {Message}", row.Mint, ex.Kind, ex.UpstreamMessage ?? ex.Message);
                continue;
            }
            catch (DbException ex) when (!cancellationToken.IsCancellationRequested)
            {
                logger.LogWarning(ex, "bot feed: scoring {Mint} failed on the database, skipped", row.Mint);
                continue;
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                // A bug or an unexpected answer must cost one token's scoring, not the bot (and the host).
                logger.LogError(ex, "bot feed: scoring {Mint} failed unexpectedly, skipped", row.Mint);
                continue;
            }

            if (prediction is not null)
            {
                feed.Emit([new BotEvent.Prediction(prediction)]);
            }
        }
    }

    /// <summary>
    /// Scores <paramref name="row"/>'s token on its feature window ending at that row and persists the traded
    /// horizon's prediction; null when the answer has no usable prediction for that horizon.
    /// </summary>
    private async Task<ModelPrediction?> ScoreAsync(FeedRow row, CancellationToken cancellationToken)
    {
        // A scope per load: the context isn't held across the ai-service call.
        List<IReadOnlyDictionary<string, double?>> window;
        await using (AsyncServiceScope scope = scopes.CreateAsyncScope())
        {
            window = await PredictEndpoints.LoadWindowAsync(scope.ServiceProvider.GetRequiredService<SoliaDbContext>(),
                row.Mint, aiOptions.Value.MaxFeatureWindowSize, cancellationToken, upTo: row.Ts);
        }

        if (window.Count == 0)
        {
            return null;
        }

        AiPredictResponse response = await ai.PredictAsync(
            AiPredictRequest.ForWindow(row.Mint, window, [_host.Horizon]), cancellationToken);
        AiPrediction? scored = response.Predictions?.FirstOrDefault(p => p?.Horizon == _host.Horizon);
        if (scored is null)
        {
            logger.LogWarning("bot feed: the ai-service returned no {Horizon} prediction for {Mint}", _host.Horizon, row.Mint);
            return null;
        }

        IReadOnlyDictionary<string, double?> features = window[^1];
        var prediction = new ModelPrediction
        {
            Mint = row.Mint,
            Ts = row.Ts,
            Horizon = scored.Horizon,
            PumpProb = scored.PumpProb,
            RugProb = scored.RugProb,
            ExpectedReturn = scored.ExpectedReturn,
            Confidence = scored.Confidence,
            Volatility = features.GetValueOrDefault("volatility"),
            ModelVersion = response.ModelVersion,
            Features = features,
        };

        // Persisting it would trip the table's CHECKs; the policy layer would fail closed on it anyway.
        if (!PredictionValidation.IsWellFormed(prediction))
        {
            logger.LogWarning("bot feed: malformed prediction for {Mint} dropped ({@Prediction})", row.Mint, scored);
            return null;
        }

        long id = await predictions.InsertAsync(prediction, row.Id, cancellationToken);
        return prediction with { PredictionId = id };
    }
}
