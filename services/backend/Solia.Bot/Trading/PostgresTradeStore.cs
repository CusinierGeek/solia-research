using System.Text.Json;
using Npgsql;
using NpgsqlTypes;
using Solia.Bot.Policy;
using Solia.Bot.Sessions;
using static Solia.Shared.Data.NpgsqlParameters;

namespace Solia.Bot.Trading;

/// <summary>
/// Persists trades to the <c>trades</c> table (docs/DATA-MODEL.md): one INSERT on open returning the
/// row id, one UPDATE on close. Values are checked against the table's CHECK constraints before they
/// reach the database, so a bad value fails with an error naming the field rather than a constraint
/// violation. The FKs are respected, not satisfied: the mint must already be in <c>tokens_history</c>
/// (written by the data pipeline) and a non-null <c>prediction_id</c> must exist in <c>predictions</c>.
/// With <paramref name="sessions"/> (the bot), every opened trade records its <see cref="BotSessions">bot session</see>;
/// without (a standalone store), <c>session_id</c> stays null.
/// </summary>
public sealed class PostgresTradeStore(NpgsqlDataSource dataSource, BotSessions? sessions = null) : ITradeStore, IClosedTradeCounter
{
    private const string OpenSql =
        "INSERT INTO trades (mint, prediction_id, opened_at, entry_price, size, strategy, confidence, features_snapshot, session_id, status) " +
        "VALUES (@mint, @prediction, @opened, @entry, @size, @strategy, @confidence, @features, @session, 'open') " +
        "RETURNING id";

    // Only an open row is closed, so a repeated or unknown close matches nothing (as InMemoryTradeStore rejects it).
    private const string CloseSql =
        "UPDATE trades SET closed_at = @closed, exit_price = @exit, duration_sec = @duration, " +
        "pnl = @pnl, pnl_pct = @pnlpct, exit_reason = @reason, status = 'closed' " +
        "WHERE id = @id AND status = 'open'";

    private const string CountClosedSql =
        "SELECT count(*) FROM trades WHERE status = 'closed' AND (@since::timestamptz IS NULL OR closed_at > @since)";

    public async Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(opening);
        double confidence = opening.Prediction.Confidence;
        if (!double.IsFinite(confidence) || confidence is < 0.0 or > 100.0)
        {
            throw new ArgumentOutOfRangeException(nameof(opening), confidence, "Confidence must be within [0, 100].");
        }

        long? session = sessions is null ? null : await sessions.CurrentIdAsync(cancellationToken);

        await using NpgsqlCommand cmd = dataSource.CreateCommand(OpenSql);
        cmd.Parameters.Add(Param("mint", NpgsqlDbType.Text, opening.Mint));
        cmd.Parameters.Add(Param("prediction", NpgsqlDbType.Bigint, opening.Prediction.PredictionId));
        cmd.Parameters.Add(Param("opened", NpgsqlDbType.TimestampTz, opening.OpenedAt));
        cmd.Parameters.Add(Param("entry", NpgsqlDbType.Numeric, opening.EntryPrice));
        cmd.Parameters.Add(Param("size", NpgsqlDbType.Numeric, opening.Size));
        cmd.Parameters.Add(Param("strategy", NpgsqlDbType.Text, opening.Strategy.ToCode()));
        cmd.Parameters.Add(Param("confidence", NpgsqlDbType.Double, confidence));
        cmd.Parameters.Add(Param("features", NpgsqlDbType.Jsonb, FeaturesJson(opening.Prediction.Features)));
        cmd.Parameters.Add(Param("session", NpgsqlDbType.Bigint, session));

        try
        {
            return (long)(await cmd.ExecuteScalarAsync(cancellationToken))!;
        }
        catch (PostgresException e) when (e.SqlState == PostgresErrorCodes.ForeignKeyViolation)
        {
            throw new InvalidOperationException(
                $"Cannot record a trade on {opening.Mint}: unknown mint or prediction id {opening.Prediction.PredictionId}.", e);
        }
    }

    public async Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(trade);
        // pnl_pct is DOUBLE PRECISION: Postgres would accept NaN, which would poison every aggregate over it.
        if (!double.IsFinite(trade.PnlPct))
        {
            throw new ArgumentOutOfRangeException(nameof(trade), trade.PnlPct, "PnlPct must be finite.");
        }

        await using NpgsqlCommand cmd = dataSource.CreateCommand(CloseSql);
        cmd.Parameters.Add(Param("id", NpgsqlDbType.Bigint, trade.TradeId));
        cmd.Parameters.Add(Param("closed", NpgsqlDbType.TimestampTz, trade.ClosedAt));
        cmd.Parameters.Add(Param("exit", NpgsqlDbType.Numeric, trade.ExitPrice));
        cmd.Parameters.Add(Param("duration", NpgsqlDbType.Integer, trade.DurationSec));
        cmd.Parameters.Add(Param("pnl", NpgsqlDbType.Numeric, trade.Pnl));
        cmd.Parameters.Add(Param("pnlpct", NpgsqlDbType.Double, trade.PnlPct));
        cmd.Parameters.Add(Param("reason", NpgsqlDbType.Text, trade.Reason.ToCode()));

        if (await cmd.ExecuteNonQueryAsync(cancellationToken) == 0)
        {
            throw new InvalidOperationException($"Trade {trade.TradeId} is not open.");
        }
    }

    public async Task<long> CountClosedSinceAsync(DateTimeOffset? since, CancellationToken cancellationToken = default)
    {
        await using NpgsqlCommand cmd = dataSource.CreateCommand(CountClosedSql);
        cmd.Parameters.Add(Param("since", NpgsqlDbType.TimestampTz, since));
        return (long)(await cmd.ExecuteScalarAsync(cancellationToken))!;
    }

    /// <summary>
    /// The prediction's feature vector as a JSON object. JSON has no NaN/Infinity, so a non-finite value
    /// is stored as null — the same "not derivable" the model already sees for a missing feature.
    /// </summary>
    private static string FeaturesJson(IReadOnlyDictionary<string, double?> features) =>
        JsonSerializer.Serialize(features.ToDictionary(
            kv => kv.Key,
            kv => kv.Value is double v && double.IsFinite(v) ? kv.Value : null));
}
