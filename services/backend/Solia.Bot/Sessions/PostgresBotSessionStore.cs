using Npgsql;
using NpgsqlTypes;
using static Solia.Shared.Data.NpgsqlParameters;

namespace Solia.Bot.Sessions;

/// <summary>
/// Keeps bot sessions in <c>bot_sessions</c> (migration 0012) with raw SQL, in one transaction: the current session
/// is locked, the champion read from <c>models</c>, and the decision (continue / adopt / new) written before commit.
/// A concurrent second bot can't open a second current session: the partial unique index rejects it.
/// </summary>
public sealed class PostgresBotSessionStore(NpgsqlDataSource dataSource) : IBotSessionStore
{
    private const string CurrentSql =
        "SELECT id, config_hash, model_id, resumed_after_trade_id FROM bot_sessions WHERE ended_at IS NULL FOR UPDATE";

    private const string ChampionSql = "SELECT id, version FROM models WHERE is_champion LIMIT 1";

    private const string LastTradeSql = "SELECT COALESCE(max(id), 0) FROM trades";

    private const string ResumeSql = "UPDATE bot_sessions SET resumed_after_trade_id = @resumed WHERE id = @id";

    private const string AdoptSql =
        "UPDATE bot_sessions SET config_hash = @hash, options = @options, model_id = @model, model_version = @version, " +
        "resumed_after_trade_id = @resumed WHERE id = @id";

    private const string EndSql = "UPDATE bot_sessions SET ended_at = GREATEST(now(), started_at) WHERE id = @id";

    private const string InsertSql =
        "INSERT INTO bot_sessions (config_hash, options, model_id, model_version, label, resumed_after_trade_id) " +
        "VALUES (@hash, @options, @model, @version, @label, @resumed) RETURNING id";

    public async Task<BotSession> BeginAsync(BotSessionRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(request);
        await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(cancellationToken);
        await using NpgsqlTransaction tx = await connection.BeginTransactionAsync(cancellationToken);

        (CurrentBotSession Session, long Resumed)? current = null;
        await using (var cmd = new NpgsqlCommand(CurrentSql, connection, tx))
        await using (NpgsqlDataReader row = await cmd.ExecuteReaderAsync(cancellationToken))
        {
            if (await row.ReadAsync(cancellationToken))
            {
                current = (new CurrentBotSession(row.GetInt64(0), row.IsDBNull(1) ? null : row.GetString(1),
                    row.IsDBNull(2) ? null : row.GetInt64(2)), row.GetInt64(3));
            }
        }

        long? championId = null;
        string? championVersion = null;
        await using (var cmd = new NpgsqlCommand(ChampionSql, connection, tx))
        await using (NpgsqlDataReader row = await cmd.ExecuteReaderAsync(cancellationToken))
        {
            if (await row.ReadAsync(cancellationToken))
            {
                championId = row.GetInt64(0);
                championVersion = row.GetString(1);
            }
        }

        string hash = BotSessionFingerprint.Hash(request.OptionsJson, championVersion);
        BotSessionDecision decision = BotSessionRules.Decide(current?.Session, hash, championId);

        // A fresh run orphans everything still open; a re-check keeps the cut-off the run started with.
        long resumed = current?.Resumed ?? 0;
        if (request.NewRun)
        {
            await using var last = new NpgsqlCommand(LastTradeSql, connection, tx);
            resumed = (long)(await last.ExecuteScalarAsync(cancellationToken))!;
        }

        long id;
        switch (decision)
        {
            case BotSessionDecision.Continue:
                id = current!.Value.Session.Id;
                await ExecAsync(ResumeSql, connection, tx, cancellationToken,
                    Param("resumed", NpgsqlDbType.Bigint, resumed), Param("id", NpgsqlDbType.Bigint, id));
                break;
            case BotSessionDecision.Adopt:
                id = current!.Value.Session.Id;
                await ExecAsync(AdoptSql, connection, tx, cancellationToken,
                    Param("hash", NpgsqlDbType.Text, hash), Param("options", NpgsqlDbType.Jsonb, request.OptionsJson),
                    Param("model", NpgsqlDbType.Bigint, championId), Param("version", NpgsqlDbType.Text, championVersion),
                    Param("resumed", NpgsqlDbType.Bigint, resumed), Param("id", NpgsqlDbType.Bigint, id));
                break;
            default:
                if (current is { } ended)
                {
                    await ExecAsync(EndSql, connection, tx, cancellationToken, Param("id", NpgsqlDbType.Bigint, ended.Session.Id));
                }

                await using (var insert = new NpgsqlCommand(InsertSql, connection, tx))
                {
                    insert.Parameters.Add(Param("hash", NpgsqlDbType.Text, hash));
                    insert.Parameters.Add(Param("options", NpgsqlDbType.Jsonb, request.OptionsJson));
                    insert.Parameters.Add(Param("model", NpgsqlDbType.Bigint, championId));
                    insert.Parameters.Add(Param("version", NpgsqlDbType.Text, championVersion));
                    insert.Parameters.Add(Param("label", NpgsqlDbType.Text, string.IsNullOrWhiteSpace(request.Label) ? null : request.Label.Trim()));
                    insert.Parameters.Add(Param("resumed", NpgsqlDbType.Bigint, resumed));
                    id = (long)(await insert.ExecuteScalarAsync(cancellationToken))!;
                }

                break;
        }

        await tx.CommitAsync(cancellationToken);
        return new BotSession(id, decision, championId, championVersion, hash);
    }

    private static async Task ExecAsync(string sql, NpgsqlConnection connection, NpgsqlTransaction tx,
        CancellationToken cancellationToken, params NpgsqlParameter[] parameters)
    {
        await using var cmd = new NpgsqlCommand(sql, connection, tx);
        cmd.Parameters.AddRange(parameters);
        await cmd.ExecuteNonQueryAsync(cancellationToken);
    }
}
