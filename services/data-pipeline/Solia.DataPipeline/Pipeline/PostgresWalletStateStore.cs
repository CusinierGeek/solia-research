using Npgsql;
using NpgsqlTypes;
using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// <see cref="IWalletStateStore"/> on PostgreSQL (<c>wallet_state</c> + the single-row <c>wallet_state_meta</c>,
/// migration 0014), written with raw SQL like every pipeline table. Built for volume: a save streams the changed
/// wallets with a binary <c>COPY</c> into a transaction-scoped staging table and upserts them in one statement, and
/// a load reads the table back with a binary <c>COPY</c>, so neither costs a round trip per wallet.
/// </summary>
public sealed class PostgresWalletStateStore(NpgsqlDataSource dataSource) : IWalletStateStore
{
    private const string Columns = "wallet, first_seen_at, last_seen_at, trade_count, tokens_traded, tokens_created, recent_launches";

    private const string UpsertSql =
        "INSERT INTO wallet_state (" + Columns + ") SELECT " + Columns + " FROM wallet_state_stage " +
        "ON CONFLICT (wallet) DO UPDATE SET first_seen_at = EXCLUDED.first_seen_at, last_seen_at = EXCLUDED.last_seen_at, " +
        "trade_count = EXCLUDED.trade_count, tokens_traded = EXCLUDED.tokens_traded, " +
        "tokens_created = EXCLUDED.tokens_created, recent_launches = EXCLUDED.recent_launches";

    private const string MetaSql =
        "INSERT INTO wallet_state_meta (id, history_start, last_event_at, saved_at) VALUES (TRUE, @start, @last, now()) " +
        "ON CONFLICT (id) DO UPDATE SET history_start = EXCLUDED.history_start, " +
        "last_event_at = EXCLUDED.last_event_at, saved_at = EXCLUDED.saved_at";

    public async Task<WalletStateSnapshot?> LoadAsync(CancellationToken cancellationToken)
    {
        await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(cancellationToken);
        DateTimeOffset start;
        DateTimeOffset last;
        await using (var meta = new NpgsqlCommand("SELECT history_start, last_event_at FROM wallet_state_meta", connection))
        await using (NpgsqlDataReader reader = await meta.ExecuteReaderAsync(cancellationToken))
        {
            if (!await reader.ReadAsync(cancellationToken))
            {
                return null;
            }

            start = reader.GetFieldValue<DateTimeOffset>(0);
            last = reader.GetFieldValue<DateTimeOffset>(1);
        }

        var wallets = new List<WalletRecord>();
        await using (NpgsqlBinaryExporter export = await connection.BeginBinaryExportAsync(
            $"COPY wallet_state ({Columns}) TO STDOUT (FORMAT BINARY)", cancellationToken))
        {
            while (await export.StartRowAsync(cancellationToken) != -1)
            {
                string wallet = await export.ReadAsync<string>(NpgsqlDbType.Text, cancellationToken);
                DateTimeOffset first = await export.ReadAsync<DateTimeOffset>(NpgsqlDbType.TimestampTz, cancellationToken);
                DateTimeOffset seen = await export.ReadAsync<DateTimeOffset>(NpgsqlDbType.TimestampTz, cancellationToken);
                int trades = await export.ReadAsync<int>(NpgsqlDbType.Integer, cancellationToken);
                int tokens = await export.ReadAsync<int>(NpgsqlDbType.Integer, cancellationToken);
                int created = await export.ReadAsync<int>(NpgsqlDbType.Integer, cancellationToken);
                DateTimeOffset[] launches = await export.ReadAsync<DateTimeOffset[]>(
                    NpgsqlDbType.Array | NpgsqlDbType.TimestampTz, cancellationToken);
                wallets.Add(new WalletRecord(wallet, first.ToUnixTimeSeconds(), seen.ToUnixTimeSeconds(), trades, tokens, created,
                    launches.Select(l => l.ToUnixTimeSeconds()).ToArray()));
            }
        }

        return new WalletStateSnapshot(start, last, wallets);
    }

    public async Task SaveAsync(WalletStateChanges changes, CancellationToken cancellationToken)
    {
        await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(cancellationToken);
        await using NpgsqlTransaction transaction = await connection.BeginTransactionAsync(cancellationToken);

        if (changes.Upserts.Count > 0)
        {
            await ExecuteAsync(connection, transaction,
                "CREATE TEMP TABLE wallet_state_stage (LIKE wallet_state INCLUDING DEFAULTS) ON COMMIT DROP", cancellationToken);
            await using (NpgsqlBinaryImporter import = await connection.BeginBinaryImportAsync(
                $"COPY wallet_state_stage ({Columns}) FROM STDIN (FORMAT BINARY)", cancellationToken))
            {
                foreach (WalletRecord w in changes.Upserts)
                {
                    await import.StartRowAsync(cancellationToken);
                    await import.WriteAsync(w.Wallet, NpgsqlDbType.Text, cancellationToken);
                    await import.WriteAsync(Utc(w.FirstSeen), NpgsqlDbType.TimestampTz, cancellationToken);
                    await import.WriteAsync(Utc(w.LastSeen), NpgsqlDbType.TimestampTz, cancellationToken);
                    await import.WriteAsync(w.TradeCount, NpgsqlDbType.Integer, cancellationToken);
                    await import.WriteAsync(w.TokensTraded, NpgsqlDbType.Integer, cancellationToken);
                    await import.WriteAsync(w.TokensCreated, NpgsqlDbType.Integer, cancellationToken);
                    await import.WriteAsync(w.RecentLaunches.Select(Utc).ToArray(),
                        NpgsqlDbType.Array | NpgsqlDbType.TimestampTz, cancellationToken);
                }

                await import.CompleteAsync(cancellationToken);
            }

            await ExecuteAsync(connection, transaction, UpsertSql, cancellationToken);
        }

        if (changes.Deletes.Count > 0)
        {
            await using var delete = new NpgsqlCommand("DELETE FROM wallet_state WHERE wallet = ANY(@wallets)", connection, transaction);
            delete.Parameters.Add(new NpgsqlParameter("wallets", NpgsqlDbType.Array | NpgsqlDbType.Text) { Value = changes.Deletes.ToArray() });
            await delete.ExecuteNonQueryAsync(cancellationToken);
        }

        await using (var meta = new NpgsqlCommand(MetaSql, connection, transaction))
        {
            meta.Parameters.Add(new NpgsqlParameter("start", NpgsqlDbType.TimestampTz) { Value = changes.HistoryStart });
            meta.Parameters.Add(new NpgsqlParameter("last", NpgsqlDbType.TimestampTz) { Value = changes.LastEvent });
            await meta.ExecuteNonQueryAsync(cancellationToken);
        }

        await transaction.CommitAsync(cancellationToken);
    }

    private static async Task ExecuteAsync(
        NpgsqlConnection connection, NpgsqlTransaction transaction, string sql, CancellationToken cancellationToken)
    {
        await using var command = new NpgsqlCommand(sql, connection, transaction);
        await command.ExecuteNonQueryAsync(cancellationToken);
    }

    private static DateTimeOffset Utc(long unixSeconds) => DateTimeOffset.FromUnixTimeSeconds(unixSeconds);
}
