using Npgsql;
using NpgsqlTypes;
using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// <see cref="IWalletStateStore"/> on PostgreSQL (<c>wallet_state</c> + the single-row <c>wallet_state_meta</c>,
/// migration 0014), written with raw SQL like every pipeline table. Built for volume: a save streams the changed
/// wallets with a binary <c>COPY</c> into a transaction-scoped staging table and upserts them in one statement, and
/// a load reads the table back with a binary <c>COPY</c>, so neither costs a round trip per wallet. The reputation
/// (FE-T10, migration 0015) rides along as parallel arrays: round trips (first-buy time, P&amp;L) and launch outcomes
/// (launch time, flags <see cref="RuggedFlag"/> / <see cref="PumpedFlag"/>).
/// </summary>
public sealed class PostgresWalletStateStore(NpgsqlDataSource dataSource) : IWalletStateStore
{
    internal const short RuggedFlag = 1;
    internal const short PumpedFlag = 2;

    private const string Columns =
        "wallet, first_seen_at, last_seen_at, trade_count, tokens_traded, tokens_created, recent_launches, " +
        "trip_opened, trip_pnl_sol, outcome_launches, outcome_flags";

    private const string UpsertSql =
        "INSERT INTO wallet_state (" + Columns + ") SELECT " + Columns + " FROM wallet_state_stage " +
        "ON CONFLICT (wallet) DO UPDATE SET first_seen_at = EXCLUDED.first_seen_at, last_seen_at = EXCLUDED.last_seen_at, " +
        "trade_count = EXCLUDED.trade_count, tokens_traded = EXCLUDED.tokens_traded, " +
        "tokens_created = EXCLUDED.tokens_created, recent_launches = EXCLUDED.recent_launches, " +
        "trip_opened = EXCLUDED.trip_opened, trip_pnl_sol = EXCLUDED.trip_pnl_sol, " +
        "outcome_launches = EXCLUDED.outcome_launches, outcome_flags = EXCLUDED.outcome_flags";

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
                DateTimeOffset[] tripOpened = await export.ReadAsync<DateTimeOffset[]>(
                    NpgsqlDbType.Array | NpgsqlDbType.TimestampTz, cancellationToken);
                double[] tripPnl = await export.ReadAsync<double[]>(NpgsqlDbType.Array | NpgsqlDbType.Double, cancellationToken);
                DateTimeOffset[] outcomeLaunches = await export.ReadAsync<DateTimeOffset[]>(
                    NpgsqlDbType.Array | NpgsqlDbType.TimestampTz, cancellationToken);
                short[] outcomeFlags = await export.ReadAsync<short[]>(NpgsqlDbType.Array | NpgsqlDbType.Smallint, cancellationToken);
                wallets.Add(new WalletRecord(wallet, first.ToUnixTimeSeconds(), seen.ToUnixTimeSeconds(), trades, tokens, created,
                    launches.Select(l => l.ToUnixTimeSeconds()).ToArray(),
                    ToTrips(tripOpened, tripPnl), ToOutcomes(outcomeLaunches, outcomeFlags)));
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
                    RoundTrip[] trips = w.Trips ?? [];
                    LaunchOutcome[] outcomes = w.Outcomes ?? [];
                    await import.WriteAsync(trips.Select(t => Utc(t.Opened)).ToArray(),
                        NpgsqlDbType.Array | NpgsqlDbType.TimestampTz, cancellationToken);
                    await import.WriteAsync(trips.Select(t => t.PnlSol).ToArray(),
                        NpgsqlDbType.Array | NpgsqlDbType.Double, cancellationToken);
                    await import.WriteAsync(outcomes.Select(o => Utc(o.Launch)).ToArray(),
                        NpgsqlDbType.Array | NpgsqlDbType.TimestampTz, cancellationToken);
                    await import.WriteAsync(outcomes.Select(Flags).ToArray(),
                        NpgsqlDbType.Array | NpgsqlDbType.Smallint, cancellationToken);
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

    internal static short Flags(LaunchOutcome o) => (short)((o.Rugged ? RuggedFlag : 0) | (o.Pumped ? PumpedFlag : 0));

    /// <summary>Pairs the parallel arrays back (a length mismatch keeps the common prefix); null when empty.</summary>
    internal static RoundTrip[]? ToTrips(DateTimeOffset[] opened, double[] pnl)
    {
        int n = Math.Min(opened.Length, pnl.Length);
        return n == 0 ? null : Enumerable.Range(0, n).Select(i => new RoundTrip(opened[i].ToUnixTimeSeconds(), pnl[i])).ToArray();
    }

    internal static LaunchOutcome[]? ToOutcomes(DateTimeOffset[] launches, short[] flags)
    {
        int n = Math.Min(launches.Length, flags.Length);
        return n == 0
            ? null
            : Enumerable.Range(0, n)
                .Select(i => new LaunchOutcome(launches[i].ToUnixTimeSeconds(), (flags[i] & RuggedFlag) != 0, (flags[i] & PumpedFlag) != 0))
                .ToArray();
    }
}
