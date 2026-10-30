using Npgsql;
using NpgsqlTypes;
using static Solia.Shared.Data.NpgsqlParameters;

namespace Solia.DataPipeline.Retention;

/// <summary>
/// Raw-SQL purge. Each batch deletes the oldest expired rows by id through the <c>ts</c> index, in its own
/// transaction, so no lock is held for long; <c>ON DELETE SET NULL</c> clears the references from
/// <c>predictions</c> / <c>trades</c>.
/// </summary>
public sealed class PostgresRetentionStore(NpgsqlDataSource dataSource) : IRetentionStore
{
    public async Task<PurgeCount> CountExpiredAsync(RetainedTable table, DateTimeOffset cutoff, CancellationToken ct)
    {
        await using NpgsqlCommand cmd = dataSource.CreateCommand(
            $"SELECT count(*), min(ts), max(ts) FROM {table.SqlName()} WHERE ts < @cutoff");
        cmd.Parameters.Add(Param("cutoff", NpgsqlDbType.TimestampTz, cutoff));
        return await ReadCountAsync(cmd, ct);
    }

    public async Task<PurgeCount> DeleteExpiredBatchAsync(
        RetainedTable table, DateTimeOffset cutoff, int batchSize, CancellationToken ct)
    {
        string name = table.SqlName();
        await using NpgsqlCommand cmd = dataSource.CreateCommand(
            $"""
             WITH doomed AS (SELECT id FROM {name} WHERE ts < @cutoff ORDER BY ts LIMIT @batch),
                  deleted AS (DELETE FROM {name} t USING doomed d WHERE t.id = d.id RETURNING t.ts)
             SELECT count(*), min(ts), max(ts) FROM deleted
             """);
        cmd.Parameters.Add(Param("cutoff", NpgsqlDbType.TimestampTz, cutoff));
        cmd.Parameters.Add(Param("batch", NpgsqlDbType.Integer, batchSize));
        return await ReadCountAsync(cmd, ct);
    }

    public async Task VacuumAsync(RetainedTable table, CancellationToken ct)
    {
        // PARALLEL 0: parallel index vacuuming needs dynamic shared memory, and Docker's default 64 MB /dev/shm
        // makes it fail ("could not resize shared memory segment") on a large table. Unbounded: on a big table
        // it takes a while, but holds no lock that blocks reads or writes.
        await using NpgsqlCommand cmd = dataSource.CreateCommand($"VACUUM (ANALYZE, PARALLEL 0) {table.SqlName()}");
        cmd.CommandTimeout = 0;
        await cmd.ExecuteNonQueryAsync(ct);
    }

    private static async Task<PurgeCount> ReadCountAsync(NpgsqlCommand cmd, CancellationToken ct)
    {
        await using NpgsqlDataReader reader = await cmd.ExecuteReaderAsync(ct);
        await reader.ReadAsync(ct);
        return new PurgeCount(
            reader.GetInt64(0),
            reader.IsDBNull(1) ? null : reader.GetFieldValue<DateTimeOffset>(1),
            reader.IsDBNull(2) ? null : reader.GetFieldValue<DateTimeOffset>(2));
    }
}
