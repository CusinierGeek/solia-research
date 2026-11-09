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

    public async Task<BandBatch> PurgeEstablishedBandBatchAsync(
        FeaturesCursor after, DateTimeOffset cutoff, EstablishedRule rule, int batchSize, bool dryRun, CancellationToken ct)
    {
        // Whether a token is kept is decided once per mint of the batch, by an index-only probe of the partial
        // features_established_idx (0018): the age threshold is spliced as a literal (a validated int) so the planner
        // can prove it implies the index predicate, which a parameter would not. The thinning probe ("an earlier row
        // of this mint in the same bucket") is index-only on features_mint_ts_key; the first row of a bucket is never
        // removed by it, so each bucket keeps exactly one row.
        long ageSec = (long)rule.Age.TotalSeconds;
        string thinning = rule.Sample > TimeSpan.Zero
            ? """
               OR (c.token_age_sec >= @full AND EXISTS (
                       SELECT 1 FROM features y
                       WHERE y.mint = c.mint AND y.ts >= date_bin(@sample, c.ts, timestamptz 'epoch') AND y.ts < c.ts))
              """
            : "";
        string removed = dryRun
            ? "removed AS (SELECT ts FROM doomed)"
            : "removed AS (DELETE FROM features t USING doomed d WHERE t.id = d.id RETURNING t.ts)";
        await using NpgsqlCommand cmd = dataSource.CreateCommand(
            $"""
             WITH cand AS (
                     SELECT id, mint, ts, token_age_sec FROM features
                     WHERE ts >= @after_ts AND ts < @cutoff AND (ts, id) > (@after_ts, @after_id)
                     ORDER BY ts, id LIMIT @batch),
                  span AS (SELECT mint, min(ts) AS lo, max(ts) AS hi FROM cand GROUP BY mint),
                  kept AS (
                     SELECT s.mint FROM span s
                     WHERE EXISTS (SELECT 1 FROM features x
                                   WHERE x.mint = s.mint AND x.ts >= s.lo AND x.ts <= s.hi + @age
                                     AND x.token_age_sec >= {ageSec})),
                  doomed AS (
                     SELECT c.id, c.ts FROM cand c
                     WHERE NOT EXISTS (SELECT 1 FROM kept k WHERE k.mint = c.mint){thinning}),
                  last AS (SELECT ts, id FROM cand ORDER BY ts DESC, id DESC LIMIT 1),
                  {removed}
             SELECT (SELECT count(*) FROM cand), (SELECT ts FROM last), (SELECT id FROM last),
                    count(*), min(ts), max(ts)
             FROM removed
             """);
        cmd.Parameters.Add(Param("after_ts", NpgsqlDbType.TimestampTz, after.Ts));
        cmd.Parameters.Add(Param("after_id", NpgsqlDbType.Bigint, after.Id));
        cmd.Parameters.Add(Param("cutoff", NpgsqlDbType.TimestampTz, cutoff));
        cmd.Parameters.Add(Param("batch", NpgsqlDbType.Integer, batchSize));
        cmd.Parameters.Add(Param("age", NpgsqlDbType.Interval, rule.Age));
        if (rule.Sample > TimeSpan.Zero)
        {
            cmd.Parameters.Add(Param("full", NpgsqlDbType.Integer, (int)rule.FullResolution.TotalSeconds));
            cmd.Parameters.Add(Param("sample", NpgsqlDbType.Interval, rule.Sample));
        }

        await using NpgsqlDataReader reader = await cmd.ExecuteReaderAsync(ct);
        await reader.ReadAsync(ct);
        long scanned = reader.GetInt64(0);
        FeaturesCursor? last = reader.IsDBNull(1)
            ? null
            : new FeaturesCursor(reader.GetFieldValue<DateTimeOffset>(1), reader.GetInt64(2));
        var purge = new PurgeCount(
            reader.GetInt64(3),
            reader.IsDBNull(4) ? null : reader.GetFieldValue<DateTimeOffset>(4),
            reader.IsDBNull(5) ? null : reader.GetFieldValue<DateTimeOffset>(5));
        return new BandBatch(scanned, last, purge);
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
