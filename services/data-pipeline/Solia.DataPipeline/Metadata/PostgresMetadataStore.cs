using Npgsql;
using NpgsqlTypes;

namespace Solia.DataPipeline.Metadata;

/// <summary>Durable cache of the per-token metadata fetch results (<c>token_metadata</c>, migration 0016).</summary>
public interface IMetadataStore
{
    /// <summary>The results fetched in [<paramref name="since"/>, <paramref name="until"/>] (no upper bound when null).</summary>
    Task<IReadOnlyList<TokenMetadataRecord>> LoadAsync(DateTimeOffset since, DateTimeOffset? until, CancellationToken cancellationToken);

    /// <summary>Stores a result; a token that already has one keeps it (fetched at most once).</summary>
    Task SaveAsync(TokenMetadataRecord record, CancellationToken cancellationToken);
}

/// <summary><see cref="IMetadataStore"/> on PostgreSQL, raw SQL like every pipeline table. One row per token (~40 k a day).</summary>
public sealed class PostgresMetadataStore(NpgsqlDataSource dataSource) : IMetadataStore
{
    private const string LoadSql =
        "SELECT mint, fetched_at, status, has_website, has_twitter, has_telegram FROM token_metadata " +
        "WHERE fetched_at >= @since AND (@until::timestamptz IS NULL OR fetched_at <= @until)";

    private const string SaveSql =
        "INSERT INTO token_metadata (mint, fetched_at, status, has_website, has_twitter, has_telegram) " +
        "VALUES (@mint, @fetched, @status, @website, @twitter, @telegram) ON CONFLICT (mint) DO NOTHING";

    public async Task<IReadOnlyList<TokenMetadataRecord>> LoadAsync(
        DateTimeOffset since, DateTimeOffset? until, CancellationToken cancellationToken)
    {
        await using NpgsqlCommand cmd = dataSource.CreateCommand(LoadSql);
        cmd.Parameters.Add(new NpgsqlParameter("since", NpgsqlDbType.TimestampTz) { Value = since });
        cmd.Parameters.Add(new NpgsqlParameter("until", NpgsqlDbType.TimestampTz) { Value = until is { } u ? u : DBNull.Value });
        await using NpgsqlDataReader reader = await cmd.ExecuteReaderAsync(cancellationToken);
        var records = new List<TokenMetadataRecord>();
        while (await reader.ReadAsync(cancellationToken))
        {
            records.Add(new TokenMetadataRecord(
                reader.GetString(0),
                reader.GetFieldValue<DateTimeOffset>(1),
                (MetadataFetchStatus)reader.GetInt16(2),
                reader.GetBoolean(3),
                reader.GetBoolean(4),
                reader.GetBoolean(5)));
        }

        return records;
    }

    public async Task SaveAsync(TokenMetadataRecord record, CancellationToken cancellationToken)
    {
        await using NpgsqlCommand cmd = dataSource.CreateCommand(SaveSql);
        cmd.Parameters.Add(new NpgsqlParameter("mint", NpgsqlDbType.Text) { Value = record.Mint });
        cmd.Parameters.Add(new NpgsqlParameter("fetched", NpgsqlDbType.TimestampTz) { Value = record.FetchedAt });
        cmd.Parameters.Add(new NpgsqlParameter("status", NpgsqlDbType.Smallint) { Value = (short)record.Status });
        cmd.Parameters.Add(new NpgsqlParameter("website", NpgsqlDbType.Boolean) { Value = record.HasWebsite });
        cmd.Parameters.Add(new NpgsqlParameter("twitter", NpgsqlDbType.Boolean) { Value = record.HasTwitter });
        cmd.Parameters.Add(new NpgsqlParameter("telegram", NpgsqlDbType.Boolean) { Value = record.HasTelegram });
        await cmd.ExecuteNonQueryAsync(cancellationToken);
    }
}
