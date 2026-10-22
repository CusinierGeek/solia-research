using System.Text.Json;
using Npgsql;
using NpgsqlTypes;
using static Solia.Shared.Data.NpgsqlParameters;

namespace Solia.Shared.Logging;

/// <summary>
/// Writes log rows to the shared `logs` table for one <paramref name="service"/> ('data-pipeline',
/// 'bot', …) — the single owner of that table's INSERT contract. Context, if any, is stored as JSONB
/// with camelCase keys (<see cref="SerializeContext"/>), so every row and service can be queried with
/// the same key spelling.
/// </summary>
public class PostgresLogSink : ILogSink
{
    private const string Sql =
        "INSERT INTO logs (service, level, message, context) VALUES (@service, @level, @message, @context)";

    private static readonly JsonSerializerOptions ContextJson = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    private readonly NpgsqlDataSource _dataSource;

    public PostgresLogSink(NpgsqlDataSource dataSource, string service)
    {
        ArgumentNullException.ThrowIfNull(dataSource);
        ArgumentException.ThrowIfNullOrWhiteSpace(service);

        _dataSource = dataSource;
        Service = service;
    }

    /// <summary>The `logs.service` value every row from this sink carries.</summary>
    public string Service { get; }

    /// <summary>The JSONB text stored for <paramref name="context"/>: camelCase keys, or null.</summary>
    public static string? SerializeContext(object? context) =>
        context is null ? null : JsonSerializer.Serialize(context, ContextJson);

    public async Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
    {
        await using NpgsqlCommand cmd = _dataSource.CreateCommand(Sql);
        cmd.Parameters.Add(Param("service", NpgsqlDbType.Text, Service));
        cmd.Parameters.Add(Param("level", NpgsqlDbType.Text, level.ToDbValue()));
        cmd.Parameters.Add(Param("message", NpgsqlDbType.Text, message));
        cmd.Parameters.Add(Param("context", NpgsqlDbType.Jsonb, SerializeContext(context)));
        await cmd.ExecuteNonQueryAsync(ct);
    }
}
