using System.Text.Json;
using Npgsql;
using NpgsqlTypes;

namespace Solia.DataPipeline.Logging;

/// <summary>
/// Writes log rows to the `logs` table with service = 'data-pipeline'. Context, if any, is
/// stored as JSONB.
/// </summary>
public sealed class PostgresLogSink(NpgsqlDataSource dataSource) : ILogSink
{
    private const string Sql =
        "INSERT INTO logs (service, level, message, context) VALUES ('data-pipeline', @level, @message, @context)";

    public async Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
    {
        await using NpgsqlCommand cmd = dataSource.CreateCommand(Sql);
        cmd.Parameters.AddWithValue("level", level.ToDbValue());
        cmd.Parameters.AddWithValue("message", message);
        cmd.Parameters.Add(new NpgsqlParameter("context", NpgsqlDbType.Jsonb)
        {
            Value = context is null ? DBNull.Value : JsonSerializer.Serialize(context),
        });
        await cmd.ExecuteNonQueryAsync(ct);
    }
}
