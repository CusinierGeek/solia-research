using Npgsql;
using Solia.Shared.Logging;

namespace Solia.Bot.Logging;

/// <summary>
/// The shared <see cref="PostgresLogSink"/> with service = 'bot'; reuses the <see cref="NpgsqlDataSource"/>
/// the host already registered.
/// </summary>
public sealed class PostgresBotLogSink(NpgsqlDataSource dataSource) : PostgresLogSink(dataSource, ServiceName), IBotLogSink
{
    public const string ServiceName = "bot";
}
