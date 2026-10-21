using Npgsql;

namespace Solia.Shared.Configuration;

/// <summary>
/// The project-wide database connection contract (see .env.example): SOLIA_DB_CONNECTION
/// wins, else the POSTGRES_* parts, whose defaults suit local dev against the compose
/// Postgres. Every service resolves its connection string here so they can't drift apart.
/// </summary>
public static class DbConnectionSettings
{
    public const string ConnectionKey = "SOLIA_DB_CONNECTION";

    /// <param name="lookup">
    /// Reads a flat key, e.g. <c>Environment.GetEnvironmentVariable</c> or <c>key =&gt; configuration[key]</c>.
    /// </param>
    public static string Resolve(Func<string, string?> lookup)
    {
        string? explicitConn = lookup(ConnectionKey);
        if (!string.IsNullOrWhiteSpace(explicitConn))
        {
            return explicitConn.Trim();
        }

        var builder = new NpgsqlConnectionStringBuilder
        {
            Host = FlatValues.Get(lookup, "POSTGRES_HOST", "localhost"),
            Port = FlatValues.GetPort(lookup, "POSTGRES_PORT", 5432),
            Database = FlatValues.Get(lookup, "POSTGRES_DB", "solia"),
            Username = FlatValues.Get(lookup, "POSTGRES_USER", "solia"),
            Password = FlatValues.Get(lookup, "POSTGRES_PASSWORD", "change_me"),
        };
        return builder.ConnectionString;
    }
}
