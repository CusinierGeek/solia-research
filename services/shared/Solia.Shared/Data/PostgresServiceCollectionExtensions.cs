using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Npgsql;
using Solia.Shared.Configuration;
using Solia.Shared.Logging;

namespace Solia.Shared.Data;

/// <summary>
/// The one place every service registers its pooled <see cref="NpgsqlDataSource"/> and its `logs`-table
/// <see cref="ILogSink"/>, so the wiring (connection resolution, pooling, fallbacks) can't drift between
/// the data-pipeline, the backend and the bot.
/// </summary>
public static class PostgresServiceCollectionExtensions
{
    /// <summary>
    /// Registers a single pooled <see cref="NpgsqlDataSource"/>, owned (and disposed) by the container. The
    /// connection string is resolved via <see cref="DbConnectionSettings"/> (SOLIA_DB_CONNECTION, else
    /// POSTGRES_*) on first use, so configuration sources added after this call still apply. Uses
    /// <c>TryAdd</c>, so a data source the host (or a test) registered first wins.
    /// </summary>
    public static IServiceCollection AddSoliaDataSource(this IServiceCollection services, IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);
        services.TryAddSingleton(_ => NpgsqlDataSource.Create(DbConnectionSettings.Resolve(key => configuration[key])));
        return services;
    }

    /// <summary>
    /// Registers the <see cref="PostgresLogSink"/> for <paramref name="service"/> (the `logs.service` value)
    /// on the shared data source. The sink is best-effort: if the connection string is rejected (malformed,
    /// an <see cref="ArgumentException"/>) it falls back to <see cref="NullLogSink"/> and says so on stderr,
    /// rather than failing the host over an optional log mirror. Any other failure building the data source
    /// (e.g. secrets that can't be decrypted) is not a log-mirror problem and propagates, so the host fails
    /// on its real cause. Uses <c>TryAdd</c>.
    /// </summary>
    public static IServiceCollection AddPostgresLogSink(
        this IServiceCollection services, IConfiguration configuration, string service)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(service);
        services.AddSoliaDataSource(configuration);
        services.TryAddSingleton(sp => CreateLogSink(sp, service));
        return services;
    }

    private static ILogSink CreateLogSink(IServiceProvider services, string service)
    {
        try
        {
            return new PostgresLogSink(services.GetRequiredService<NpgsqlDataSource>(), service);
        }
        catch (ArgumentException ex)
        {
            // Not through ILogger (the sink is being built for the logging pipeline itself), so not redacted
            // either: name the rejected setting, never echo the message, which can quote the connection string.
            Console.Error.WriteLine(DisabledMessage(service, ex));
            return NullLogSink.Instance;
        }
    }

    /// <summary>The stderr line for a rejected connection string: the setting it names, never its text.</summary>
    public static string DisabledMessage(string service, ArgumentException ex) =>
        $"[{service}] `logs` table sink disabled: the database connection string was rejected " +
        $"({ex.GetType().Name}{(ex.ParamName is null ? "" : $", setting '{ex.ParamName}'")}). " +
        "Check SOLIA_DB_CONNECTION / POSTGRES_* (or the db_connection / db_password secrets).";
}
