using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Shared.Data;

namespace Solia.Core.Data;

/// <summary>
/// Registers the <see cref="SoliaDbContext"/> over the shared pooled <see cref="NpgsqlDataSource"/>.
/// </summary>
public static class SoliaDbContextServiceCollectionExtensions
{
    /// <summary>
    /// Registers a pooled <see cref="SoliaDbContext"/> bound to the single shared
    /// <see cref="NpgsqlDataSource"/> (see <see cref="PostgresServiceCollectionExtensions.AddSoliaDataSource"/>),
    /// so the context and the raw-SQL writers (log sink, bot) never open a second connection pool. The
    /// connection string is resolved the project-wide way (SOLIA_DB_CONNECTION, else POSTGRES_*); the DB
    /// password moves to the secrets provider in T7. The context is read-only (no-tracking, writes rejected —
    /// see <see cref="SoliaDbContext"/>).
    /// </summary>
    public static IServiceCollection AddSoliaDbContext(this IServiceCollection services, IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        services.AddSoliaDataSource(configuration);
        services.AddDbContextPool<SoliaDbContext>((sp, options) => options
            .UseNpgsql(sp.GetRequiredService<NpgsqlDataSource>()));

        return services;
    }
}
