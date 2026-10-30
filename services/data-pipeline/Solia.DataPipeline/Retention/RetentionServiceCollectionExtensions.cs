using Microsoft.Extensions.DependencyInjection.Extensions;

namespace Solia.DataPipeline.Retention;

public static class RetentionServiceCollectionExtensions
{
    /// <summary>
    /// The purge job over the registered <c>NpgsqlDataSource</c> and log sink (options are bound by
    /// <c>AddPipelineOptions</c>); <paramref name="hosted"/> adds the periodic <see cref="RetentionService"/>.
    /// </summary>
    public static IServiceCollection AddRetention(this IServiceCollection services, bool hosted)
    {
        services.TryAddSingleton(TimeProvider.System);
        services.AddSingleton<IRetentionStore, PostgresRetentionStore>();
        services.AddSingleton<RetentionJob>();
        if (hosted)
        {
            services.AddHostedService<RetentionService>();
        }

        return services;
    }
}
