using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Logging;

namespace Solia.Api.Logging;

/// <summary>Wires the <see cref="PostgresLoggerProvider"/> into the host's logging.</summary>
public static class PostgresLoggingBuilderExtensions
{
    /// <summary>
    /// Adds the Postgres `logs`-table logger provider, resolving its <see cref="ILogSink"/> from the
    /// container (the caller registers one — see <c>AddBackendLogSink</c>). Registered with
    /// <c>TryAddEnumerable</c> so it's added at most once.
    /// </summary>
    public static ILoggingBuilder AddPostgresLogSink(this ILoggingBuilder builder)
    {
        builder.Services.TryAddEnumerable(ServiceDescriptor.Singleton<ILoggerProvider, PostgresLoggerProvider>());
        return builder;
    }
}
