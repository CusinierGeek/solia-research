using System.Text.Json;
using Microsoft.AspNetCore.Http.Json;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Options;
using Solia.Api.Logging;

namespace Solia.Api.Live;

/// <summary>Wires the SignalR live hub and its database feed (<see cref="LiveNotificationListener"/>).</summary>
public static class LiveHubServiceCollectionExtensions
{
    /// <summary>
    /// SignalR whose JSON payloads use a copy of the REST serializer options (<c>ConfigureHttpJsonOptions</c>), so
    /// an event and the matching REST item can't serialize differently; the Postgres LISTEN feed and the
    /// dispatcher. Needs the unpooled <see cref="Npgsql.NpgsqlDataSource"/> and the EF context, which
    /// <c>Program.cs</c> registers.
    /// </summary>
    public static IServiceCollection AddLiveHub(this IServiceCollection services)
    {
        services.AddSignalR().AddJsonProtocol();
        services.AddOptions<JsonHubProtocolOptions>().Configure<IOptions<JsonOptions>>((hub, http) =>
            hub.PayloadSerializerOptions = new JsonSerializerOptions(http.Value.SerializerOptions));

        services.AddSingleton<ILiveRowReader, DbLiveRowReader>();
        services.AddSingleton<ILiveBroadcaster, HubLiveBroadcaster>();
        services.AddSingleton<LiveEventDispatcher>();
        services.AddHostedService<LiveNotificationListener>();

        // The feed's own diagnostics stay out of the `logs` table: each row there is itself a live event, so a
        // failing log_written would otherwise log a warning, which raises another log_written, and so on. (What
        // other categories log while dispatching — EF, Npgsql — is kept out by the queue's suppression scope.)
        services.Configure<LoggerFilterOptions>(options =>
            options.AddFilter<PostgresLoggerProvider>(typeof(LiveHub).Namespace, LogLevel.None));
        return services;
    }

    /// <summary>Maps the <see cref="LiveHub"/> at <see cref="LiveEvents.Path"/>.</summary>
    public static IEndpointRouteBuilder MapLiveHub(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapHub<LiveHub>(LiveEvents.Path);
        return endpoints;
    }
}
