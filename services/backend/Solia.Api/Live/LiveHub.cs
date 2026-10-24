using Microsoft.AspNetCore.SignalR;

namespace Solia.Api.Live;

/// <summary>
/// The dashboard's live feed (<see cref="LiveEvents.Path"/>): server → client only, so the hub has no
/// methods — clients just connect and handle the <see cref="LiveEvents"/>. Single-user internal network: no
/// auth and no CORS policy (docs/API-CONTRACTS.md, "Live hub").
/// </summary>
public sealed class LiveHub : Hub;

/// <summary>Pushes one event to every connected client.</summary>
public interface ILiveBroadcaster
{
    Task BroadcastAsync(string eventName, object payload, CancellationToken cancellationToken);
}

/// <summary>Broadcasts through the <see cref="LiveHub"/>'s <see cref="IHubContext{THub}"/>.</summary>
internal sealed class HubLiveBroadcaster(IHubContext<LiveHub> hub) : ILiveBroadcaster
{
    public Task BroadcastAsync(string eventName, object payload, CancellationToken cancellationToken) =>
        hub.Clients.All.SendAsync(eventName, payload, cancellationToken);
}
