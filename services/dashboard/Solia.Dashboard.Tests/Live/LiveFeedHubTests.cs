using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.Dashboard.Api;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

/// <summary>
/// The real <see cref="HubLiveConnection"/> against an in-memory SignalR hub at <c>/hubs/live</c> that pushes the
/// contract's raw snake_case JSON — the wire shape the backend sends.
/// </summary>
public class LiveFeedHubTests : IAsyncLifetime
{
    private static readonly TimeSpan Timeout = TimeSpan.FromSeconds(10);

    private const string TradeJson = """
        { "id": 7, "mint": "So1abc", "prediction_id": 42, "opened_at": "2026-10-26T12:00:00Z",
          "closed_at": null, "entry_price": 0.0000001234, "exit_price": null, "size": 0.5, "duration_sec": null,
          "pnl": null, "pnl_pct": null, "strategy": "buy", "confidence": 74.0, "exit_reason": null, "status": "open" }
        """;

    private WebApplication _server = null!;
    private LiveFeed _feed = null!;

    public async Task InitializeAsync()
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder();
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Services.AddSignalR();
        _server = builder.Build();
        _server.MapHub<TestHub>("/hubs/live");
        await _server.StartAsync();

        TestServer testServer = _server.GetTestServer();
        HubLiveConnection connection = HubLiveConnection.Create(
            LiveFeedServiceCollectionExtensions.HubUrl("http://backend"),
            LiveFeedRetryPolicy.Instance,
            NullLoggerFactory.Instance,
            http =>
            {
                http.HttpMessageHandlerFactory = _ => testServer.CreateHandler();
                http.Transports = HttpTransportType.LongPolling;
            });
        _feed = new LiveFeed(connection, LiveFeedRetryPolicy.Instance, NullLogger<LiveFeed>.Instance);
    }

    public async Task DisposeAsync()
    {
        await _feed.DisposeAsync();
        await _server.DisposeAsync();
    }

    [Fact]
    public async Task Connects_and_delivers_snake_case_payloads_as_typed_dtos()
    {
        TaskCompletionSource<TradeDto> received = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource resynced = new(TaskCreationOptions.RunContinuationsAsynchronously);
        _feed.Subscribe(LiveFeedEvents.TradeOpened, t => Task.FromResult(received.TrySetResult(t)));
        _feed.SubscribeResynced(() => Task.FromResult(resynced.TrySetResult()));

        await _feed.StartAsync(CancellationToken.None);
        await resynced.Task.WaitAsync(Timeout);
        Assert.Equal(LiveFeedState.Connected, _feed.State);

        IHubContext<TestHub> hub = _server.Services.GetRequiredService<IHubContext<TestHub>>();
        await hub.Clients.All.SendAsync("trade_opened", JsonDocument.Parse(TradeJson).RootElement);

        TradeDto trade = await received.Task.WaitAsync(Timeout);
        Assert.Equal((7L, "So1abc", 42L, 0.0000001234m, "open"), (trade.Id, trade.Mint, trade.PredictionId!.Value, trade.EntryPrice!.Value, trade.Status));
        Assert.Null(trade.ClosedAt);
    }

    [Fact]
    public async Task Stats_changed_signal_deserializes()
    {
        TaskCompletionSource<StatsChangedDto> received = new(TaskCreationOptions.RunContinuationsAsynchronously);
        TaskCompletionSource resynced = new(TaskCreationOptions.RunContinuationsAsynchronously);
        _feed.Subscribe(LiveFeedEvents.StatsChanged, s => Task.FromResult(received.TrySetResult(s)));
        _feed.SubscribeResynced(() => Task.FromResult(resynced.TrySetResult()));
        await _feed.StartAsync(CancellationToken.None);
        await resynced.Task.WaitAsync(Timeout);

        IHubContext<TestHub> hub = _server.Services.GetRequiredService<IHubContext<TestHub>>();
        await hub.Clients.All.SendAsync(
            "stats_changed", JsonDocument.Parse("""{ "trade_id": 7, "closed_at": "2026-10-26T12:01:30Z" }""").RootElement);

        Assert.Equal(new StatsChangedDto(7, DateTimeOffset.Parse("2026-10-26T12:01:30Z")), await received.Task.WaitAsync(Timeout));
    }

    public sealed class TestHub : Hub;
}
