using System.Text.Json;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Npgsql;
using Solia.Api.Live;

namespace Solia.Api.Tests.Live;

/// <summary>
/// A real Postgres (migrations applied, so the NOTIFY triggers are in place) with one known token and model, the
/// real host pointed at it, and a SignalR client connected to <see cref="LiveEvents.Path"/> recording every event
/// it receives. The hub's broadcaster is wrapped so a test can make the next broadcasts fail.
/// </summary>
public sealed class LiveHubFixture : PostgresFixture, IAsyncLifetime
{
    public const string Mint = "live1";

    public const string ModelVersion = "v-live";

    private static readonly TimeSpan Patience = TimeSpan.FromSeconds(30);

    private readonly List<(string Event, JsonElement Payload)> _received = [];
    private WebApplicationFactory<Program> _factory = null!;
    private HubConnection _hub = null!;

    public HttpClient Client { get; private set; } = null!;

    public FlakyBroadcaster Broadcaster { get; private set; } = null!;

    async Task IAsyncLifetime.InitializeAsync()
    {
        await InitializeAsync();
        await using (NpgsqlCommand seed = DataSource.CreateCommand(
            $"INSERT INTO tokens_history (mint, symbol, first_seen_at) VALUES ('{Mint}', 'LIVE', now()); " +
            $"INSERT INTO models (version, path, is_champion) VALUES ('{ModelVersion}', 'models/live.pt', TRUE)"))
        {
            await seed.ExecuteNonQueryAsync();
        }

        _factory = TestApiFactory.Create("Production", [("SOLIA_DB_CONNECTION", ConnectionString)],
            builder => builder.ConfigureTestServices(services =>
                services.AddSingleton<ILiveBroadcaster>(sp =>
                    new FlakyBroadcaster(ActivatorUtilities.CreateInstance<HubLiveBroadcaster>(sp)))));
        Client = _factory.CreateClient();
        Broadcaster = (FlakyBroadcaster)_factory.Services.GetRequiredService<ILiveBroadcaster>();

        TestServer server = _factory.Server;
        _hub = new HubConnectionBuilder()
            .WithUrl(new Uri(server.BaseAddress, LiveEvents.Path), options =>
            {
                options.HttpMessageHandlerFactory = _ => server.CreateHandler();
                options.Transports = HttpTransportType.LongPolling;
            })
            .Build();
        foreach (string name in (string[])[.. LiveEvents.RowEvents, LiveEvents.StatsChanged])
        {
            _hub.On<JsonElement>(name, payload =>
            {
                lock (_received)
                {
                    _received.Add((name, payload.Clone()));
                }
            });
        }

        await _hub.StartAsync();

        // Rows written before the listener's LISTEN are never announced: wait for it.
        LiveNotificationListener listener = _factory.Services.GetServices<IHostedService>().OfType<LiveNotificationListener>().Single();
        await Eventually(() => listener.IsListening, "the live listener never started listening");
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        await _hub.DisposeAsync();
        Client.Dispose();
        await _factory.DisposeAsync();
        await DisposeAsync();
    }

    /// <summary>The first <paramref name="eventName"/> received whose <c>id_property</c> is <paramref name="id"/>.</summary>
    public async Task<JsonElement> NextAsync(string eventName, long id, string idProperty = "id")
    {
        JsonElement? found = null;
        await Eventually(() =>
        {
            lock (_received)
            {
                found = _received.Where(r => r.Event == eventName && HasId(r.Payload, idProperty, id))
                    .Select(r => (JsonElement?)r.Payload).FirstOrDefault();
            }

            return found is not null;
        }, $"no {eventName} for {idProperty} {id}");
        return found!.Value;
    }

    /// <summary>Every <paramref name="eventName"/> received so far whose <c>id_property</c> is <paramref name="id"/>.</summary>
    public int CountOf(string eventName, long id, string idProperty = "id")
    {
        lock (_received)
        {
            return _received.Count(r => r.Event == eventName && HasId(r.Payload, idProperty, id));
        }
    }

    // A payload without the property (stats_changed has trade_id, not id) just doesn't match.
    private static bool HasId(JsonElement payload, string idProperty, long id) =>
        payload.TryGetProperty(idProperty, out JsonElement value) && value.TryGetInt64(out long actual) && actual == id;

    public async Task<long> InsertAsync(string sql)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        return (long)(await cmd.ExecuteScalarAsync())!;
    }

    private static async Task Eventually(Func<bool> condition, string failure)
    {
        DateTime deadline = DateTime.UtcNow + Patience;
        while (!condition())
        {
            Assert.True(DateTime.UtcNow < deadline, failure);
            await Task.Delay(25);
        }
    }

    /// <summary>The real hub broadcaster, except that the next <see cref="FailNext"/> broadcasts throw.</summary>
    public sealed class FlakyBroadcaster(ILiveBroadcaster inner) : ILiveBroadcaster
    {
        private int _failNext;

        public int FailNext
        {
            get => Volatile.Read(ref _failNext);
            set => Volatile.Write(ref _failNext, value);
        }

        public Task BroadcastAsync(string eventName, object payload, CancellationToken cancellationToken)
        {
            // Takes one pending failure only if there is one, so a concurrent send can never wipe a fresh FailNext.
            int pending = Volatile.Read(ref _failNext);
            while (pending > 0)
            {
                int seen = Interlocked.CompareExchange(ref _failNext, pending - 1, pending);
                if (seen == pending)
                {
                    return Task.FromException(new IOException("simulated broadcast failure"));
                }

                pending = seen;
            }

            return inner.BroadcastAsync(eventName, payload, cancellationToken);
        }
    }
}
