using Npgsql;
using Solia.Api.Hosting;

namespace Solia.Api.Live;

/// <summary>
/// Feeds the live hub from the database: holds one connection that <c>LISTEN</c>s on
/// <see cref="LiveEvents.Channel"/>, where the 0004 triggers announce every new prediction, opened/closed trade
/// and notable log row, whichever process wrote it. The connection comes from the unpooled data source
/// (<see cref="SecretsServiceCollectionExtensions.UnpooledDataSourceKey"/>), so it never holds a slot of the pool
/// the writers use. Notifications go through the bounded <see cref="LiveEventQueue"/> (oldest dropped when full)
/// to the <see cref="LiveEventDispatcher"/>, so neither a slow client nor a slow read ever pushes back on the
/// connection — and the writers (the bot first) never wait on the feed at all: NOTIFY doesn't block on
/// listeners. A lost connection is re-opened with a capped exponential backoff; what was committed meanwhile is
/// not replayed (at-most-once — clients re-fetch over REST after a reconnect). The data source is resolved on
/// each attempt, so a malformed connection string is retried like an unreachable database instead of stopping
/// the host.
/// </summary>
public sealed class LiveNotificationListener(
    IServiceProvider services,
    LiveEventDispatcher dispatcher,
    ILogger<LiveNotificationListener> logger,
    int capacity = 1024) : BackgroundService
{
    internal static readonly TimeSpan MinReconnectDelay = TimeSpan.FromSeconds(1);

    internal static readonly TimeSpan MaxReconnectDelay = TimeSpan.FromSeconds(30);

    /// <summary>How long the connection may stay silent before a ping checks it is still alive.</summary>
    internal static readonly TimeSpan KeepAlive = TimeSpan.FromSeconds(30);

    private readonly LiveEventQueue _queue = new(capacity);

    private volatile bool _isListening;

    /// <summary>
    /// Whether a connection is listening right now — false while (re)connecting, backing off or stopped. Tests
    /// wait on it before writing rows.
    /// </summary>
    internal bool IsListening => _isListening;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Opening the connection doesn't hold up the host's startup.
        await Task.Yield();
        Task pump = PumpAsync(stoppingToken);
        try
        {
            await ListenAsync(stoppingToken);
        }
        finally
        {
            _queue.Complete();
            await pump;
        }
    }

    private async Task ListenAsync(CancellationToken stoppingToken)
    {
        TimeSpan delay = MinReconnectDelay;
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                NpgsqlDataSource dataSource = services.GetRequiredKeyedService<NpgsqlDataSource>(
                    SecretsServiceCollectionExtensions.UnpooledDataSourceKey);
                await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(stoppingToken);
                connection.Notification += OnNotification;
                await using (NpgsqlCommand listen = new($"LISTEN {LiveEvents.Channel}", connection))
                {
                    await listen.ExecuteNonQueryAsync(stoppingToken);
                }

                _isListening = true;
                logger.LogInformation("live: listening on {Channel}", LiveEvents.Channel);
                delay = MinReconnectDelay;
                while (true)
                {
                    if (!await connection.WaitAsync(KeepAlive, stoppingToken))
                    {
                        await using NpgsqlCommand ping = new("SELECT 1", connection);
                        await ping.ExecuteScalarAsync(stoppingToken);
                    }
                }
            }
            catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
            {
                return;
            }
            catch (Exception ex)
            {
                logger.LogWarning(ex, "live: {Channel} connection lost, reconnecting in {Delay}", LiveEvents.Channel, delay);
            }
            finally
            {
                _isListening = false;
            }

            try
            {
                await Task.Delay(delay, stoppingToken);
            }
            catch (OperationCanceledException)
            {
                return;
            }

            delay = TimeSpan.FromTicks(Math.Min(delay.Ticks * 2, MaxReconnectDelay.Ticks));
        }
    }

    // Raised on the listening connection's thread: never blocks (a full queue drops its oldest entry).
    private void OnNotification(object sender, NpgsqlNotificationEventArgs e)
    {
        if (LiveNotification.TryParse(e.Payload, out LiveNotification notification))
        {
            _queue.Enqueue(notification);
        }
        else
        {
            logger.LogDebug("live: ignoring malformed notification {Payload}", e.Payload);
        }
    }

    private async Task PumpAsync(CancellationToken stoppingToken)
    {
        try
        {
            await _queue.PumpAsync(dispatcher, stoppingToken);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Shutdown abandons the backlog — clients re-fetch over REST anyway.
        }
    }
}
