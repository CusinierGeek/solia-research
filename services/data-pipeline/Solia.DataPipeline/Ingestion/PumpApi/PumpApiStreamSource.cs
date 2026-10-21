using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>
/// Live <see cref="IDataSource"/> over the pumpapi WebSocket (<c>wss://stream.pumpapi.io/</c>).
/// The server pushes every event (no subscribe message); we filter client-side via the parser.
/// A single <see cref="StreamAsync"/> enumeration spans reconnects transparently: on any drop it
/// marks the connection down, backs off (exponential + jitter) and reconnects, so the consumer
/// (T7 engine) sees one uninterrupted stream until cancellation.
/// </summary>
public sealed class PumpApiStreamSource : IDataSource
{
    private const int ReceiveBufferSize = 16 * 1024;

    private readonly Func<IWebSocket> _socketFactory;
    private readonly PumpApiOptions _options;
    private readonly StreamConnectionState _state;
    private readonly ILogger<PumpApiStreamSource> _logger;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;

    public PumpApiStreamSource(
        Func<IWebSocket> socketFactory,
        IOptions<PumpApiOptions> options,
        StreamConnectionState state,
        ILogger<PumpApiStreamSource> logger)
        : this(socketFactory, options.Value, state, logger, static (d, ct) => Task.Delay(d, ct))
    {
    }

    // Test seam: inject the backoff delay so unit tests run without real waits.
    internal PumpApiStreamSource(
        Func<IWebSocket> socketFactory,
        PumpApiOptions options,
        StreamConnectionState state,
        ILogger<PumpApiStreamSource> logger,
        Func<TimeSpan, CancellationToken, Task> delay)
    {
        _socketFactory = socketFactory;
        _options = options;
        _state = state;
        _logger = logger;
        _delay = delay;
    }

    public async IAsyncEnumerable<RawEvent> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var uri = new Uri(_options.WsUrl);
        int attempt = 0;

        while (!cancellationToken.IsCancellationRequested)
        {
            IWebSocket socket = _socketFactory();
            bool connected = await TryConnectAsync(socket, uri, cancellationToken);

            if (connected)
            {
                attempt = 0;
                await foreach (RawEvent evt in ReadLoopAsync(socket, cancellationToken))
                {
                    yield return evt;
                }
            }

            await socket.DisposeAsync();

            if (cancellationToken.IsCancellationRequested)
            {
                break;
            }

            TimeSpan wait = ComputeBackoff(attempt++, _options.ReconnectBaseDelaySeconds, _options.ReconnectMaxDelaySeconds, Random.Shared.NextDouble());
            _logger.LogInformation("stream reconnecting in {Delay}", wait);
            try
            {
                await _delay(wait, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                break;
            }
        }

        _state.MarkDisconnected("stopped");
    }

    private async Task<bool> TryConnectAsync(IWebSocket socket, Uri uri, CancellationToken cancellationToken)
    {
        try
        {
            await socket.ConnectAsync(uri, cancellationToken);
            _state.MarkConnected();
            _logger.LogInformation("stream connected to {Uri}", uri);
            return true;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return false;
        }
        catch (Exception ex)
        {
            _state.MarkDisconnected(ex.Message);
            _logger.LogWarning(ex, "stream connect failed");
            return false;
        }
    }

    private async IAsyncEnumerable<RawEvent> ReadLoopAsync(
        IWebSocket socket, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            // yield can't live inside try/catch, so receiving (which may throw) is isolated here.
            string? message = await TryReceiveMessageAsync(socket, cancellationToken);
            if (message is null)
            {
                break; // closed / error / cancelled — outer loop reconnects (or stops)
            }

            RawEvent? evt = PumpApiEventParser.Parse(message);
            if (evt is not null)
            {
                yield return evt;
            }
        }
    }

    /// <summary>Reassembles fragmented frames into one UTF-8 message. Null on close/error/cancel.</summary>
    private async Task<string?> TryReceiveMessageAsync(IWebSocket socket, CancellationToken cancellationToken)
    {
        var buffer = new byte[ReceiveBufferSize];
        using var message = new MemoryStream();

        try
        {
            while (true)
            {
                WebSocketReceive result = await socket.ReceiveAsync(buffer, cancellationToken);
                if (result.Closed)
                {
                    _state.MarkDisconnected("closed by server");
                    return null;
                }

                message.Write(buffer, 0, result.Count);
                if (result.EndOfMessage)
                {
                    break;
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return null;
        }
        catch (Exception ex)
        {
            _state.MarkDisconnected(ex.Message);
            _logger.LogWarning(ex, "stream receive error");
            return null;
        }

        return Encoding.UTF8.GetString(message.GetBuffer(), 0, (int)message.Length);
    }

    /// <summary>Exponential backoff capped at max, plus up to 20% jitter (also capped).</summary>
    internal static TimeSpan ComputeBackoff(int attempt, double baseSeconds, double maxSeconds, double jitter01)
    {
        double exponential = Math.Min(maxSeconds, baseSeconds * Math.Pow(2, attempt));
        double withJitter = exponential + (jitter01 * exponential * 0.2);
        return TimeSpan.FromSeconds(Math.Min(maxSeconds, withJitter));
    }
}
