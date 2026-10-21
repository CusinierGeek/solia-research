using System.Buffers;
using System.Runtime.CompilerServices;
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

    // 2^20 × base already exceeds any allowed max delay; stops the counter growing forever.
    private const int MaxBackoffAttempt = 20;

    private readonly Func<IWebSocket> _socketFactory;
    private readonly PumpApiOptions _options;
    private readonly StreamConnectionState _state;
    private readonly ILogger<PumpApiStreamSource> _logger;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;
    private readonly TimeSpan _idleTimeout;

    public PumpApiStreamSource(
        Func<IWebSocket> socketFactory,
        IOptions<PumpApiOptions> options,
        StreamConnectionState state,
        ILogger<PumpApiStreamSource> logger)
        : this(socketFactory, options.Value, state, logger, static (d, ct) => Task.Delay(d, ct))
    {
    }

    // Test seam: inject the backoff delay (and optionally a short idle timeout) so unit tests
    // run without real waits.
    internal PumpApiStreamSource(
        Func<IWebSocket> socketFactory,
        PumpApiOptions options,
        StreamConnectionState state,
        ILogger<PumpApiStreamSource> logger,
        Func<TimeSpan, CancellationToken, Task> delay,
        TimeSpan? idleTimeout = null)
    {
        _socketFactory = socketFactory;
        _options = options;
        _state = state;
        _logger = logger;
        _delay = delay;
        _idleTimeout = idleTimeout ?? TimeSpan.FromSeconds(options.ReceiveIdleTimeoutSeconds);
    }

    public async IAsyncEnumerable<RawEvent> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var uri = new Uri(_options.WsUrl);
        int attempt = 0;

        // try/finally (not catch) so cleanup also runs when the consumer stops enumerating early.
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                bool receivedAny = false;
                IWebSocket socket = _socketFactory();
                try
                {
                    if (await TryConnectAsync(socket, uri, cancellationToken))
                    {
                        using var session = new Session(cancellationToken);
                        await foreach (RawEvent evt in ReadLoopAsync(socket, session, cancellationToken))
                        {
                            yield return evt;
                        }

                        receivedAny = session.ReceivedAny;
                    }
                }
                finally
                {
                    await socket.DisposeAsync();
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                // Only a connection that actually delivered data resets the backoff; one the
                // server accepts then closes straight away (rate limit, overload) keeps growing it.
                if (receivedAny)
                {
                    attempt = 0;
                }

                TimeSpan wait = ComputeBackoff(attempt, _options.ReconnectBaseDelaySeconds, _options.ReconnectMaxDelaySeconds, Random.Shared.NextDouble());
                attempt = Math.Min(attempt + 1, MaxBackoffAttempt);
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
        }
        finally
        {
            _state.MarkDisconnected("stopped");
        }
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
        IWebSocket socket, Session session, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            // yield can't live inside try/catch, so receiving and parsing (which may throw) are isolated.
            ReadOnlyMemory<byte>? message = await TryReceiveMessageAsync(socket, session, cancellationToken);
            if (message is not { } utf8)
            {
                break; // closed / error / idle / cancelled — outer loop reconnects (or stops)
            }

            RawEvent? evt = TryParse(utf8);
            if (evt is not null)
            {
                yield return evt;
            }
        }
    }

    /// <summary>The parser shouldn't throw, but if it ever does, drop the frame rather than the stream.</summary>
    private RawEvent? TryParse(ReadOnlyMemory<byte> utf8)
    {
        try
        {
            return PumpApiEventParser.ParseUtf8(utf8);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "dropping a frame the parser failed on");
            return null;
        }
    }

    /// <summary>
    /// Reassembles fragmented frames into one UTF-8 message held in the session's reused buffer
    /// (valid until the next receive). Null on close/error/idle timeout/oversize/cancel.
    /// </summary>
    private async Task<ReadOnlyMemory<byte>?> TryReceiveMessageAsync(
        IWebSocket socket, Session session, CancellationToken cancellationToken)
    {
        MemoryStream message = session.Message;
        message.SetLength(0);
        session.Idle.CancelAfter(_idleTimeout); // restarts the idle timer for this message

        try
        {
            while (true)
            {
                WebSocketReceive result = await socket.ReceiveAsync(session.Buffer, session.Idle.Token);
                if (result.Closed)
                {
                    _state.MarkDisconnected("closed by server");
                    return null;
                }

                if (message.Length + result.Count > _options.MaxMessageBytes)
                {
                    string error = $"message exceeded {_options.MaxMessageBytes} bytes";
                    _state.MarkDisconnected(error);
                    _logger.LogWarning("stream {Error}; dropping connection", error);
                    return null;
                }

                message.Write(session.Buffer, 0, result.Count);
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
        catch (OperationCanceledException) when (session.Idle.IsCancellationRequested)
        {
            string error = $"no data for {_idleTimeout}";
            _state.MarkDisconnected(error);
            _logger.LogWarning("stream {Error}; assuming a dead connection", error);
            return null;
        }
        catch (Exception ex)
        {
            _state.MarkDisconnected(ex.Message);
            _logger.LogWarning(ex, "stream receive error");
            return null;
        }

        session.ReceivedAny = true;
        return new ReadOnlyMemory<byte>(message.GetBuffer(), 0, (int)message.Length);
    }

    /// <summary>Exponential backoff capped at max, plus up to 20% jitter (also capped).</summary>
    internal static TimeSpan ComputeBackoff(int attempt, double baseSeconds, double maxSeconds, double jitter01)
    {
        double exponential = Math.Min(maxSeconds, baseSeconds * Math.Pow(2, attempt));
        double withJitter = exponential + (jitter01 * exponential * 0.2);
        return TimeSpan.FromSeconds(Math.Min(maxSeconds, withJitter));
    }

    /// <summary>Per-connection receive resources, reused across messages instead of allocated per message.</summary>
    private sealed class Session(CancellationToken stopping) : IDisposable
    {
        public byte[] Buffer { get; } = ArrayPool<byte>.Shared.Rent(ReceiveBufferSize);

        public MemoryStream Message { get; } = new();

        /// <summary>Linked to shutdown; also fires when no message arrives within the idle timeout.</summary>
        public CancellationTokenSource Idle { get; } = CancellationTokenSource.CreateLinkedTokenSource(stopping);

        public bool ReceivedAny { get; set; }

        public void Dispose()
        {
            Idle.Dispose();
            Message.Dispose();
            ArrayPool<byte>.Shared.Return(Buffer);
        }
    }
}
