using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Solia.Shared.Logging;

namespace Solia.Api.Logging;

/// <summary>
/// An <see cref="ILoggerProvider"/> that mirrors notable <c>ILogger</c> output into the shared `logs`
/// table (service = 'backend') through an <see cref="ILogSink"/> — the backend's equivalent of the bot's
/// <c>PostgresBotLogSink</c> path, so warnings/errors surface on the dashboard without an explicit sink
/// call at every site. Writing is <strong>best-effort</strong>: entries are queued and drained on a single
/// background task, so a logging call never blocks on the database; a full queue drops the oldest write, a
/// failed or slow write is bounded and swallowed (a lost `logs` row must never crash the app, stall logging
/// or hold up shutdown). Which levels reach the table is the normal logging filter's job — configure it
/// under the <c>Postgres</c> provider alias (defaults to Warning in appsettings).
/// </summary>
[ProviderAlias("Postgres")]
public sealed class PostgresLoggerProvider : ILoggerProvider
{
    private readonly ILogSink _sink;
    private readonly TimeSpan _writeTimeout;
    private readonly TimeSpan _drainTimeout;
    private readonly Channel<PendingLog> _queue;
    private readonly CancellationTokenSource _shutdown = new();
    private readonly Task _consumer;

    public PostgresLoggerProvider(ILogSink sink, int capacity = 1024, TimeSpan? writeTimeout = null, TimeSpan? drainTimeout = null)
    {
        ArgumentNullException.ThrowIfNull(sink);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);

        _sink = sink;
        _writeTimeout = writeTimeout ?? TimeSpan.FromSeconds(2);
        _drainTimeout = drainTimeout ?? TimeSpan.FromSeconds(2);
        _queue = Channel.CreateBounded<PendingLog>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.DropOldest,
            SingleReader = true,
        });
        _consumer = Task.Run(ConsumeAsync);
    }

    public ILogger CreateLogger(string categoryName) => new PostgresLogger(categoryName, this);

    /// <summary>Queues one entry; returns at once, dropping it if the queue is saturated.</summary>
    internal void Enqueue(LogLevel level, string category, EventId eventId, string message, Exception? exception) =>
        _queue.Writer.TryWrite(new PendingLog(ToSeverity(level), message, BuildContext(category, eventId, exception)));

    private async Task ConsumeAsync()
    {
        try
        {
            await foreach (PendingLog entry in _queue.Reader.ReadAllAsync(_shutdown.Token))
            {
                await WriteBoundedAsync(entry);
            }
        }
        catch (OperationCanceledException)
        {
            // Shutdown abandoned the remaining backlog — expected, not a failure.
        }
    }

    private async Task WriteBoundedAsync(PendingLog entry)
    {
        using var bounded = CancellationTokenSource.CreateLinkedTokenSource(_shutdown.Token);
        bounded.CancelAfter(_writeTimeout);
        try
        {
            // WaitAsync also bounds a sink that ignores its token.
            await _sink.WriteAsync(entry.Level, entry.Message, entry.Context, bounded.Token).WaitAsync(bounded.Token);
        }
        catch (OperationCanceledException) when (_shutdown.IsCancellationRequested)
        {
            // Dropped on shutdown — nothing to report.
        }
        catch (Exception ex)
        {
            // Report to stderr, never back through ILogger: that could loop straight into this sink.
            Console.Error.WriteLine($"[PostgresLoggerProvider] dropped log '{entry.Message}': {ex.Message}");
        }
    }

    public void Dispose()
    {
        _queue.Writer.TryComplete();
        try
        {
            // Give the backlog a brief grace to drain, then abandon the rest so shutdown isn't held up.
            if (!_consumer.Wait(_drainTimeout))
            {
                _shutdown.Cancel();
                _consumer.Wait(_writeTimeout);
            }
        }
        catch (Exception)
        {
            // A faulted/cancelled consumer must not throw out of Dispose.
        }
        _shutdown.Dispose();
    }

    private static object BuildContext(string category, EventId eventId, Exception? exception)
    {
        var context = new Dictionary<string, object?> { ["category"] = category };
        if (eventId.Id != 0)
        {
            context["eventId"] = eventId.Id;
        }
        if (exception is not null)
        {
            context["exception"] = exception.ToString();
        }
        return context;
    }

    private static LogSeverity ToSeverity(LogLevel level) => level switch
    {
        LogLevel.Trace or LogLevel.Debug => LogSeverity.Debug,
        LogLevel.Information => LogSeverity.Info,
        LogLevel.Warning => LogSeverity.Warn,
        LogLevel.Error or LogLevel.Critical => LogSeverity.Error,
        _ => LogSeverity.Info,
    };

    private readonly record struct PendingLog(LogSeverity Level, string Message, object? Context);
}
