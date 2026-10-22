using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.Shared.Logging;

namespace Solia.Bot.Logging;

/// <summary>
/// Writes operational entries to an <see cref="IBotLogSink"/> without ever failing or stalling the
/// caller for long: logging is additive and must never abort an event, break the engine's
/// re-delivery invariant (a thrown write after a committed close would re-run the close on retry), or
/// hold up shutdown.
/// <list type="bullet">
///   <item>Each write is bounded by <see cref="Timeout"/>, so a degraded database delays the single
///   event loop by at most that much, and a final write at shutdown can't overrun the host's stop
///   timeout.</item>
///   <item>A failure or timeout is reported on <see cref="ILogger"/> (Warning) — a lost `logs` row stays
///   diagnosable from stdout.</item>
///   <item>A write abandoned because the caller's token fired is not a failure: it's skipped quietly
///   (Debug).</item>
/// </list>
/// </summary>
public sealed class BestEffortLogWriter
{
    /// <summary>Default per-write bound — well under the host's default 30 s shutdown timeout.</summary>
    public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(2);

    private readonly IBotLogSink _sink;
    private readonly ILogger _logger;

    public BestEffortLogWriter(IBotLogSink sink, ILogger? logger = null, TimeSpan? timeout = null)
    {
        ArgumentNullException.ThrowIfNull(sink);
        if (timeout is { } t && t <= TimeSpan.Zero)
        {
            throw new ArgumentOutOfRangeException(nameof(timeout), t, "The write timeout must be positive.");
        }

        _sink = sink;
        _logger = logger ?? NullLogger.Instance;
        Timeout = timeout ?? DefaultTimeout;
    }

    /// <summary>The longest a single write may take before it's abandoned.</summary>
    public TimeSpan Timeout { get; }

    /// <summary>Writes one entry; never throws.</summary>
    public async Task WriteAsync(LogSeverity level, string message, object? context, CancellationToken cancellationToken)
    {
        using var bounded = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        bounded.CancelAfter(Timeout);
        try
        {
            // WaitAsync also bounds a sink that ignores its token.
            await _sink.WriteAsync(level, message, context, bounded.Token).WaitAsync(bounded.Token);
        }
        catch (Exception) when (cancellationToken.IsCancellationRequested)
        {
            // Npgsql may surface a cancelled command as a non-OCE error; either way it's not an outage.
            _logger.LogDebug("operational log '{Message}' skipped: cancelled", message);
        }
        catch (OperationCanceledException) when (bounded.IsCancellationRequested)
        {
            _logger.LogWarning("could not write operational log '{Message}': timed out after {Timeout}", message, Timeout);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "could not write operational log '{Message}'", message);
        }
    }
}
