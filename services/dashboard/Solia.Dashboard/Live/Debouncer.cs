namespace Solia.Dashboard.Live;

/// <summary>
/// Coalesces a burst of live events into one action: the first <see cref="Trigger"/> arms a one-shot timer, and the
/// triggers that follow before it fires ride along, so the action runs once, <c>delay</c> after the burst started.
/// Arming on the first trigger rather than re-arming on each keeps a steady stream from postponing it forever.
/// <see cref="Cancel"/> drops a pending run (e.g. when the host just refreshed anyway). The action runs on the timer's
/// thread and is not awaited: its failures are logged, never thrown (an <see cref="ObjectDisposedException"/> from a
/// torn-down host is expected and ignored). Thread-safe.
/// </summary>
public sealed class Debouncer : IDisposable
{
    private readonly Lock _gate = new();
    private readonly TimeSpan _delay;
    private readonly Func<Task> _action;
    private readonly ILogger _logger;
    private readonly ITimer _timer;
    private bool _pending;
    private bool _disposed;

    public Debouncer(TimeProvider clock, TimeSpan delay, Func<Task> action, ILogger logger)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(delay, TimeSpan.Zero);
        _delay = delay;
        _action = action;
        _logger = logger;
        _timer = clock.CreateTimer(_ => Fire(), null, Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
    }

    /// <summary>Whether a run is armed and not yet started.</summary>
    public bool IsPending
    {
        get
        {
            lock (_gate)
            {
                return _pending;
            }
        }
    }

    /// <summary>Schedules the action, unless a run is already pending.</summary>
    public void Trigger()
    {
        lock (_gate)
        {
            if (_pending || _disposed)
            {
                return;
            }
            _pending = true;
            _timer.Change(_delay, Timeout.InfiniteTimeSpan);
        }
    }

    /// <summary>Drops the pending run, if any.</summary>
    public void Cancel()
    {
        lock (_gate)
        {
            _pending = false;
            if (!_disposed)
            {
                _timer.Change(Timeout.InfiniteTimeSpan, Timeout.InfiniteTimeSpan);
            }
        }
    }

    // A tick that raced a Cancel or Dispose finds nothing pending and does nothing.
    private void Fire()
    {
        lock (_gate)
        {
            if (!_pending || _disposed)
            {
                return;
            }
            _pending = false;
        }
        _ = RunAsync();
    }

    private async Task RunAsync()
    {
        try
        {
            await _action();
        }
        catch (ObjectDisposedException)
        {
            // the host (e.g. a circuit) was torn down between the tick and its Dispose: nothing to refresh
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "A debounced live refresh failed.");
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
            {
                return;
            }
            _disposed = true;
            _pending = false;
        }
        _timer.Dispose();
    }
}
