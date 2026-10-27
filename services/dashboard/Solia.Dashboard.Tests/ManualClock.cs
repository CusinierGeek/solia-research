namespace Solia.Dashboard.Tests;

/// <summary>A fixed clock whose timers only fire when the test ticks them.</summary>
internal sealed class ManualClock(DateTimeOffset now) : TimeProvider
{
    private readonly List<ManualTimer> _timers = [];

    public override DateTimeOffset GetUtcNow() => now;

    public override ITimer CreateTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
    {
        var timer = new ManualTimer(callback, state, dueTime, period);
        _timers.Add(timer);
        return timer;
    }

    /// <summary>Fires every periodic timer of that <paramref name="period"/>; a disposed one too when it raced its disposal.</summary>
    public void Tick(TimeSpan period, bool includeDisposed = false)
    {
        foreach (ManualTimer timer in _timers.Where(t => t.Period == period && (includeDisposed || !t.Disposed)).ToList())
        {
            timer.Callback(timer.State);
        }
    }

    /// <summary>Whether a one-shot timer is armed to fire after <paramref name="dueTime"/>.</summary>
    public bool IsArmed(TimeSpan dueTime) => _timers.Any(t => !t.Disposed && t.IsOneShot && t.DueTime == dueTime);

    /// <summary>Fires the one-shot timers armed for <paramref name="dueTime"/>, which disarms them (as it would elapse).</summary>
    public void Elapse(TimeSpan dueTime)
    {
        foreach (ManualTimer timer in _timers.Where(t => !t.Disposed && t.IsOneShot && t.DueTime == dueTime).ToList())
        {
            timer.DueTime = Timeout.InfiniteTimeSpan;
            timer.Callback(timer.State);
        }
    }

    private sealed class ManualTimer(TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period) : ITimer
    {
        public TimerCallback Callback { get; } = callback;
        public object? State { get; } = state;
        public TimeSpan DueTime { get; set; } = dueTime;
        public TimeSpan Period { get; private set; } = period;
        public bool Disposed { get; private set; }

        public bool IsOneShot => Period == Timeout.InfiniteTimeSpan && DueTime != Timeout.InfiniteTimeSpan;

        public bool Change(TimeSpan dueTime, TimeSpan period)
        {
            DueTime = dueTime;
            Period = period;
            return !Disposed;
        }

        public void Dispose() => Disposed = true;

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
