using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

public class DebouncerTests
{
    private static readonly TimeSpan Delay = TimeSpan.FromSeconds(1);

    private readonly ManualClock _clock = new(DateTimeOffset.UnixEpoch);
    private int _runs;

    private Debouncer Create(Func<Task>? action = null) =>
        new(_clock, Delay, action ?? (() => { _runs++; return Task.CompletedTask; }), NullLogger.Instance);

    [Fact]
    public void A_burst_of_triggers_runs_the_action_once_after_the_delay()
    {
        using Debouncer debouncer = Create();

        debouncer.Trigger();
        debouncer.Trigger();
        debouncer.Trigger();

        Assert.Equal(0, _runs);
        Assert.True(debouncer.IsPending);
        _clock.Elapse(Delay);
        Assert.Equal(1, _runs);
        Assert.False(debouncer.IsPending);
    }

    [Fact]
    public void A_trigger_after_the_run_arms_a_new_one()
    {
        using Debouncer debouncer = Create();
        debouncer.Trigger();
        _clock.Elapse(Delay);

        debouncer.Trigger();
        _clock.Elapse(Delay);

        Assert.Equal(2, _runs);
    }

    [Fact]
    public void Nothing_runs_without_a_trigger()
    {
        using Debouncer debouncer = Create();

        Assert.False(_clock.IsArmed(Delay));
        _clock.Elapse(Delay);

        Assert.Equal(0, _runs);
    }

    [Fact]
    public void Cancel_drops_the_pending_run()
    {
        using Debouncer debouncer = Create();
        debouncer.Trigger();

        debouncer.Cancel();

        Assert.False(debouncer.IsPending);
        Assert.False(_clock.IsArmed(Delay));
        _clock.Elapse(Delay);
        Assert.Equal(0, _runs);
    }

    [Fact]
    public void A_disposed_debouncer_ignores_triggers()
    {
        Debouncer debouncer = Create();
        debouncer.Trigger();

        debouncer.Dispose();
        debouncer.Trigger();
        debouncer.Cancel(); // harmless after disposal

        Assert.False(debouncer.IsPending);
        Assert.False(_clock.IsArmed(Delay));
        Assert.Equal(0, _runs);
    }

    [Fact]
    public async Task A_failing_action_is_swallowed_and_the_next_trigger_still_runs()
    {
        TaskCompletionSource ran = new();
        int attempts = 0;
        using Debouncer debouncer = Create(() =>
        {
            if (++attempts == 1)
            {
                throw new InvalidOperationException("boom");
            }
            ran.SetResult();
            return Task.CompletedTask;
        });

        debouncer.Trigger();
        _clock.Elapse(Delay);
        debouncer.Trigger();
        _clock.Elapse(Delay);

        await ran.Task.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(2, attempts);
    }

    [Fact]
    public async Task A_torn_down_host_is_ignored_while_other_failures_are_logged()
    {
        RecordingLogger logger = new();
        TaskCompletionSource ran = new();
        int attempts = 0;
        using Debouncer debouncer = new(_clock, Delay, () =>
        {
            attempts++;
            if (attempts == 1)
            {
                throw new ObjectDisposedException("renderer");
            }
            ran.SetResult();
            throw new InvalidOperationException("boom");
        }, logger);

        debouncer.Trigger();
        _clock.Elapse(Delay);
        debouncer.Trigger();
        _clock.Elapse(Delay);

        await ran.Task.WaitAsync(TimeSpan.FromSeconds(5));
        await Task.Yield();
        Assert.Equal([LogLevel.Warning], logger.Levels);
        Assert.IsType<InvalidOperationException>(Assert.Single(logger.Exceptions));
    }

    [Fact]
    public void The_delay_must_be_positive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new Debouncer(_clock, TimeSpan.Zero, () => Task.CompletedTask, NullLogger.Instance));
    }

    private sealed class RecordingLogger : ILogger
    {
        public List<LogLevel> Levels { get; } = [];
        public List<Exception?> Exceptions { get; } = [];

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
        {
            Levels.Add(logLevel);
            Exceptions.Add(exception);
        }
    }
}
