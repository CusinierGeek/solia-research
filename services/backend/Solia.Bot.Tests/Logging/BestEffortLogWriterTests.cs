using Microsoft.Extensions.Logging;
using Solia.Bot.Logging;
using Solia.Shared.Logging;

namespace Solia.Bot.Tests.Logging;

/// <summary>
/// The writer every bot log goes through: it never throws, bounds a slow sink, reports real failures on
/// <see cref="ILogger"/> and stays quiet when the caller cancelled.
/// </summary>
public class BestEffortLogWriterTests
{
    private static readonly TimeSpan Short = TimeSpan.FromMilliseconds(100);

    [Fact]
    public async Task Forwards_the_entry_to_the_sink()
    {
        var sink = new RecordingSink();
        var writer = new BestEffortLogWriter(sink);

        await writer.WriteAsync(LogSeverity.Warn, "hello", new { a = 1 }, default);

        (LogSeverity level, string message) = Assert.Single(sink.Entries);
        Assert.Equal(LogSeverity.Warn, level);
        Assert.Equal("hello", message);
    }

    [Fact]
    public async Task A_failing_sink_is_swallowed_and_reported_as_a_warning()
    {
        var logger = new RecordingLogger();
        var writer = new BestEffortLogWriter(new DelegateSink((_, _) => throw new InvalidOperationException("db down")), logger);

        await writer.WriteAsync(LogSeverity.Info, "position opened", null, default);

        (LogLevel level, Exception? error) = Assert.Single(logger.Entries);
        Assert.Equal(LogLevel.Warning, level);
        Assert.IsType<InvalidOperationException>(error);
    }

    [Fact]
    public async Task A_hanging_sink_is_abandoned_after_the_timeout()
    {
        var logger = new RecordingLogger();
        // Ignores its token entirely — the writer must still bound it.
        var writer = new BestEffortLogWriter(new DelegateSink((_, _) => Task.Delay(Timeout.Infinite)), logger, Short);

        await writer.WriteAsync(LogSeverity.Info, "bot stopping", null, default).WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal(LogLevel.Warning, Assert.Single(logger.Entries).Level);
    }

    [Fact]
    public async Task A_write_cancelled_by_the_caller_is_not_reported_as_a_failure()
    {
        var logger = new RecordingLogger();
        var writer = new BestEffortLogWriter(new DelegateSink((_, ct) => Task.Delay(Timeout.Infinite, ct)), logger);
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await writer.WriteAsync(LogSeverity.Info, "bot started", null, cts.Token);

        Assert.DoesNotContain(logger.Entries, e => e.Level >= LogLevel.Warning);
    }

    [Fact]
    public async Task A_non_cancellation_error_caused_by_the_callers_cancel_is_not_a_failure()
    {
        var logger = new RecordingLogger();
        using var cts = new CancellationTokenSource();
        // Npgsql can surface a cancelled command as a broken-connection error rather than an OCE.
        var writer = new BestEffortLogWriter(new DelegateSink((_, _) =>
        {
            cts.Cancel();
            throw new IOException("connection broken by cancel");
        }), logger);

        await writer.WriteAsync(LogSeverity.Info, "bot started", null, cts.Token);

        Assert.DoesNotContain(logger.Entries, e => e.Level >= LogLevel.Warning);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void A_non_positive_timeout_is_rejected(int ms) =>
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new BestEffortLogWriter(new RecordingSink(), timeout: TimeSpan.FromMilliseconds(ms)));

    [Fact]
    public void A_null_sink_is_rejected() => Assert.Throws<ArgumentNullException>(() => new BestEffortLogWriter(null!));

    private sealed class RecordingSink : IBotLogSink
    {
        public List<(LogSeverity Level, string Message)> Entries { get; } = [];

        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
        {
            Entries.Add((level, message));
            return Task.CompletedTask;
        }
    }

    private sealed class DelegateSink(Func<string, CancellationToken, Task> write) : IBotLogSink
    {
        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
            write(message, ct);
    }
}

/// <summary>An <see cref="ILogger"/> that keeps each entry's level and exception, for assertions.</summary>
internal sealed class RecordingLogger : ILogger
{
    private readonly List<(LogLevel Level, Exception? Error)> _entries = [];

    public IReadOnlyList<(LogLevel Level, Exception? Error)> Entries
    {
        get
        {
            lock (_entries)
            {
                return [.. _entries];
            }
        }
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        lock (_entries)
        {
            _entries.Add((logLevel, exception));
        }
    }
}
