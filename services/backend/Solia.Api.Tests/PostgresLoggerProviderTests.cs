using System.Diagnostics;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Solia.Api.Logging;
using Solia.Shared.Logging;

namespace Solia.Api.Tests;

public class PostgresLoggerProviderTests
{
    private static PostgresLoggerProvider Provider(ILogSink sink) =>
        new(sink, capacity: 64, writeTimeout: TimeSpan.FromMilliseconds(200), drainTimeout: TimeSpan.FromMilliseconds(200));

    [Fact]
    public async Task Logged_entry_reaches_the_sink()
    {
        var sink = new CapturingSink();
        using PostgresLoggerProvider provider = Provider(sink);
        ILogger logger = provider.CreateLogger("Solia.Api.Test");

        logger.LogInformation("hello {Who}", "world");

        (LogSeverity level, string message, object? context) = await sink.NextAsync();
        Assert.Equal(LogSeverity.Info, level);
        Assert.Equal("hello world", message);
        Dictionary<string, object?> ctx = Assert.IsType<Dictionary<string, object?>>(context);
        Assert.Equal("Solia.Api.Test", ctx["category"]);
    }

    [Fact]
    public async Task Entries_logged_inside_a_suppression_scope_never_reach_the_sink_whatever_their_category()
    {
        var sink = new CapturingSink();
        using PostgresLoggerProvider provider = Provider(sink);

        using (PostgresLoggerProvider.Suppress())
        {
            provider.CreateLogger("Microsoft.EntityFrameworkCore.Query").LogError("read failed");
            await Task.Run(() => provider.CreateLogger("Npgsql").LogError("flowed into a child task"));
        }

        provider.CreateLogger("cat").LogWarning("after the scope");

        Assert.Equal("after the scope", (await sink.NextAsync()).Message);
        Assert.False(PostgresLoggerProvider.IsSuppressed);
    }

    [Theory]
    [InlineData(LogLevel.Trace, LogSeverity.Debug)]
    [InlineData(LogLevel.Debug, LogSeverity.Debug)]
    [InlineData(LogLevel.Information, LogSeverity.Info)]
    [InlineData(LogLevel.Warning, LogSeverity.Warn)]
    [InlineData(LogLevel.Error, LogSeverity.Error)]
    [InlineData(LogLevel.Critical, LogSeverity.Error)]
    public async Task Log_levels_map_onto_sink_severities(LogLevel level, LogSeverity expected)
    {
        var sink = new CapturingSink();
        using PostgresLoggerProvider provider = Provider(sink);

        provider.CreateLogger("cat").Log(level, "msg");

        Assert.Equal(expected, (await sink.NextAsync()).Level);
    }

    [Fact]
    public async Task Sink_failure_never_surfaces_to_the_caller()
    {
        using PostgresLoggerProvider provider = Provider(new ThrowingSink());
        ILogger logger = provider.CreateLogger("cat");

        logger.LogError("boom"); // must not throw
        await Task.Delay(100); // let the background consumer run and swallow the failure

        logger.LogError("still alive"); // the provider keeps working after a failed write
    }

    [Fact]
    public void Dispose_is_bounded_even_when_the_sink_hangs()
    {
        PostgresLoggerProvider provider = Provider(new HangingSink());
        provider.CreateLogger("cat").LogWarning("queued");

        var sw = Stopwatch.StartNew();
        provider.Dispose();
        sw.Stop();

        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(2), $"Dispose took {sw.Elapsed}");
    }

    private sealed class CapturingSink : ILogSink
    {
        private readonly Channel<(LogSeverity, string, object?)> _entries =
            Channel.CreateUnbounded<(LogSeverity, string, object?)>();

        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
        {
            _entries.Writer.TryWrite((level, message, context));
            return Task.CompletedTask;
        }

        public async Task<(LogSeverity Level, string Message, object? Context)> NextAsync()
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            return await _entries.Reader.ReadAsync(cts.Token);
        }
    }

    private sealed class ThrowingSink : ILogSink
    {
        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
            throw new InvalidOperationException("sink down");
    }

    private sealed class HangingSink : ILogSink
    {
        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
            Task.Delay(Timeout.Infinite, ct);
    }
}
