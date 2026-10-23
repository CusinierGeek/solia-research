using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

namespace Solia.Api.Tests.EndToEnd;

/// <summary>
/// Records every log entry the host emits — rendered message plus the full exception text — as a provider sees
/// it, i.e. after the redacting logger factory. Thread-safe (the host logs from background services).
/// </summary>
public sealed class CapturingLoggerProvider : ILoggerProvider
{
    private readonly ConcurrentQueue<string> _entries = new();

    public IReadOnlyCollection<string> Entries => _entries.ToArray();

    public ILogger CreateLogger(string categoryName) => new CapturingLogger(categoryName, _entries);

    public void Dispose()
    {
    }

    private sealed class CapturingLogger(string category, ConcurrentQueue<string> entries) : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(
            LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
            entries.Enqueue($"{logLevel} {category}: {formatter(state, exception)}{(exception is null ? "" : "\n" + exception)}");
    }
}
