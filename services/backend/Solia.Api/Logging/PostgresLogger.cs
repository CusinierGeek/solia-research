using Microsoft.Extensions.Logging;

namespace Solia.Api.Logging;

/// <summary>
/// The <see cref="ILogger"/> handed out by <see cref="PostgresLoggerProvider"/> for one category. It only
/// formats the entry and hands it to the provider's queue; the minimum level is enforced by the logging
/// framework's filter pipeline (the <c>Postgres</c> provider alias) before <see cref="Log"/> is called.
/// </summary>
internal sealed class PostgresLogger(string category, PostgresLoggerProvider provider) : ILogger
{
    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => logLevel != LogLevel.None;

    public void Log<TState>(
        LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        if (!IsEnabled(logLevel))
        {
            return;
        }

        ArgumentNullException.ThrowIfNull(formatter);
        string message = formatter(state, exception);
        if (string.IsNullOrEmpty(message) && exception is null)
        {
            return;
        }

        provider.Enqueue(logLevel, category, eventId, message, exception);
    }
}
