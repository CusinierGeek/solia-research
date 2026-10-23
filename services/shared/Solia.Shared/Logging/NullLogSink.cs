namespace Solia.Shared.Logging;

/// <summary>
/// An <see cref="ILogSink"/> that drops every entry — the fallback when the `logs` table can't be reached
/// at all (e.g. a malformed connection string), so an optional log mirror never stops a host from starting.
/// </summary>
public sealed class NullLogSink : ILogSink
{
    public static readonly NullLogSink Instance = new();

    private NullLogSink()
    {
    }

    public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
        Task.CompletedTask;
}
