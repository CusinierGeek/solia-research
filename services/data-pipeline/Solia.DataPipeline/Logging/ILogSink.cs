namespace Solia.DataPipeline.Logging;

/// <summary>
/// Persists operational log entries to the shared `logs` table (surfaced on the dashboard).
/// This is for notable operational events — not a replacement for standard ILogger output.
/// </summary>
public interface ILogSink
{
    Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default);
}
