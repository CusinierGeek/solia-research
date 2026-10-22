using Solia.Shared.Logging;

namespace Solia.Bot.Logging;

/// <summary>
/// The no-op <see cref="IBotLogSink"/>: the engine's and host's default when nothing operational should
/// be written (unit tests, deterministic replays). Keeps them runnable without a database and keeps
/// operational logging strictly additive — never a reason for an event to behave differently.
/// </summary>
public sealed class NullBotLogSink : IBotLogSink
{
    public static readonly NullBotLogSink Instance = new();

    private NullBotLogSink() { }

    public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
        Task.CompletedTask;
}
