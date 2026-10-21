namespace Solia.DataPipeline.Logging;

/// <summary>
/// The levels accepted by the `logs` table (CHECK level IN ('debug','info','warn','error')).
/// An enum so a wrong spelling is a compile error rather than a rejected INSERT.
/// </summary>
public enum LogSeverity
{
    Debug,
    Info,
    Warn,
    Error,
}

public static class LogSeverityExtensions
{
    public static string ToDbValue(this LogSeverity severity) => severity switch
    {
        LogSeverity.Debug => "debug",
        LogSeverity.Info => "info",
        LogSeverity.Warn => "warn",
        LogSeverity.Error => "error",
        _ => throw new ArgumentOutOfRangeException(nameof(severity), severity, null),
    };
}
