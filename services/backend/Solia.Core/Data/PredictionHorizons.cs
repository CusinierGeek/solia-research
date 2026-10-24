namespace Solia.Core.Data;

/// <summary>
/// The <c>predictions.horizon</c> values, shortest first. The CHECK constraint in services/db/migrations is the
/// source of truth; a test fails when the two drift.
/// </summary>
public static class PredictionHorizons
{
    public static readonly IReadOnlyList<string> All = ["30s", "1m", "5m", "15m", "1h"];

    public static bool IsKnown(string? horizon) => horizon is not null && All.Contains(horizon, StringComparer.Ordinal);
}
