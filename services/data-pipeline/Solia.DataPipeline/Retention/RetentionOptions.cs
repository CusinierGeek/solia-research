using System.ComponentModel.DataAnnotations;

namespace Solia.DataPipeline.Retention;

/// <summary>
/// Data retention knobs (the "Retention" section; flat RETENTION_* keys in .env.example). The data-pipeline is the
/// single owner of the purge: it owns <c>features</c>, the bulk of the volume, and the replay that must not write
/// rows the next purge would wipe. Validated at startup.
/// </summary>
public sealed class RetentionOptions
{
    public const string Section = "Retention";

    /// <summary>
    /// Run the periodic purge. Off for a bare local run (like <c>Pipeline:Ingest</c>); compose turns it on. Also
    /// arms the replay guard: a replay of a window the purge would delete is refused.
    /// </summary>
    public bool Enabled { get; set; } = false;

    /// <summary>Minutes between two purges (the first one runs at startup).</summary>
    [Range(1, 1440)]
    public int IntervalMinutes { get; set; } = 60;

    /// <summary>
    /// Days of <c>features</c> kept. Bounds what a retrain reads, so keep it above what training needs
    /// (held-out windows + label horizons, max 1 h).
    /// </summary>
    [Range(1, 3650)]
    public int FeaturesTtlDays { get; set; } = 14;

    /// <summary>Days of <c>logs</c> kept (every service's rows).</summary>
    [Range(1, 3650)]
    public int LogsTtlDays { get; set; } = 14;

    /// <summary>Days of <c>predictions</c> kept (docs/DATA-MODEL.md: ~30 days hot).</summary>
    [Range(1, 3650)]
    public int PredictionsTtlDays { get; set; } = 30;

    /// <summary>Rows deleted per statement (each its own short transaction).</summary>
    [Range(100, 1_000_000)]
    public int BatchSize { get; set; } = 10_000;

    public TimeSpan Interval => TimeSpan.FromMinutes(IntervalMinutes);

    public TimeSpan TtlOf(RetainedTable table) => TimeSpan.FromDays(table switch
    {
        RetainedTable.Features => FeaturesTtlDays,
        RetainedTable.Logs => LogsTtlDays,
        RetainedTable.Predictions => PredictionsTtlDays,
        _ => throw new ArgumentOutOfRangeException(nameof(table), table, null),
    });
}
