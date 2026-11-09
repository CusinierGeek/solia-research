using System.ComponentModel.DataAnnotations;

namespace Solia.DataPipeline.Retention;

/// <summary>
/// Data retention knobs (the "Retention" section; flat RETENTION_* keys in .env.example). The data-pipeline is the
/// single owner of the purge: it owns <c>features</c>, the bulk of the volume, and the replay that must not write
/// rows the next purge would wipe. Validated at startup.
/// </summary>
public sealed class RetentionOptions : IValidatableObject
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

    /// <summary>
    /// A token is <em>established</em> once one of its rows reaches this <c>token_age_sec</c> (hours of continuous
    /// activity: the age restarts when an idle token is evicted and comes back). Its rows outlive
    /// <see cref="FeaturesTtlDays"/>, up to <see cref="EstablishedTtlDays"/>. At least 1 h: the partial index
    /// <c>features_established_idx</c> (migration 0018) covers rows aged 1 h or more.
    /// </summary>
    [Range(1, 720)]
    public int EstablishedAgeHours { get; set; } = 6;

    /// <summary>
    /// Days of <c>features</c> kept for established tokens. Equal to <see cref="FeaturesTtlDays"/>: no special
    /// treatment; lower: rejected.
    /// </summary>
    [Range(1, 3650)]
    public int EstablishedTtlDays { get; set; } = 90;

    /// <summary>
    /// Once an established token's rows pass <see cref="FeaturesTtlDays"/>, those younger than this age
    /// (<c>token_age_sec</c>) are all kept; older ones are thinned to one per <see cref="EstablishedSampleSeconds"/>.
    /// </summary>
    [Range(0, 43_200)]
    public int EstablishedFullResolutionMinutes { get; set; } = 60;

    /// <summary>
    /// Bucket width of the thinning: past <see cref="FeaturesTtlDays"/>, an established token keeps the first row of
    /// each bucket once older than <see cref="EstablishedFullResolutionMinutes"/>. 0 keeps every row.
    /// </summary>
    [Range(0, 86_400)]
    public int EstablishedSampleSeconds { get; set; } = 60;

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

    /// <summary>Established tokens' rows outlive the general features TTL.</summary>
    public bool KeepsEstablished => EstablishedTtlDays > FeaturesTtlDays;

    public TimeSpan EstablishedTtl => TimeSpan.FromDays(EstablishedTtlDays);

    public EstablishedRule Established => new(
        TimeSpan.FromHours(EstablishedAgeHours),
        TimeSpan.FromMinutes(EstablishedFullResolutionMinutes),
        TimeSpan.FromSeconds(EstablishedSampleSeconds));

    public IEnumerable<ValidationResult> Validate(ValidationContext validationContext)
    {
        if (EstablishedTtlDays < FeaturesTtlDays)
        {
            yield return new ValidationResult(
                $"Retention:EstablishedTtlDays ({EstablishedTtlDays}) must be >= Retention:FeaturesTtlDays ({FeaturesTtlDays}).",
                [nameof(EstablishedTtlDays)]);
        }
    }

    public TimeSpan TtlOf(RetainedTable table) => TimeSpan.FromDays(table switch
    {
        RetainedTable.Features => FeaturesTtlDays,
        RetainedTable.Logs => LogsTtlDays,
        RetainedTable.Predictions => PredictionsTtlDays,
        _ => throw new ArgumentOutOfRangeException(nameof(table), table, null),
    });
}
