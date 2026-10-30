using Solia.DataPipeline.Ingestion.Replay;

namespace Solia.DataPipeline.Retention;

public enum ReplayRetentionVerdict
{
    Ok,
    Warn,
    Refuse,
}

/// <summary>
/// Stops a historical replay from silently feeding the purge: rows of a window older than the <c>features</c> TTL
/// would be deleted by the next retention run. Such a replay is refused (or only warned about with
/// <c>--allow-expired</c>); one whose oldest rows expire within a day is warned about.
/// </summary>
public static class ReplayRetentionGuard
{
    /// <summary>A window starting this close to the cutoff is replayed, with a warning that it expires soon.</summary>
    public static readonly TimeSpan ExpiringSoon = TimeSpan.FromDays(1);

    public static (ReplayRetentionVerdict Verdict, string? Message) Check(
        ReplayWindow window, RetentionOptions retention, DateTimeOffset nowUtc, bool allowExpired)
    {
        if (!retention.Enabled)
        {
            return (ReplayRetentionVerdict.Ok, null);
        }

        DateTimeOffset cutoff = nowUtc - retention.TtlOf(RetainedTable.Features);
        if (window.StartUtc < cutoff)
        {
            string message =
                $"the window starts at {window.StartUtc:O}, before the features retention cutoff {cutoff:O} " +
                $"(RETENTION_FEATURES_TTL_DAYS={retention.FeaturesTtlDays}): the next retention run would delete what it " +
                "writes. To train on that history, raise RETENTION_FEATURES_TTL_DAYS (on the live pipeline too) to cover " +
                "the window, or set RETENTION_ENABLED=false on the live pipeline while you use it";
            return allowExpired
                ? (ReplayRetentionVerdict.Warn, message + " — replaying anyway (--allow-expired)")
                : (ReplayRetentionVerdict.Refuse, message + "; --allow-expired replays it regardless.");
        }

        return window.StartUtc < cutoff + ExpiringSoon
            ? (ReplayRetentionVerdict.Warn,
                $"the window's oldest rows ({window.StartUtc:O}) are deleted by retention from " +
                $"{window.StartUtc + retention.TtlOf(RetainedTable.Features):O} (RETENTION_FEATURES_TTL_DAYS={retention.FeaturesTtlDays})")
            : (ReplayRetentionVerdict.Ok, null);
    }
}
