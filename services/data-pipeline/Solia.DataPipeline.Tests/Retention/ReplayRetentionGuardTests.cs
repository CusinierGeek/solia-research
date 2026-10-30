using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Retention;

namespace Solia.DataPipeline.Tests.Retention;

public class ReplayRetentionGuardTests
{
    private static readonly DateTimeOffset Now = new(2026, 10, 30, 12, 0, 0, TimeSpan.Zero);

    private static readonly RetentionOptions On = new() { Enabled = true, FeaturesTtlDays = 14 };

    private static ReplayWindow StartingDaysAgo(double days) => new(Now.AddDays(-days), Now.AddDays(-days).AddHours(1));

    [Fact]
    public void A_window_older_than_the_features_ttl_is_refused()
    {
        (ReplayRetentionVerdict verdict, string? message) = ReplayRetentionGuard.Check(StartingDaysAgo(20), On, Now, allowExpired: false);

        Assert.Equal(ReplayRetentionVerdict.Refuse, verdict);
        Assert.Contains("RETENTION_FEATURES_TTL_DAYS=14", message);
        Assert.Contains("--allow-expired", message);
    }

    [Fact]
    public void A_window_starting_just_before_the_cutoff_is_refused()
    {
        Assert.Equal(ReplayRetentionVerdict.Refuse,
            ReplayRetentionGuard.Check(new ReplayWindow(Now.AddDays(-14).AddSeconds(-1), Now), On, Now, false).Verdict);
    }

    [Fact]
    public void Allow_expired_turns_the_refusal_into_a_warning()
    {
        (ReplayRetentionVerdict verdict, string? message) = ReplayRetentionGuard.Check(StartingDaysAgo(20), On, Now, allowExpired: true);

        Assert.Equal(ReplayRetentionVerdict.Warn, verdict);
        Assert.Contains("replaying anyway", message);
    }

    [Fact]
    public void A_window_expiring_within_a_day_is_warned_about()
    {
        Assert.Equal(ReplayRetentionVerdict.Warn, ReplayRetentionGuard.Check(StartingDaysAgo(13.5), On, Now, false).Verdict);
    }

    [Fact]
    public void A_recent_window_passes()
    {
        Assert.Equal((ReplayRetentionVerdict.Ok, null), ReplayRetentionGuard.Check(StartingDaysAgo(2), On, Now, false));
    }

    [Fact]
    public void Retention_disabled_lets_any_window_through()
    {
        Assert.Equal(ReplayRetentionVerdict.Ok,
            ReplayRetentionGuard.Check(StartingDaysAgo(100), new RetentionOptions { Enabled = false }, Now, false).Verdict);
    }
}
