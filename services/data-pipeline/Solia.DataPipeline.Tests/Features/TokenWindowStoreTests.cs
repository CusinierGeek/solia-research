using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests.Features;

public class TokenWindowStoreTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    private static PipelineOptions Opts(int max = 1000, int window = 300) =>
        new() { MaxTrackedTokens = max, WindowSeconds = window };

    private static RawEvent Ev(string mint, EventKind kind, int offsetSec, string? trader = null) =>
        new()
        {
            Mint = mint,
            Kind = kind,
            Timestamp = T0.AddSeconds(offsetSec),
            Venue = "pumpfun",
            Price = 1m,
            QuoteAmount = 1m,
            Trader = trader,
        };

    [Fact]
    public void Transfer_only_token_is_not_tracked()
    {
        var store = new TokenWindowStore(Opts());

        store.Apply(Ev("USDC", EventKind.Transfer, 0, "x"));

        Assert.Equal(0, store.TrackedCount);
    }

    [Fact]
    public void Market_event_starts_tracking_and_transfer_updates_existing()
    {
        var store = new TokenWindowStore(Opts());

        store.Apply(Ev("M", EventKind.Buy, 0, "A"));
        store.Apply(Ev("M", EventKind.Transfer, 1, "B")); // updates the already-tracked token

        Assert.Equal(1, store.TrackedCount);
        FeatureSnapshot s = Assert.Single(store.SnapshotActive(T0.AddSeconds(1)));
        Assert.Equal(2, s.Holders); // A + B
    }

    [Fact]
    public void Eviction_keeps_within_max_tracked()
    {
        var store = new TokenWindowStore(Opts(max: 2));

        store.Apply(Ev("A", EventKind.Buy, 0));
        store.Apply(Ev("B", EventKind.Buy, 1));
        store.Apply(Ev("C", EventKind.Buy, 2)); // evicts the least-recently-active (A)

        Assert.Equal(2, store.TrackedCount);
    }

    [Fact]
    public void Other_events_do_not_start_tracking()
    {
        var store = new TokenWindowStore(Opts());

        store.Apply(Ev("X", EventKind.Other, 0));

        Assert.Equal(0, store.TrackedCount);
    }

    [Theory]
    [InlineData(EventKind.Create)]
    [InlineData(EventKind.Sell)]
    [InlineData(EventKind.Migrate)]
    [InlineData(EventKind.CreatePool)]
    [InlineData(EventKind.AddLiquidity)]
    [InlineData(EventKind.RemoveLiquidity)]
    public void Market_events_start_tracking(EventKind kind)
    {
        var store = new TokenWindowStore(Opts());

        store.Apply(Ev("M", kind, 0));

        Assert.Equal(1, store.TrackedCount);
    }

    [Fact]
    public void Eviction_removes_the_least_recently_updated_token()
    {
        var store = new TokenWindowStore(Opts(max: 2));
        store.Apply(Ev("A", EventKind.Buy, 0));
        store.Apply(Ev("B", EventKind.Buy, 1));
        store.Apply(Ev("A", EventKind.Buy, 2)); // A is now the most recently updated

        store.Apply(Ev("C", EventKind.Buy, 3)); // evicts B

        Assert.Equal(new[] { "A", "C" }, store.SnapshotActive(T0.AddSeconds(3)).Select(s => s.Mint).Order());
    }

    [Fact]
    public void At_capacity_idle_tokens_are_dropped_before_a_live_one_is_evicted()
    {
        var store = new TokenWindowStore(Opts(max: 3, window: 60));
        store.Apply(Ev("OLD1", EventKind.Buy, 0));
        store.Apply(Ev("OLD2", EventKind.Buy, 1));
        store.Apply(Ev("LIVE", EventKind.Buy, 100));

        store.Apply(Ev("NEW", EventKind.Buy, 110)); // both OLD tokens are idle beyond the window

        Assert.Equal(2, store.TrackedCount);
        Assert.Equal(new[] { "LIVE", "NEW" }, store.SnapshotActive(T0.AddSeconds(110)).Select(s => s.Mint).Order());
    }

    [Fact]
    public void Snapshot_runs_once_per_call_so_growth_rates_survive_a_second_look()
    {
        var store = new TokenWindowStore(Opts());
        store.Apply(Ev("M", EventKind.Buy, 0, "A"));
        store.SnapshotActive(T0);
        store.Apply(Ev("M", EventKind.Buy, 10, "B"));

        IReadOnlyList<FeatureSnapshot> snapshots = store.SnapshotActive(T0.AddSeconds(10));

        Assert.Single(snapshots);       // inspecting the result...
        FeatureSnapshot s = snapshots[0]; // ...doesn't re-snapshot and zero the growth rates
        Assert.Equal(0.1, s.HoldersGrowthRate!.Value, 10);
        Assert.Equal(0.1, s.VolumeGrowthRate!.Value, 10);
    }

    [Fact]
    public void Future_dated_event_cannot_pin_a_token()
    {
        var store = new TokenWindowStore(Opts(window: 60));
        store.Apply(Ev("M", EventKind.Buy, 0));
        store.Apply(Ev("M", EventKind.Buy, 86_400)); // a day ahead of the stream

        Assert.Empty(store.SnapshotActive(T0.AddSeconds(120)));
        Assert.Equal(0, store.TrackedCount);

        store.Apply(Ev("M", EventKind.Buy, 130)); // the next real event re-tracks it cleanly
        Assert.Equal(1, Assert.Single(store.SnapshotActive(T0.AddSeconds(130))).BuyEvents);
    }

    [Fact]
    public void Small_clock_skew_is_kept()
    {
        var store = new TokenWindowStore(Opts());
        store.Apply(Ev("M", EventKind.Buy, 30)); // slightly ahead of the snapshot time, within MaxFutureSkew

        store.Prune(T0);

        Assert.Equal(1, store.TrackedCount);
    }

    [Fact]
    public void A_gap_in_the_stream_does_not_block_later_events()
    {
        var store = new TokenWindowStore(Opts());
        store.Apply(Ev("A", EventKind.Buy, 0));
        store.Apply(Ev("B", EventKind.Buy, 3_600)); // e.g. a missing replay hour

        Assert.Equal("B", Assert.Single(store.SnapshotActive(T0.AddSeconds(3_600))).Mint);
    }

    [Fact]
    public void Token_holding_a_far_future_timestamp_is_pruned()
    {
        var store = new TokenWindowStore(Opts(window: 60));
        store.Apply(Ev("BOGUS", EventKind.Buy, 86_400)); // first event ever: nothing to compare against yet

        store.Prune(T0);

        Assert.Equal(0, store.TrackedCount);
    }

    [Fact]
    public void Removed_tokens_are_reported_once_and_re_tracked_ones_are_not()
    {
        var store = new TokenWindowStore(Opts(window: 60));
        store.Apply(Ev("A", EventKind.Buy, 0));
        store.Apply(Ev("B", EventKind.Buy, 0));
        store.Prune(T0.AddSeconds(120)); // both idle -> removed
        store.Apply(Ev("B", EventKind.Buy, 121)); // B comes back

        Assert.Equal(new[] { "A" }, store.DrainRemoved());
        Assert.Empty(store.DrainRemoved());
    }

    [Fact]
    public void Evicted_tokens_are_reported()
    {
        var store = new TokenWindowStore(Opts(max: 1));
        store.Apply(Ev("A", EventKind.Buy, 0));
        store.Apply(Ev("B", EventKind.Buy, 1));

        Assert.Equal(new[] { "A" }, store.DrainRemoved());
    }

    [Fact]
    public void Idle_tokens_are_pruned_and_not_snapshotted()
    {
        var store = new TokenWindowStore(Opts(window: 60));
        store.Apply(Ev("M", EventKind.Buy, 0));

        List<FeatureSnapshot> snapshots = store.SnapshotActive(T0.AddSeconds(120)).ToList();

        Assert.Empty(snapshots);
        Assert.Equal(0, store.TrackedCount); // idle beyond the window -> pruned
    }
}
