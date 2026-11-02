using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion;

namespace Solia.DataPipeline.Tests.Features;

/// <summary>FE-T12 (#196): name / ticker normalization and narrative heat.</summary>
public class NarrativeTrackerTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private const int Hour = 3600;
    private const int Full = 6 * Hour; // NarrativeTracker.FullCoverage

    private static RawEvent Create(string mint, double offsetSec, string? name, string? symbol) => new()
    {
        Mint = mint,
        Kind = EventKind.Create,
        Timestamp = T0.AddSeconds(offsetSec),
        Name = name,
        Symbol = symbol,
    };

    private static RawEvent Migrate(string mint, double offsetSec) => new()
    {
        Mint = mint,
        Kind = EventKind.Migrate,
        Timestamp = T0.AddSeconds(offsetSec),
    };

    private static DateTimeOffset At(double offsetSec) => T0.AddSeconds(offsetSec);

    /// <summary>Heartbeats every 30 s from <paramref name="from"/> to <paramref name="to"/>, so the event clock follows time.</summary>
    private static void Walk(NarrativeTracker tracker, double from, double to)
    {
        for (double t = from; t < to; t += 30)
        {
            tracker.Apply(new RawEvent { Mint = "TICK", Kind = EventKind.Other, Timestamp = At(t) });
        }

        tracker.Apply(new RawEvent { Mint = "TICK", Kind = EventKind.Other, Timestamp = At(to) });
    }

    /// <summary>Walks the clock to each event's time before applying it, like a real stream.</summary>
    private static NarrativeTracker Feed(double from, params RawEvent[] events)
    {
        var tracker = new NarrativeTracker();
        double clock = from;
        Walk(tracker, from, from);
        foreach (RawEvent e in events.OrderBy(e => e.Timestamp))
        {
            double t = (e.Timestamp - T0).TotalSeconds;
            if (t > clock)
            {
                Walk(tracker, clock, t);
                clock = t;
            }

            tracker.Apply(e);
        }

        return tracker;
    }

    [Theory]
    [InlineData("$PEPE", "pepe")]
    [InlineData("Pépé", "pepe")]
    [InlineData("pepe 🐸", "pepe")]
    [InlineData("ＰＥＰＥ", "pepe")] // full-width
    [InlineData("Super Agent", "superagent")]
    [InlineData("咪咪", "咪咪")]
    [InlineData("I'm In U", "iminu")]
    [InlineData("X", null)] // too short to match on
    [InlineData("🐸🐸", null)]
    [InlineData("   ", null)]
    [InlineData(null, null)]
    public void Names_are_normalized_before_matching(string? text, string? expected)
    {
        Assert.Equal(expected, NarrativeKeys.Normalize(text));
    }

    [Fact]
    public void A_token_has_its_distinct_name_and_ticker_keys()
    {
        Assert.Equal(["pepe"], NarrativeKeys.Of("Pepe", "PEPE"));
        Assert.Equal(["superagent", "sa"], NarrativeKeys.Of("Super Agent", "SA"));
        Assert.Equal(["sa"], NarrativeKeys.Of("🐸", "SA"));
        Assert.Empty(NarrativeKeys.Of(null, "S"));
    }

    [Fact]
    public void Counts_other_launches_of_the_last_hour_sharing_a_name_or_ticker_once_each()
    {
        double ts = Full + Hour;
        NarrativeTracker tracker = Feed(0,
            Create("SELF", ts - 600, "Pepe", "PEPE"),
            Create("BY_TICKER", ts - 900, "Frog", "pepe"),
            Create("BY_NAME", ts - 1200, "pepe!", "PPP"),
            Create("BOTH_KEYS", ts - 1500, "PEPE", "Pepe"),
            Create("TOO_OLD", ts - Hour - 1, "Pepe", "PEPE"),
            Create("UNRELATED", ts - 300, "Doge", "DOGE"),
            Create("SAME_SECOND", ts, "Pepe", "PEPE")); // not strictly before the snapshot's second

        (int coverage, int? matches, _) = tracker.Count(NarrativeKeys.Of("Pepe", "PEPE"), "SELF", At(ts));

        Assert.Equal(Full, coverage);
        Assert.Equal(3, matches); // BY_TICKER, BY_NAME, BOTH_KEYS; never SELF
    }

    [Fact]
    public void Copycats_launched_after_the_token_count_once_they_happened()
    {
        double created = Full + Hour;
        NarrativeTracker tracker = Feed(0, Create("SELF", created, "Pepe", "PEPE"), Create("COPY", created + 60, "Pepe 2", "PEPE"));
        string[] keys = NarrativeKeys.Of("Pepe", "PEPE");

        Assert.Equal(0, tracker.Count(keys, "SELF", At(created + 60)).Matches);
        Assert.Equal(1, tracker.Count(keys, "SELF", At(created + 61)).Matches);
    }

    [Fact]
    public void Graduated_namesakes_count_within_three_hours_once_each_and_never_the_token_itself()
    {
        double ts = Full + Hour;
        NarrativeTracker tracker = Feed(0,
            Create("GRAD", ts - (2 * Hour), "Pepe", "PEPE"),
            Migrate("GRAD", ts - Hour),
            Migrate("GRAD", ts - 1800), // a second migration of the same token
            Create("OLD_GRAD", ts - (5 * Hour), "Pepe", "OG"),
            Migrate("OLD_GRAD", ts - (3 * Hour) - 1), // graduated before the window
            Create("SLOW", ts - (5 * Hour) - 1800, "Pepe", "SLOW"),
            Migrate("SLOW", ts - 1200), // created more than CreationMemory before graduating
            Create("SELF", ts - (2 * Hour), "PEPE", "PEPE"),
            Migrate("SELF", ts - 600),
            Create("OTHER", ts - Hour, "Doge", "DOGE"),
            Migrate("OTHER", ts - 300));

        Assert.Equal(1, tracker.Count(NarrativeKeys.Of("PEPE", "PEPE"), "SELF", At(ts)).Graduations); // GRAD only
    }

    [Fact]
    public void Counts_stay_null_until_their_history_is_covered()
    {
        string[] keys = NarrativeKeys.Of("Pepe", "PEPE");
        NarrativeTracker tracker = Feed(0, Create("A", 60, "Pepe", "PEPE"));
        Walk(tracker, 60, Full);

        Assert.Equal((Hour - 1, null, null), tracker.Count(keys, "SELF", At(Hour - 1)));
        Assert.Equal((Hour, 1, null), tracker.Count(keys, "SELF", At(Hour)));
        Assert.Equal((Full, 0, 0), tracker.Count(keys, "SELF", At(Full)));
        Assert.Equal((Full, null, null), tracker.Count([], "SELF", At(Full))); // no usable name / ticker
    }

    [Fact]
    public void Pruning_never_changes_a_count()
    {
        double ts = Full + (2 * Hour);
        NarrativeTracker tracker = Feed(0,
            Create("A", ts - 1800, "Pepe", "PEPE"),
            Create("G", ts - (4 * Hour), "Pepe", "G"),
            Migrate("G", ts - (2 * Hour)),
            Create("STALE", ts - (5 * Hour), "Pepe", "S"));
        string[] keys = NarrativeKeys.Of("Pepe", "PEPE");
        var before = tracker.Count(keys, "SELF", At(ts));

        tracker.Prune(At(ts));

        Assert.Equal(before, tracker.Count(keys, "SELF", At(ts)));
        Assert.Equal((Full, 1, 1), before);
        Assert.Equal(1, tracker.RememberedCreations); // only A is still within the creation memory
    }
}
