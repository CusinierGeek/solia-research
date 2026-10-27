using Solia.Dashboard.Api;
using Solia.Dashboard.Predictions;

namespace Solia.Dashboard.Tests.Predictions;

public class LivePredictionBoardTests
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private static PredictionRecordDto P(long id, string mint, int ts, string horizon = "1m", double? pump = 0.5, double? rug = 0.1, double? confidence = 50) =>
        new(id, mint, null, T0.AddSeconds(ts), horizon, pump, rug, "hold", confidence, null, null, null, "v1");

    private static string[] Mints(IEnumerable<PredictionRecordDto> rows) => rows.Select(r => r.Mint).ToArray();

    [Fact]
    public void Keeps_one_row_per_mint_and_horizon()
    {
        var board = new LivePredictionBoard(10);

        Assert.True(board.Upsert(P(1, "m1", 0, "1m")));
        Assert.True(board.Upsert(P(2, "m1", 0, "5m")));
        Assert.True(board.Upsert(P(3, "m1", 1, "1m")));

        Assert.Equal(2, board.Count);
        Assert.Contains(board.Sorted(PredictionSort.Ts), r => r.Id == 3);
        Assert.DoesNotContain(board.Sorted(PredictionSort.Ts), r => r.Id == 1);
    }

    [Fact]
    public void Ignores_an_older_or_duplicate_prediction()
    {
        var board = new LivePredictionBoard(10);
        board.Upsert(P(5, "m1", 10));

        Assert.False(board.Upsert(P(4, "m1", 5)));   // older ts
        Assert.False(board.Upsert(P(5, "m1", 10)));  // duplicate
        Assert.False(board.Upsert(P(3, "m1", 10)));  // same ts, lower id
        Assert.True(board.Upsert(P(6, "m1", 10)));   // same ts, later id

        Assert.Equal(6, Assert.Single(board.Sorted(PredictionSort.Ts)).Id);
    }

    [Fact]
    public void Drops_the_oldest_row_past_capacity()
    {
        var board = new LivePredictionBoard(2);
        board.Upsert(P(1, "old", 0));
        board.Upsert(P(2, "mid", 10));

        Assert.True(board.Upsert(P(3, "new", 20)));
        Assert.Equal(["new", "mid"], Mints(board.Sorted(PredictionSort.Ts)));

        Assert.False(board.Upsert(P(4, "ancient", -5))); // dropped straight away
        Assert.Equal(2, board.Count);
    }

    [Fact]
    public void Reset_replaces_every_row()
    {
        var board = new LivePredictionBoard(10);
        board.Upsert(P(1, "gone", 0));

        board.Reset([P(2, "m1", 0), P(3, "m2", 1)]);

        Assert.Equal(["m2", "m1"], Mints(board.Sorted(PredictionSort.Ts)));
    }

    [Fact]
    public void Sorts_by_value_with_missing_values_last_in_both_directions()
    {
        var board = new LivePredictionBoard(10);
        board.Reset([P(1, "low", 0, rug: 0.1), P(2, "none", 5, rug: null), P(3, "high", 1, rug: 0.9)]);

        Assert.Equal(["high", "low", "none"], Mints(board.Sorted(PredictionSort.RugProb)));
        Assert.Equal(["low", "high", "none"], Mints(board.Sorted(PredictionSort.RugProb, descending: false)));
    }

    [Fact]
    public void Ties_are_newest_first()
    {
        var board = new LivePredictionBoard(10);
        board.Reset([P(1, "older", 0, confidence: 70), P(2, "newer", 9, confidence: 70)]);

        Assert.Equal(["newer", "older"], Mints(board.Sorted(PredictionSort.Confidence)));
        Assert.Equal(["older", "newer"], Mints(board.Sorted(PredictionSort.Ts, descending: false)));
    }

    [Fact]
    public void Eviction_follows_the_latest_ts_of_each_row()
    {
        var board = new LivePredictionBoard(2);
        board.Upsert(P(1, "a", 0));
        board.Upsert(P(2, "b", 10));
        board.Upsert(P(3, "a", 20)); // "a" is now the newest

        Assert.True(board.Upsert(P(4, "c", 15)));

        Assert.Equal(["a", "c"], Mints(board.Sorted(PredictionSort.Ts)));
    }

    [Fact]
    public void Same_ts_rows_are_evicted_lowest_id_first()
    {
        var board = new LivePredictionBoard(2);
        board.Upsert(P(2, "b", 0));
        board.Upsert(P(1, "a", 0));

        Assert.True(board.Upsert(P(3, "c", 0)));

        Assert.Equal(["c", "b"], Mints(board.Sorted(PredictionSort.Ts)));
    }

    [Fact]
    public void Sorted_can_keep_only_one_horizon()
    {
        var board = new LivePredictionBoard(10);
        board.Reset([P(1, "a", 0, "1m"), P(2, "b", 5, "5m"), P(3, "c", 9, "5m")]);

        Assert.Equal(["c", "b"], Mints(board.Sorted(PredictionSort.Ts, horizon: "5m")));
        Assert.Empty(board.Sorted(PredictionSort.Ts, horizon: "1h"));
        Assert.Equal(3, board.Sorted(PredictionSort.Ts).Count);
    }

    [Fact]
    public void Sorted_reuses_its_result_until_a_row_changes()
    {
        var board = new LivePredictionBoard(10);
        board.Upsert(P(1, "a", 0));
        IReadOnlyList<PredictionRecordDto> first = board.Sorted(PredictionSort.Ts);

        Assert.Same(first, board.Sorted(PredictionSort.Ts));
        Assert.NotSame(first, board.Sorted(PredictionSort.PumpProb));

        IReadOnlyList<PredictionRecordDto> before = board.Sorted(PredictionSort.Ts);
        board.Upsert(P(0, "a", -1)); // ignored: older
        Assert.Same(before, board.Sorted(PredictionSort.Ts));

        board.Upsert(P(2, "b", 1));
        Assert.Equal(["b", "a"], Mints(board.Sorted(PredictionSort.Ts)));

        board.Reset([]);
        Assert.Empty(board.Sorted(PredictionSort.Ts));
    }

    [Fact]
    public void Rejects_a_capacity_below_one() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new LivePredictionBoard(0));
}
