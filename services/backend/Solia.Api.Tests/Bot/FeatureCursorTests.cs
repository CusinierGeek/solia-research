using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Time.Testing;
using Solia.Api.Bot;

namespace Solia.Api.Tests.Bot;

/// <summary>The feed's id cursor: an id that commits after a higher one is still read, exactly once.</summary>
public class FeatureCursorTests
{
    private readonly FakeTimeProvider _time = new();

    private FeatureCursor Cursor() => new(_time, TimeSpan.FromSeconds(30), NullLogger.Instance);

    [Fact]
    public void An_id_skipped_because_it_committed_late_is_waited_for_and_read()
    {
        FeatureCursor cursor = Cursor();
        Assert.True(cursor.TryAccept(100));
        Assert.True(cursor.TryAccept(103)); // 101 and 102 still in flight

        Assert.Equal(103, cursor.MaxSeen);
        Assert.Equal([101L, 102L], cursor.Gaps.Order());

        Assert.True(cursor.TryAccept(102)); // committed later
        Assert.Equal([101L], cursor.Gaps);
        Assert.Equal(103, cursor.MaxSeen);
    }

    [Fact]
    public void An_id_already_read_is_rejected()
    {
        FeatureCursor cursor = Cursor();
        Assert.True(cursor.TryAccept(5));
        Assert.True(cursor.TryAccept(7));
        Assert.True(cursor.TryAccept(6));

        Assert.False(cursor.TryAccept(5));
        Assert.False(cursor.TryAccept(6));
        Assert.False(cursor.TryAccept(7));
        Assert.Empty(cursor.Gaps);
    }

    [Fact]
    public void The_first_id_read_leaves_no_gap_below_it()
    {
        FeatureCursor cursor = Cursor();
        Assert.Null(cursor.MaxSeen);

        Assert.True(cursor.TryAccept(1_000));

        Assert.Empty(cursor.Gaps);
    }

    [Fact]
    public void A_seeded_cursor_waits_for_the_ids_after_the_seed()
    {
        FeatureCursor cursor = Cursor();
        cursor.Seed(0); // empty table at startup

        Assert.True(cursor.TryAccept(2)); // 1 still in flight
        Assert.Equal([1L], cursor.Gaps);

        cursor.Seed(50); // only seeds a fresh cursor
        Assert.Equal(2, cursor.MaxSeen);
    }

    [Fact]
    public void A_missing_id_is_given_up_after_the_commit_delay()
    {
        FeatureCursor cursor = Cursor();
        cursor.TryAccept(1);
        cursor.TryAccept(3); // 2 rolled back: it never commits

        _time.Advance(TimeSpan.FromSeconds(29));
        cursor.ExpireGaps();
        Assert.Equal([2L], cursor.Gaps);

        cursor.TryAccept(5); // 4 missed later: its own delay
        _time.Advance(TimeSpan.FromSeconds(1));
        cursor.ExpireGaps();
        Assert.Equal([4L], cursor.Gaps);
    }

    [Fact]
    public void A_jump_wider_than_the_tracked_gap_is_skipped_not_tracked()
    {
        FeatureCursor cursor = Cursor();
        cursor.TryAccept(1);

        Assert.True(cursor.TryAccept(2 + FeatureCursor.MaxTrackedGap + 1));

        Assert.Empty(cursor.Gaps);
        Assert.Equal(2 + FeatureCursor.MaxTrackedGap + 1, cursor.MaxSeen);
    }
}
