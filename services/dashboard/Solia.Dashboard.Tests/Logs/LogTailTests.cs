using Solia.Dashboard.Api;
using Solia.Dashboard.Logs;

namespace Solia.Dashboard.Tests.Logs;

public class LogTailTests
{
    private static readonly DateTimeOffset T0 = new(2026, 10, 27, 12, 0, 0, TimeSpan.Zero);

    private static LogDto Log(long id, int second = -1) =>
        new(id, T0.AddSeconds(second < 0 ? id : second), "bot", "info", $"entry {id}", null);

    private static long[] Ids(LogTail tail) => [.. tail.Rows.Select(l => l.Id)];

    [Fact]
    public void A_snapshot_is_held_newest_first_by_ts_then_id()
    {
        var tail = new LogTail();

        tail.Reset([Log(1), Log(3), Log(2), Log(5, second: 3), Log(4, second: 3)], capacity: 10);

        Assert.Equal([5, 4, 3, 2, 1], Ids(tail));
    }

    [Fact]
    public void A_snapshot_larger_than_the_capacity_keeps_the_newest()
    {
        var tail = new LogTail();

        tail.Reset([Log(4), Log(3), Log(2), Log(1)], capacity: 2);

        Assert.Equal(2, tail.Capacity);
        Assert.Equal([4, 3], Ids(tail));
    }

    [Fact]
    public void A_pushed_entry_goes_on_top_and_the_oldest_is_dropped_past_the_cap()
    {
        var tail = new LogTail();
        tail.Reset([Log(3), Log(2), Log(1)], capacity: 3);

        Assert.True(tail.Add(Log(4)));
        Assert.True(tail.Add(Log(5)));

        Assert.Equal([5, 4, 3], Ids(tail));
    }

    [Fact]
    public void A_late_entry_is_put_in_its_place()
    {
        var tail = new LogTail();
        tail.Reset([Log(5), Log(3)], capacity: 5);

        Assert.True(tail.Add(Log(4)));

        Assert.Equal([5, 4, 3], Ids(tail));
    }

    [Fact]
    public void An_entry_older_than_a_full_tail_is_not_added()
    {
        var tail = new LogTail();
        tail.Reset([Log(5), Log(4)], capacity: 2);

        Assert.False(tail.Add(Log(1)));

        Assert.Equal([5, 4], Ids(tail));
    }

    [Fact]
    public void An_entry_already_held_is_not_added_twice()
    {
        var tail = new LogTail();
        tail.Reset([Log(2), Log(1)], capacity: 5);

        Assert.False(tail.Add(Log(2)));

        Assert.Equal([2, 1], Ids(tail));
    }

    [Fact]
    public void A_dropped_entry_can_come_back()
    {
        var tail = new LogTail();
        tail.Reset([Log(2), Log(1)], capacity: 2);
        tail.Add(Log(3)); // drops 1

        tail.Reset([Log(1)], capacity: 2);

        Assert.Equal([1], Ids(tail));
    }

    [Fact]
    public void Reset_replaces_everything_and_the_capacity()
    {
        var tail = new LogTail();
        tail.Reset([Log(2), Log(1)], capacity: 2);

        tail.Reset([Log(9), Log(8), Log(7)], capacity: 3);

        Assert.Equal([9, 8, 7], Ids(tail));
        Assert.True(tail.Add(Log(10)));
        Assert.Equal([10, 9, 8], Ids(tail));
    }

    [Fact]
    public void A_snapshot_holds_an_entry_once()
    {
        var tail = new LogTail();

        tail.Reset([Log(2), Log(1), Log(2)], capacity: 10);

        Assert.Equal([2, 1], Ids(tail));
        Assert.True(tail.Contains(1));
        Assert.False(tail.Contains(3));
    }

    [Fact]
    public void An_entry_dropped_past_the_cap_is_no_longer_held()
    {
        var tail = new LogTail(capacity: 2);
        tail.Reset([Log(2), Log(1)], capacity: 2);

        tail.Add(Log(3));

        Assert.False(tail.Contains(1));
        Assert.True(tail.Contains(3));
    }

    [Fact]
    public void Rejects_a_capacity_that_is_not_positive()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LogTail(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new LogTail().Reset([], capacity: 0));
    }
}
