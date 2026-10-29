using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

public class RetryPolicyTests
{
    private static readonly RetryPolicy Three = new(3);

    [Fact]
    public void Delays_grow_until_the_last_attempt_then_give_up()
    {
        TimeSpan first = Three.DelayAfter(1)!.Value;
        TimeSpan second = Three.DelayAfter(2)!.Value;

        Assert.InRange(first.TotalSeconds, 0.5, 1.5); // base 1 s ± jitter
        Assert.InRange(second.TotalSeconds, 1, 3);
        Assert.Null(Three.DelayAfter(3));
        Assert.True(new RetryPolicy(10).DelayAfter(9)!.Value <= TimeSpan.FromSeconds(30 * 1.5)); // capped
    }

    [Fact]
    public async Task Succeeds_after_transient_failures()
    {
        int calls = 0;
        var waits = new List<TimeSpan>();
        var seen = new List<(int Attempt, bool GivingUp)>();

        bool ok = await Three.RunAsync(
            _ => ++calls < 3 ? throw new IOException("blip") : Task.CompletedTask,
            (d, _) => { waits.Add(d); return Task.CompletedTask; },
            (_, attempt, wait) => seen.Add((attempt, wait is null)),
            default);

        Assert.True(ok);
        Assert.Equal(3, calls);
        Assert.Equal(2, waits.Count);
        Assert.Equal(new[] { (1, false), (2, false) }, seen);
    }

    [Fact]
    public async Task Gives_up_after_the_last_attempt_without_throwing()
    {
        int calls = 0;
        var seen = new List<(int Attempt, bool GivingUp)>();

        bool ok = await Three.RunAsync(
            _ => { calls++; throw new IOException("down"); },
            (_, _) => Task.CompletedTask,
            (_, attempt, wait) => seen.Add((attempt, wait is null)),
            default);

        Assert.False(ok);
        Assert.Equal(3, calls);
        Assert.Equal((3, true), seen[^1]);
    }

    [Fact]
    public async Task Caller_cancellation_is_not_retried()
    {
        using var cts = new CancellationTokenSource();
        int calls = 0;

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Three.RunAsync(
            async ct => { calls++; await cts.CancelAsync(); ct.ThrowIfCancellationRequested(); },
            (_, _) => Task.CompletedTask, (_, _, _) => { }, cts.Token));

        Assert.Equal(1, calls);
    }

    [Fact]
    public async Task Invalid_attempt_count_is_rejected()
    {
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            new RetryPolicy(0).RunAsync(_ => Task.CompletedTask, (_, _) => Task.CompletedTask, (_, _, _) => { }, default));
    }
}
