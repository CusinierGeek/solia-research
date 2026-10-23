using Microsoft.Extensions.Time.Testing;
using Solia.Api.Bot;
using Solia.Bot.Domain;

namespace Solia.Api.Tests.Bot;

/// <summary>The feed's reorder buffer: events read out of time order reach the engine in time order.</summary>
public class ReorderBufferTests
{
    private static readonly DateTimeOffset T = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    private readonly FakeTimeProvider _time = new();

    private static BotEvent Tick(string mint, double seconds) =>
        new BotEvent.Price(new PriceTick { Mint = mint, Ts = T.AddSeconds(seconds), Price = 1m });

    [Fact]
    public void A_snapshot_written_a_poll_late_is_put_back_in_time_order()
    {
        var buffer = new ReorderBuffer(_time, TimeSpan.FromSeconds(5));
        buffer.Add(Tick("a", 10));
        Assert.Empty(buffer.Release()); // might still be overtaken

        buffer.Add(Tick("b", 3)); // next poll: an older snapshot of another token
        buffer.Add(Tick("a", 16)); // 5 s past a@10 and b@3
        List<BotEvent> due = buffer.Release();

        Assert.Equal([("b", T.AddSeconds(3)), ("a", T.AddSeconds(10))], due.Select(e => (e.Mint, e.Ts)));
        Assert.Equal(1, buffer.Count);
    }

    [Fact]
    public void A_quiet_feed_is_released_once_the_window_has_passed()
    {
        var buffer = new ReorderBuffer(_time, TimeSpan.FromSeconds(2));
        buffer.Add(Tick("a", 0));
        buffer.Add(Tick("a", 1));

        _time.Advance(TimeSpan.FromSeconds(1.9));
        Assert.Empty(buffer.Release());

        _time.Advance(TimeSpan.FromSeconds(0.1));
        Assert.Equal([T, T.AddSeconds(1)], buffer.Release().Select(e => e.Ts));
        Assert.Equal(0, buffer.Count);
    }

    [Fact]
    public void Waiting_out_the_window_also_releases_everything_older()
    {
        var buffer = new ReorderBuffer(_time, TimeSpan.FromSeconds(2));
        buffer.Add(Tick("a", 5));
        _time.Advance(TimeSpan.FromSeconds(2));
        buffer.Add(Tick("b", 4)); // just arrived, but older than one that is due

        Assert.Equal(["b", "a"], buffer.Release().Select(e => e.Mint));
    }

    [Fact]
    public void Events_at_the_same_time_keep_their_arrival_order()
    {
        var buffer = new ReorderBuffer(_time, TimeSpan.Zero);
        var prediction = new BotEvent.Prediction(new ModelPrediction
        {
            Mint = "a",
            Ts = T,
            Horizon = "1m",
            PumpProb = 0.5,
            RugProb = 0.1,
            ExpectedReturn = 0.1,
            Confidence = 60,
            ModelVersion = "v1",
        });
        buffer.Add(Tick("a", 0));
        buffer.Add(prediction);

        List<BotEvent> due = buffer.Release();

        Assert.IsType<BotEvent.Price>(due[0]);
        Assert.Same(prediction, due[1]);
    }

    [Fact]
    public void A_zero_window_releases_each_batch_at_once_in_time_order()
    {
        var buffer = new ReorderBuffer(_time, TimeSpan.Zero);
        buffer.Add(Tick("a", 2));
        buffer.Add(Tick("b", 1));

        Assert.Equal(["b", "a"], buffer.Release().Select(e => e.Mint));
        Assert.Empty(buffer.Release());
    }
}
