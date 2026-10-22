using Solia.Bot.Domain;
using Solia.Bot.Intake;
using Solia.Bot.Tests.Policy;

namespace Solia.Bot.Tests.Intake;

public class InMemoryBotEventSourceTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static BotEvent Tick(int sec) =>
        new BotEvent.Price(new PriceTick { Mint = "m", Ts = T0.AddSeconds(sec), Price = 1m });

    private static BotEvent Predict(int sec) =>
        new BotEvent.Prediction(PolicyFixtures.Buyable() with { Ts = T0.AddSeconds(sec) });

    [Fact]
    public async Task Replays_the_events_in_order()
    {
        BotEvent[] script = [Tick(0), Predict(1), Tick(2)];
        var source = new InMemoryBotEventSource(script);

        var read = new List<BotEvent>();
        await foreach (BotEvent e in source.ReadAsync())
        {
            read.Add(e);
        }

        Assert.Equal(script, read);
    }

    [Fact]
    public async Task The_empty_source_completes_without_yielding()
    {
        var read = new List<BotEvent>();
        await foreach (BotEvent e in InMemoryBotEventSource.Empty.ReadAsync())
        {
            read.Add(e);
        }

        Assert.Empty(read);
    }

    [Fact]
    public async Task Cancellation_stops_the_stream_between_events()
    {
        var source = new InMemoryBotEventSource([Tick(0), Tick(1), Tick(2)]);
        using var cts = new CancellationTokenSource();

        var read = new List<BotEvent>();
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
        {
            await foreach (BotEvent e in source.ReadAsync(cts.Token))
            {
                read.Add(e);
                cts.Cancel();
            }
        });

        Assert.Single(read);
    }

    [Fact]
    public void A_null_sequence_is_rejected() =>
        Assert.Throws<ArgumentNullException>(() => new InMemoryBotEventSource(null!));
}
