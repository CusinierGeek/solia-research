using Microsoft.Extensions.Logging;
using Solia.Bot.Domain;
using Solia.Bot.Policy;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

/// <summary>#219: a prediction the model said buy on, but the bot did not enter, is logged with why.</summary>
public class DeclinedEntryLogTests
{
    private const string Mint = "mint1";
    private const decimal Entry = 0.0001m;
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private readonly InMemoryTradeStore _store = new();
    private readonly MessageLogger<PositionManager> _logger = new();

    private PositionManager Bot() => new(PolicyFixtures.Options(), _store, logger: _logger);

    private static BotEvent Tick(int sec, decimal price) =>
        new BotEvent.Price(new PriceTick { Mint = Mint, Ts = T0.AddSeconds(sec), Price = price });

    private static BotEvent Predict(int sec, ModelPrediction p) =>
        new BotEvent.Prediction(p with { Mint = Mint, Ts = T0.AddSeconds(sec) });

    private static async Task Run(PositionManager bot, params BotEvent[] events)
    {
        foreach (BotEvent e in events)
        {
            await bot.HandleAsync(e);
        }
    }

    [Fact]
    public async Task A_buy_signal_with_no_market_price_is_logged_with_its_reason_and_heads()
    {
        await Run(Bot(), Predict(1, PolicyFixtures.Buyable()));

        string line = Assert.Single(_logger.Messages);
        Assert.Contains($"entry declined: {EntryRejectReason.NoMarketPrice} for {Mint}", line);
        Assert.Contains("pump ", line);
        Assert.Contains("rug ", line);
        Assert.Contains("confidence ", line);
        Assert.Contains("expected return ", line);
    }

    [Fact]
    public async Task A_buy_signal_without_upside_is_logged_as_no_positive_edge()
    {
        await Run(Bot(), Tick(0, Entry), Predict(1, PolicyFixtures.Buyable() with { ExpectedReturn = 0.01 }));

        Assert.Contains(_logger.Messages, m => m.Contains($"entry declined: {EntryRejectReason.NoPositiveEdge}"));
    }

    [Fact]
    public async Task A_prediction_that_is_not_a_buy_is_not_logged()
    {
        await Run(Bot(), Tick(0, Entry), Predict(1, PolicyFixtures.Buyable() with { PumpProb = 0.01 }));

        Assert.Empty(_logger.Messages);
    }

    [Fact]
    public async Task An_accepted_entry_is_not_logged_as_declined()
    {
        PositionManager bot = Bot();
        await Run(bot, Tick(0, Entry), Predict(1, PolicyFixtures.Buyable()));

        Assert.Single(bot.OpenPositions);
        Assert.DoesNotContain(_logger.Messages, m => m.Contains("entry declined"));
    }
}

/// <summary>An <see cref="ILogger{T}"/> that keeps each formatted message.</summary>
internal sealed class MessageLogger<T> : ILogger<T>
{
    private readonly List<string> _messages = [];

    public IReadOnlyList<string> Messages
    {
        get
        {
            lock (_messages)
            {
                return [.. _messages];
            }
        }
    }

    public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel logLevel) => true;

    public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
    {
        lock (_messages)
        {
            _messages.Add(formatter(state, exception));
        }
    }
}
