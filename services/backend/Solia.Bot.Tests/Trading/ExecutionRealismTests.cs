using Solia.Bot.Backtest;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Logging;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;
using Solia.Shared.Logging;

namespace Solia.Bot.Tests.Trading;

/// <summary>
/// Execution realism (#175): latency, network costs and failed fills. Idealized pool fills unless stated, so the
/// numbers stay round: a <see cref="PolicyFixtures.Buyable"/> prediction sizes to 20 % of the balance and takes TP at
/// +50 %, SL at −30 % of its fill mark. Every knob not under test is 0 (<see cref="PolicyFixtures.WithoutExecution"/>).
/// </summary>
public class ExecutionRealismTests
{
    private const string Mint = "mint1";
    private const decimal Entry = 0.0001m;
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private readonly InMemoryTradeStore _store = new();

    private static BotEvent Tick(double sec, decimal price, string mint = Mint) =>
        new BotEvent.Price(new PriceTick { Mint = mint, Ts = T0.AddSeconds(sec), Price = price });

    private static BotEvent PoolTick(double sec, decimal price, string venue, decimal liquidity, string mint = Mint) =>
        new BotEvent.Price(new PriceTick { Mint = mint, Ts = T0.AddSeconds(sec), Price = price, Venue = venue, Liquidity = liquidity });

    private static BotEvent Predict(double sec, ModelPrediction? p = null, string mint = Mint) =>
        new BotEvent.Prediction((p ?? PolicyFixtures.Buyable()) with { Mint = mint, Ts = T0.AddSeconds(sec) });

    private static ModelPrediction SellSignal() => PolicyFixtures.Buyable() with { PumpProb = 0.1, RugProb = 0.6 };

    private static BotOptions Opts(Action<BotOptions> configure, bool realistic = false)
    {
        BotOptions options = realistic ? PolicyFixtures.RealisticOptions() : PolicyFixtures.Options();
        configure(options);
        return options;
    }

    private PositionManager Bot(Action<BotOptions> configure, bool realistic = false, IBotLogSink? sink = null) =>
        new(Opts(configure, realistic), _store, sink);

    private static async Task Run(PositionManager bot, params BotEvent[] events)
    {
        foreach (BotEvent e in events)
        {
            await bot.HandleAsync(e);
        }
    }

    // --- Entry latency ---

    [Theory]
    [InlineData(0.00012, 0.00018, 0.000084)] // the price ran up while the order was in flight
    [InlineData(0.00008, 0.00012, 0.000056)] // … or down
    public async Task An_entry_fills_at_the_first_mark_after_its_latency_with_levels_from_that_mark(
        double fillMark, double tp, double sl)
    {
        PositionManager bot = Bot(o => o.EntryLatencyMs = 3_000);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(2, 0.0002m)); // due at 4 s: the 2 s mark is too early

        Assert.Empty(bot.OpenPositions);
        Assert.Equal(1, bot.PendingEntryCount);

        await Run(bot, Tick(4, (decimal)fillMark)); // exactly at the due time

        OpenPosition p = Assert.Single(bot.OpenPositions.Values);
        Assert.Equal((decimal)fillMark, p.Entry.EntryPrice);
        Assert.Equal(T0.AddSeconds(4), p.Entry.OpenedAt);
        Assert.Equal(T0.AddSeconds(904), p.Deadline);
        Assert.Equal((decimal)tp, p.Entry.Levels.TakeProfitPrice);
        Assert.Equal((decimal)sl, p.Entry.Levels.StopLossPrice);
        Assert.Equal(2m, p.Entry.Size); // sized at decision time
        Assert.Equal(8m, bot.Balance);
        Assert.Equal(0, bot.PendingEntryCount);
        Assert.Same(p.Entry, _store.Open[p.TradeId]);
    }

    [Fact]
    public async Task A_pending_entry_holds_its_capital_and_its_slot()
    {
        PositionManager bot = Bot(o =>
        {
            o.EntryLatencyMs = 3_000;
            o.MaxConcurrentPositions = 1;
        });
        await Run(bot, Tick(0, Entry), Tick(0, Entry, mint: "b"), Predict(1), Predict(2, mint: "b"), Predict(2));

        Assert.Equal(1, bot.PendingEntryCount); // b found no slot, the repeat buy no second order
        Assert.Equal(8m, bot.Balance);
        Assert.Equal(10m, bot.Equity); // the reserved size is still the bot's

        await Run(bot, Tick(4, Entry), Tick(4, Entry, mint: "b"));

        Assert.Equal([Mint], bot.OpenPositions.Keys);
        Assert.Single(_store.Open);
        Assert.Equal(8m, bot.Balance); // reserved once, not again at the fill
    }

    [Fact]
    public async Task An_entry_with_no_mark_after_its_latency_is_abandoned_without_locking_capital()
    {
        var sink = new MessageSink();
        PositionManager bot = Bot(o => o.EntryLatencyMs = 3_000, sink: sink);
        await Run(bot, Tick(0, Entry), Predict(1));

        await bot.AdvanceToAsync(T0.AddSeconds(64)); // due 4 s + 60 s of mark age: still within the window
        Assert.Equal(1, bot.PendingEntryCount);

        await bot.AdvanceToAsync(T0.AddSeconds(64.001));

        Assert.Equal(0, bot.PendingEntryCount);
        Assert.Equal(1, bot.AbandonedEntryCount);
        Assert.Empty(bot.OpenPositions);
        Assert.Empty(_store.Open);
        Assert.Equal(10m, bot.Balance);
        Assert.Equal(0m, bot.RealizedPnl);
        Assert.Contains("entry abandoned", sink.Messages);

        await Run(bot, Tick(70, Entry), Predict(71), Tick(74, Entry)); // the mint can be entered again
        Assert.Single(bot.OpenPositions);
    }

    [Fact]
    public async Task A_mark_at_the_end_of_the_window_still_fills_the_entry()
    {
        PositionManager bot = Bot(o => o.EntryLatencyMs = 3_000);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(64, Entry));
        Assert.Single(bot.OpenPositions);
    }

    [Fact]
    public async Task A_store_failure_on_a_pending_fill_is_resumed_by_redelivering_the_tick()
    {
        var store = new FailOnceOpenStore();
        var bot = new PositionManager(Opts(o => o.EntryLatencyMs = 3_000), store);
        await Run(bot, Tick(0, Entry), Predict(1));

        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.HandleAsync(Tick(4, 0.00011m)));
        Assert.Equal(1, bot.PendingEntryCount);
        Assert.Equal(8m, bot.Balance);

        await bot.HandleAsync(Tick(4, 0.00011m));
        Assert.Equal(0.00011m, Assert.Single(bot.OpenPositions.Values).Entry.EntryPrice);
        Assert.Equal(8m, bot.Balance);
    }

    [Fact]
    public async Task With_realistic_fills_a_delayed_entry_trades_against_the_execution_pool()
    {
        BotOptions options = Opts(o => o.EntryLatencyMs = 3_000, realistic: true);
        var bot = new PositionManager(options, _store);
        await Run(bot, PoolTick(0, Entry, "pumpswap", 100m), Predict(1), PoolTick(4, Entry, "pumpswap", 50m));

        OpenPosition p = Assert.Single(bot.OpenPositions.Values);
        Assert.Equal(2m, p.Entry.Size); // capped by the 100 SOL pool it was decided on
        Assert.Equal(Fills.BuyPrice(2m, Entry, new Pool("pumpswap", 50m), options), p.Entry.EntryPrice);
    }

    // --- Exit latency ---

    [Theory]
    [InlineData(0.00016, 0.00014, ExitReason.TakeProfit, 0.8)]
    [InlineData(0.00006, 0.00005, ExitReason.StopLoss, -1.0)]
    public async Task An_exit_fills_at_the_first_mark_after_its_latency_and_ignores_triggers_meanwhile(
        double triggerMark, double fillMark, ExitReason reason, double pnl)
    {
        PositionManager bot = Bot(o => o.ExitLatencyMs = 2_000);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(10, (decimal)triggerMark));

        Assert.True(bot.HasPendingExit(Mint));
        Assert.Single(bot.OpenPositions);

        // Before the 12 s due time, a tick through the opposite level queues nothing.
        await Run(bot, Tick(11, reason == ExitReason.TakeProfit ? 0.00001m : 0.0002m), Predict(11.5, SellSignal()));
        Assert.Empty(_store.Closed);

        await Run(bot, Tick(12, (decimal)fillMark));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(reason, t.Reason);
        Assert.Equal((decimal)fillMark, t.ExitPrice);
        Assert.Equal(T0.AddSeconds(12), t.ClosedAt);
        Assert.Equal((decimal)pnl, t.Pnl);
        Assert.False(bot.HasPendingExit(Mint));
        Assert.Empty(bot.OpenPositions);
    }

    [Theory]
    [InlineData(null, 0.00007)] // the feed went silent too: the unobserved market is assumed at the stop
    [InlineData(60.0, 0.00011)] // the feed is live, only the mint is quiet: its last mark
    public async Task A_pending_exit_with_no_mark_falls_back_to_the_stale_mark_rule(double? otherTickAt, double exit)
    {
        PositionManager bot = Bot(o => o.ExitLatencyMs = 2_000);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(5, 0.00011m), Predict(6, SellSignal())); // due at 8 s
        if (otherTickAt is double at)
        {
            await Run(bot, Tick(at, 0.5m, mint: "other"));
        }

        await bot.AdvanceToAsync(T0.AddSeconds(68)); // 8 s + 60 s: still within the window
        Assert.Empty(_store.Closed);

        await bot.AdvanceToAsync(T0.AddSeconds(69));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Signal, t.Reason);
        Assert.Equal((decimal)exit, t.ExitPrice);
        Assert.Equal(T0.AddSeconds(68), t.ClosedAt);
    }

    [Fact]
    public async Task A_timeout_while_an_exit_is_pending_queues_no_second_exit()
    {
        PositionManager bot = Bot(o => o.ExitLatencyMs = 2_000);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(900, 0.00016m)); // TP sent, due 902; deadline 901
        await bot.AdvanceToAsync(T0.AddSeconds(901.5));

        Assert.Empty(_store.Closed);

        await Run(bot, Tick(902, 0.00015m));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.TakeProfit, t.Reason);
        Assert.Equal(0.00015m, t.ExitPrice);
    }

    [Fact]
    public async Task A_timeout_sends_its_exit_at_the_deadline_and_fills_after_the_latency()
    {
        PositionManager bot = Bot(o => o.ExitLatencyMs = 2_000);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(890, Entry), Tick(903, 0.00012m)); // deadline 901, due 903

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Timeout, t.Reason);
        Assert.Equal(0.00012m, t.ExitPrice);
        Assert.Equal(T0.AddSeconds(903), t.ClosedAt);
    }

    // --- Network costs ---

    [Fact]
    public async Task Network_fees_and_rent_are_paid_from_the_balance_and_show_in_pnl()
    {
        PositionManager bot = Bot(o =>
        {
            o.NetworkFeeSol = 0.001m;
            o.AccountRentSol = 0.002m;
        });
        await Run(bot, Tick(0, Entry), Predict(1));

        OpenPosition p = Assert.Single(bot.OpenPositions.Values);
        Assert.Equal(1.9992m, p.Entry.Size); // 20 % of the 9.996 left once a round trip's 0.004 is set aside
        Assert.Equal(0.003m, p.Entry.EntryCost);
        Assert.Equal(10m - 1.9992m - 0.003m, bot.Balance);

        await Run(bot, Tick(10, 0.00015m));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(0.001m, t.ExitCost);
        Assert.Equal(1.9992m * 0.5m - 0.004m, t.Pnl); // +50 % less the two fees and the rent
        Assert.Equal((double)(t.Pnl / 1.9992m), t.PnlPct, 12);
        Assert.Equal(10m + t.Pnl, bot.Balance);
        Assert.Equal(bot.Balance, bot.Equity);
        Assert.Equal(t.Pnl, bot.RealizedPnl);
    }

    [Fact]
    public async Task A_round_trip_at_an_unchanged_price_loses_exactly_the_fixed_costs()
    {
        PositionManager bot = Bot(o =>
        {
            o.NetworkFeeSol = 0.0002m;
            o.AccountRentSol = 0.00203928m;
        });
        await Run(bot, Tick(0, Entry), Predict(1), Tick(5, Entry), Predict(6, SellSignal()));

        Assert.Equal(-(2m * 0.0002m + 0.00203928m), Assert.Single(_store.Closed).Pnl);
    }

    [Fact]
    public async Task An_exit_into_a_drained_pool_still_pays_its_network_fee()
    {
        var bot = new PositionManager(Opts(o => o.NetworkFeeSol = 0.001m, realistic: true), _store);
        await Run(bot, PoolTick(0, Entry, "raydium", 100m), Predict(1), PoolTick(10, Entry * 2m, "raydium", 0m));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(0m, t.ExitPrice);
        Assert.Equal(-t.Entry.Size - 0.002m, t.Pnl);
        Assert.Equal(10m + t.Pnl, bot.Balance);
    }

    // --- Slippage tolerance / failed fills ---

    [Fact]
    public async Task An_entry_past_the_slippage_tolerance_fails_and_still_pays_the_network_fee()
    {
        var sink = new MessageSink();
        PositionManager bot = Bot(o =>
        {
            o.EntryLatencyMs = 3_000;
            o.MaxSlippagePct = 0.1;
            o.NetworkFeeSol = 0.001m;
        }, sink: sink);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(4, 0.000111m)); // +11 % > 10 %

        Assert.Empty(bot.OpenPositions);
        Assert.Empty(_store.Open);
        Assert.Equal(0, bot.PendingEntryCount);
        Assert.Equal(1, bot.FailedEntryCount);
        Assert.Equal(10m - 0.001m, bot.Balance);
        Assert.Equal(-0.001m, bot.RealizedPnl);
        Assert.Equal(bot.Balance, bot.Equity);
        Assert.Contains("entry failed", sink.Messages);
    }

    [Theory]
    [InlineData(0.00011)] // exactly at the 10 % tolerance
    [InlineData(0.00005)] // a cheaper fill never breaches it
    public async Task An_entry_within_the_slippage_tolerance_fills(double fillMark)
    {
        PositionManager bot = Bot(o =>
        {
            o.EntryLatencyMs = 3_000;
            o.MaxSlippagePct = 0.1;
        });
        await Run(bot, Tick(0, Entry), Predict(1), Tick(4, (decimal)fillMark));

        Assert.Equal((decimal)fillMark, Assert.Single(bot.OpenPositions.Values).Entry.EntryPrice);
        Assert.Equal(0, bot.FailedEntryCount);
    }

    [Fact]
    public async Task A_delayed_entry_into_a_pool_drained_meanwhile_fails()
    {
        var bot = new PositionManager(Opts(o =>
        {
            o.EntryLatencyMs = 3_000;
            o.NetworkFeeSol = 0.001m;
        }, realistic: true), _store);
        await Run(bot, PoolTick(0, Entry, "raydium", 100m), Predict(1), PoolTick(4, Entry, "raydium", 0m));

        Assert.Empty(bot.OpenPositions);
        Assert.Equal(1, bot.FailedEntryCount);
        Assert.Equal(10m - 0.001m, bot.Balance);
    }

    [Fact]
    public async Task An_exit_past_the_slippage_tolerance_is_retried_at_the_next_mark_and_pays_each_attempt()
    {
        var sink = new MessageSink();
        PositionManager bot = Bot(o =>
        {
            o.ExitLatencyMs = 2_000;
            o.MaxSlippagePct = 0.1;
            o.MaxExitRetries = 2;
            o.NetworkFeeSol = 0.001m;
        }, sink: sink);
        await Run(bot, Tick(0, Entry), Predict(1), Tick(10, 0.00007m)); // SL sent, quoted 0.00007, due 12

        await Run(bot, Tick(12, 0.00006m)); // −14 %: fails, re-sent quoted 0.00006, due 14
        Assert.Empty(_store.Closed);
        Assert.Equal(1, bot.FailedExitCount);
        Assert.True(bot.HasPendingExit(Mint));

        await Run(bot, Tick(13, 0.00001m), Tick(14, 0.000055m)); // too early, then −8 %: fills

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.StopLoss, t.Reason);
        Assert.Equal(0.000055m, t.ExitPrice);
        Assert.Equal(T0.AddSeconds(14), t.ClosedAt);
        Assert.Equal(0.002m, t.ExitCost); // two attempts
        Assert.Equal(1, bot.FailedExitCount);
        Assert.Contains("exit failed", sink.Messages);
    }

    [Fact]
    public async Task After_the_last_retry_an_exit_fills_at_any_price()
    {
        PositionManager bot = Bot(o =>
        {
            o.ExitLatencyMs = 2_000;
            o.MaxSlippagePct = 0.1;
            o.MaxExitRetries = 1;
            o.NetworkFeeSol = 0.001m;
        });
        await Run(bot, Tick(0, Entry), Predict(1), Tick(10, 0.00007m), Tick(12, 0.00006m), Tick(14, 0.00003m));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(0.00003m, t.ExitPrice); // −50 %, but the retries are spent: never dropped
        Assert.Equal(0.002m, t.ExitCost);
        Assert.Equal(1, bot.FailedExitCount);
    }

    // --- Replay ---

    [Fact]
    public async Task A_replay_with_every_production_knob_is_deterministic_and_counts_the_failures()
    {
        BotEvent[] events =
        [
            PoolTick(0, Entry, "pumpswap", 100m),
            PoolTick(0, Entry, "pumpswap", 100m, mint: "b"),
            Predict(1),
            Predict(1, mint: "b"),
            PoolTick(4, 0.00013m, "pumpswap", 100m), // +30 % > the 15 % default: fails
            PoolTick(4, 0.000105m, "pumpswap", 100m, mint: "b"), // fills
            PoolTick(20, 0.00016m, "pumpswap", 100m, mint: "b"), // TP sent, due 22
            PoolTick(22, 0.00016m, "pumpswap", 100m, mint: "b"), // fills
        ];

        ReplayResult first = await ReplayHarness.RunAsync(PolicyFixtures.ProductionOptions(), events);
        ReplayResult second = await ReplayHarness.RunAsync(PolicyFixtures.ProductionOptions(), events);

        Assert.Equal(first, second);
        Assert.Equal(1, first.FailedEntryCount);
        ClosedTrade t = Assert.Single(first.Trades);
        Assert.Equal("b", t.Entry.Mint);
        Assert.Equal(T0.AddSeconds(4), t.Entry.OpenedAt);
        Assert.Equal(T0.AddSeconds(22), t.ClosedAt);
        Assert.Equal(new BotOptions().NetworkFeeSol + new BotOptions().AccountRentSol, t.Entry.EntryCost);
        Assert.Equal(t.Pnl - new BotOptions().NetworkFeeSol, first.RealizedPnl); // the failed entry's fee
        Assert.Equal(first.StartingBalance + first.RealizedPnl, first.FinalBalance);
    }

    private sealed class MessageSink : IBotLogSink
    {
        public List<string> Messages { get; } = [];

        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
        {
            Messages.Add(message);
            return Task.CompletedTask;
        }
    }

    private sealed class FailOnceOpenStore : ITradeStore
    {
        private readonly InMemoryTradeStore _inner = new();
        private bool _failed;

        public Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default)
        {
            if (!_failed)
            {
                _failed = true;
                throw new InvalidOperationException("db blip");
            }

            return _inner.OpenAsync(opening, cancellationToken);
        }

        public Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default) =>
            _inner.CloseAsync(trade, cancellationToken);
    }
}
