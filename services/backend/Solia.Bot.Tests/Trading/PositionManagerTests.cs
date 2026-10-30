using System.Text.Json;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Logging;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;
using Solia.Shared.Logging;

namespace Solia.Bot.Tests.Trading;

/// <summary>
/// Engine scenarios on known numbers. With the default options a <see cref="PolicyFixtures.Buyable"/>
/// prediction sizes to the 20% cap of the 10 starting balance = 2, filled at 0.0001: TP 0.00015 (+50%),
/// SL 0.00007 (−30%, no volatility), timeout 900 s.
/// </summary>
public class PositionManagerTests
{
    private const string Mint = "mint1";
    private const decimal Entry = 0.0001m;
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private readonly InMemoryTradeStore _store = new();
    private readonly PositionManager _bot;

    public PositionManagerTests() => _bot = new PositionManager(PolicyFixtures.Options(), _store);

    private static BotEvent Tick(int sec, decimal price, string mint = Mint) =>
        new BotEvent.Price(new PriceTick { Mint = mint, Ts = T0.AddSeconds(sec), Price = price });

    private static BotEvent Predict(int sec, ModelPrediction? p = null, string mint = Mint) =>
        new BotEvent.Prediction((p ?? PolicyFixtures.Buyable()) with { Mint = mint, Ts = T0.AddSeconds(sec) });

    private static ModelPrediction SellSignal() => PolicyFixtures.Buyable() with { PumpProb = 0.1, RugProb = 0.6 };

    private async Task Run(params BotEvent[] events)
    {
        foreach (BotEvent e in events)
        {
            await _bot.HandleAsync(e);
        }
    }

    /// <summary>Marks the mint at the entry price, then opens a position on it at t = 1.</summary>
    private Task Opened() => Run(Tick(0, Entry), Predict(1));

    [Fact]
    public async Task A_buy_opens_at_the_market_price_and_locks_the_size()
    {
        await Opened();

        OpenPosition p = Assert.Single(_bot.OpenPositions.Values);
        Assert.Equal(Entry, p.Entry.EntryPrice);
        Assert.Equal(2m, p.Entry.Size);
        Assert.Equal(T0.AddSeconds(1), p.Entry.OpenedAt);
        Assert.Equal(T0.AddSeconds(901), p.Deadline);
        Assert.Equal(0.00015m, p.Entry.Levels.TakeProfitPrice);
        Assert.Equal(0.00007m, p.Entry.Levels.StopLossPrice);
        Assert.Equal(8m, _bot.Balance);
        Assert.Equal(10m, _bot.Equity);
        Assert.Same(p.Entry, _store.Open[p.TradeId]);
    }

    [Fact]
    public async Task Volatility_on_the_entry_prediction_sets_a_dynamic_stop()
    {
        await Run(Tick(0, Entry), Predict(1, PolicyFixtures.Buyable() with { Volatility = 0.04 }));
        Assert.Equal(0.000088m, _bot.OpenPositions[Mint].Entry.Levels.StopLossPrice);
    }

    [Fact]
    public async Task No_market_price_means_no_position()
    {
        await Run(Predict(1));

        Assert.Empty(_bot.OpenPositions);
        Assert.Empty(_store.Open);
        Assert.Equal(10m, _bot.Balance);
    }

    [Fact]
    public async Task A_rejected_prediction_opens_nothing()
    {
        await Run(Tick(0, Entry), Predict(1, PolicyFixtures.Buyable() with { Confidence = 10 }));
        Assert.Empty(_bot.OpenPositions);
    }

    [Fact]
    public async Task Marks_between_the_levels_keep_the_position_open_and_move_equity()
    {
        await Opened();
        await Run(Tick(2, 0.00012m));

        Assert.Single(_bot.OpenPositions);
        Assert.Equal(8m, _bot.Balance);
        Assert.Equal(10.4m, _bot.Equity); // 8 + 2 · 1.2
        Assert.Equal(0, _bot.ClosedTradeCount);
    }

    [Fact]
    public async Task Take_profit_closes_at_the_tick_price()
    {
        await Opened();
        await Run(Tick(31, 0.00016m)); // gaps past TP 0.00015

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.TakeProfit, t.Reason);
        Assert.Equal(0.00016m, t.ExitPrice);
        Assert.Equal(1.2m, t.Pnl);
        Assert.Equal(0.6, t.PnlPct, 12);
        Assert.Equal(30, t.DurationSec);
        Assert.Equal(11.2m, _bot.Balance);
        Assert.Equal(11.2m, _bot.Equity);
        Assert.Equal(1.2m, _bot.RealizedPnl);
        Assert.Equal(1, _bot.ClosedTradeCount);
        Assert.Empty(_bot.OpenPositions);
        Assert.Empty(_store.Open);
    }

    [Fact]
    public async Task Take_profit_triggers_exactly_at_the_level()
    {
        await Opened();
        await Run(Tick(2, 0.00015m));
        Assert.Equal(ExitReason.TakeProfit, Assert.Single(_store.Closed).Reason);
    }

    [Fact]
    public async Task Stop_loss_closes_at_the_tick_price()
    {
        await Opened();
        await Run(Tick(11, 0.00006m)); // gaps through SL 0.00007

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.StopLoss, t.Reason);
        Assert.Equal(-0.8m, t.Pnl);
        Assert.Equal(-0.4, t.PnlPct, 12);
        Assert.Equal(9.2m, _bot.Balance);
        Assert.Equal(-0.8m, _bot.RealizedPnl);
    }

    [Fact]
    public async Task Stop_loss_triggers_exactly_at_the_level()
    {
        await Opened();
        await Run(Tick(2, 0.00007m));
        Assert.Equal(ExitReason.StopLoss, Assert.Single(_store.Closed).Reason);
    }

    [Fact]
    public async Task A_fresh_sell_signal_closes_at_the_last_mark()
    {
        await Opened();
        await Run(Tick(5, 0.00011m), Predict(6, SellSignal()));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Signal, t.Reason);
        Assert.Equal(0.00011m, t.ExitPrice);
        Assert.Equal(0.2m, t.Pnl);
        Assert.Equal(T0.AddSeconds(6), t.ClosedAt);
    }

    [Theory]
    [MemberData(nameof(NonSellPredictions))]
    public async Task A_non_sell_prediction_keeps_the_position(ModelPrediction prediction)
    {
        await Opened();
        await Run(Predict(6, prediction));

        Assert.Single(_bot.OpenPositions);
        Assert.Empty(_store.Closed);
        Assert.Single(_store.Open); // a repeat buy never opens a second position on the mint
    }

    public static TheoryData<ModelPrediction> NonSellPredictions() =>
    [
        PolicyFixtures.Buyable(),                                                // buy again
        PolicyFixtures.Buyable() with { PumpProb = 0.3 },                         // hold
        PolicyFixtures.Buyable() with { PumpProb = 0.1, RugProb = 0.6, Confidence = 10 }, // sell, but not confident
        PolicyFixtures.Buyable() with { RugProb = double.NaN },                  // malformed
    ];

    [Fact]
    public async Task A_sell_signal_for_another_mint_does_not_touch_the_position()
    {
        await Opened();
        await Run(Predict(6, SellSignal(), mint: "other"));
        Assert.Single(_bot.OpenPositions);
    }

    [Fact]
    public async Task Timeout_at_entry_price_is_break_even_and_closes_at_the_deadline()
    {
        await Opened();
        await Run(Tick(890, Entry), Tick(1000, 0.5m, mint: "other")); // fresh mark, then any later event sweeps

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Timeout, t.Reason);
        Assert.Equal(T0.AddSeconds(901), t.ClosedAt);
        Assert.Equal(Entry, t.ExitPrice);
        Assert.Equal(0m, t.Pnl);
        Assert.Equal(900, t.DurationSec);
        Assert.Equal(10m, _bot.Balance);
        Assert.Equal(1, _bot.ClosedTradeCount);
    }

    [Fact]
    public async Task Timeout_closes_at_the_last_mark_before_the_deadline()
    {
        await Opened();
        await Run(Tick(880, 0.00012m), Tick(2000, 0.00013m, mint: "other"));

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Timeout, t.Reason);
        Assert.Equal(0.00012m, t.ExitPrice);
        Assert.Equal(0.4m, t.Pnl);
    }

    [Fact]
    public async Task Timeout_takes_precedence_over_a_take_profit_tick_at_the_deadline()
    {
        await Opened();
        await Run(Tick(880, Entry), Tick(901, 0.0002m)); // deadline reached: the position expired before this tick

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Timeout, t.Reason);
        Assert.Equal(Entry, t.ExitPrice);
    }

    [Fact]
    public async Task Timeout_takes_precedence_over_a_sell_signal_at_the_deadline()
    {
        await Opened();
        await Run(Predict(901, SellSignal()));
        Assert.Equal(ExitReason.Timeout, Assert.Single(_store.Closed).Reason);
    }

    [Fact]
    public async Task A_take_profit_one_second_before_the_deadline_wins()
    {
        await Opened();
        await Run(Tick(900, 0.0002m));
        Assert.Equal(ExitReason.TakeProfit, Assert.Single(_store.Closed).Reason);
    }

    [Fact]
    public async Task Advancing_the_clock_sweeps_timeouts_without_an_event()
    {
        await Opened();

        await _bot.AdvanceToAsync(T0.AddSeconds(900));
        Assert.Single(_bot.OpenPositions);

        await _bot.AdvanceToAsync(T0.AddSeconds(901));
        Assert.Empty(_bot.OpenPositions);
        Assert.Equal(T0.AddSeconds(901), _bot.Clock);
    }

    [Fact]
    public async Task Expired_positions_close_earliest_deadline_first()
    {
        await Run(Tick(0, Entry), Tick(0, Entry, mint: "b"), Predict(1), Predict(5, mint: "b"));
        await _bot.AdvanceToAsync(T0.AddDays(1));

        Assert.Equal([Mint, "b"], _store.Closed.Select(t => t.Entry.Mint));
        Assert.Equal(2, _bot.ClosedTradeCount);
    }

    [Fact]
    public async Task A_closed_mint_can_be_entered_again()
    {
        await Opened();
        await Run(Tick(2, 0.00015m), Predict(3));

        Assert.Single(_bot.OpenPositions);
        Assert.Equal(0.00015m, _bot.OpenPositions[Mint].Entry.EntryPrice);
    }

    [Fact]
    public async Task Capacity_is_capped_at_max_concurrent_positions()
    {
        var bot = new PositionManager(new BotOptions { MaxConcurrentPositions = 1, RealisticFills = false }, _store);
        foreach (BotEvent e in new[] { Tick(0, Entry), Tick(0, Entry, mint: "b"), Predict(1), Predict(2, mint: "b") })
        {
            await bot.HandleAsync(e);
        }

        Assert.Equal([Mint], bot.OpenPositions.Keys);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public async Task A_non_positive_tick_is_ignored(int price)
    {
        await Opened();
        await Run(Tick(2, price));

        Assert.Single(_bot.OpenPositions);
        Assert.Equal(Entry, _bot.MarkOf(Mint));
    }

    [Fact]
    public async Task A_late_prediction_within_the_lag_opens_at_the_clock()
    {
        await Run(Tick(0, Entry), Tick(3, 0.5m, mint: "other"), Predict(1)); // 2 s late, allowed lag 5 s

        OpenPosition p = Assert.Single(_bot.OpenPositions.Values);
        Assert.Equal(T0.AddSeconds(3), p.Entry.OpenedAt);
        Assert.Equal(T0.AddSeconds(903), p.Deadline);
        Assert.Equal(T0.AddSeconds(3), _bot.Clock);
        Assert.Equal(0, _bot.DroppedEventCount);
    }

    [Fact]
    public async Task An_event_later_than_the_allowed_lag_is_dropped()
    {
        await Opened();
        await Run(Tick(10, 0.00011m), Tick(4, 0.00001m)); // 6 s late

        Assert.Equal(1, _bot.DroppedEventCount);
        Assert.Single(_bot.OpenPositions);
        Assert.Equal(0.00011m, _bot.MarkOf(Mint));
        Assert.Equal(T0.AddSeconds(10), _bot.Clock);
    }

    [Fact]
    public async Task A_late_tick_older_than_the_mark_never_overwrites_it()
    {
        await Opened();
        await Run(Tick(5, 0.00012m), Tick(3, 0.00001m)); // within the lag, but older than the mint's mark

        Assert.Equal(0.00012m, _bot.MarkOf(Mint));
        Assert.Single(_bot.OpenPositions);
        Assert.Equal(0, _bot.DroppedEventCount);
    }

    [Fact]
    public async Task A_late_tick_from_before_the_fill_marks_but_does_not_exit()
    {
        await Run(Tick(0, Entry), Tick(3, 0.5m, mint: "other"), Predict(3)); // opened at t = 3
        await Run(Tick(2, 0.00001m)); // newer than the mint's mark (t = 0), older than the fill

        Assert.Equal(0.00001m, _bot.MarkOf(Mint));
        Assert.Single(_bot.OpenPositions);
    }

    [Fact]
    public async Task Advancing_the_clock_backwards_is_a_no_op()
    {
        await Opened();
        await _bot.AdvanceToAsync(T0);
        Assert.Equal(T0.AddSeconds(1), _bot.Clock);
    }

    [Theory]
    [InlineData(60, true)]  // exactly MaxMarkAgeSeconds old: still the market
    [InlineData(61, false)] // older: no market, no entry
    public async Task Entry_needs_a_fresh_mark(int predictAt, bool opens)
    {
        await Run(Tick(0, Entry), Predict(predictAt));
        Assert.Equal(opens, _bot.OpenPositions.Count == 1);
    }

    [Fact]
    public async Task A_signal_exit_on_a_stale_mark_fills_at_the_stop()
    {
        await Opened();
        await Run(Predict(100, SellSignal())); // last mark is 100 s old

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Signal, t.Reason);
        Assert.Equal(0.00007m, t.ExitPrice);
        Assert.Equal(-0.6m, t.Pnl);
    }

    [Fact]
    public async Task A_signal_exit_on_a_fresh_mark_fills_at_the_mark()
    {
        await Opened();
        await Run(Tick(40, 0.00009m), Predict(100, SellSignal())); // mark 60 s old: still fresh

        Assert.Equal(0.00009m, Assert.Single(_store.Closed).ExitPrice);
    }

    [Fact]
    public async Task A_timeout_on_a_stale_mark_fills_at_the_stop()
    {
        await Opened();
        await _bot.AdvanceToAsync(T0.AddSeconds(901)); // no tick since entry: the feed went silent

        ClosedTrade t = Assert.Single(_store.Closed);
        Assert.Equal(ExitReason.Timeout, t.Reason);
        Assert.Equal(0.00007m, t.ExitPrice);
        Assert.Equal(-0.6m, t.Pnl);
    }

    [Fact]
    public async Task A_position_closed_early_is_not_swept_again()
    {
        await Opened();
        await Run(Tick(2, 0.00015m), Predict(3)); // TP, then re-entry on the same mint (deadline 903)

        await _bot.AdvanceToAsync(T0.AddSeconds(902)); // past the first trade's deadline only
        Assert.Single(_store.Closed);
        Assert.Single(_bot.OpenPositions);

        await _bot.AdvanceToAsync(T0.AddSeconds(903));
        Assert.Equal(2, _store.Closed.Count);
        Assert.Empty(_bot.OpenPositions);
    }

    [Fact]
    public async Task A_store_failure_mid_sweep_is_resumed_by_redelivering_the_event()
    {
        var store = new FlakyCloseStore(failingCall: 2);
        var bot = new PositionManager(PolicyFixtures.Options(), store);
        foreach (BotEvent e in new[] { Tick(0, Entry), Tick(0, Entry, mint: "b"), Predict(1), Predict(5, mint: "b") })
        {
            await bot.HandleAsync(e);
        }

        BotEvent trigger = Tick(1000, Entry, mint: "c");
        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.HandleAsync(trigger));
        Assert.Equal([Mint], store.Closed.Select(t => t.Entry.Mint)); // the first close stays applied
        Assert.Equal(T0.AddSeconds(901), bot.Clock);                  // and the clock stands at it
        Assert.Null(bot.MarkOf("c"));                                // the event itself is not applied yet

        await bot.HandleAsync(trigger);
        Assert.Equal([Mint, "b"], store.Closed.Select(t => t.Entry.Mint)); // nothing closed twice
        Assert.Equal(Entry, bot.MarkOf("c"));
        Assert.Equal(T0.AddSeconds(1000), bot.Clock);
        Assert.Equal(2, bot.ClosedTradeCount);
    }

    [Fact]
    public async Task Events_at_the_same_timestamp_are_accepted()
    {
        await Run(Tick(1, Entry), Predict(1));
        Assert.Single(_bot.OpenPositions);
    }

    [Fact]
    public async Task A_store_failure_on_open_leaves_the_book_untouched()
    {
        var bot = new PositionManager(PolicyFixtures.Options(), new FailingStore(failOpen: true));
        await bot.HandleAsync(Tick(0, Entry));

        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.HandleAsync(Predict(1)));
        Assert.Empty(bot.OpenPositions);
        Assert.Equal(10m, bot.Balance);
    }

    [Fact]
    public async Task A_store_failure_on_close_keeps_the_position_open()
    {
        var bot = new PositionManager(PolicyFixtures.Options(), new FailingStore(failOpen: false));
        await bot.HandleAsync(Tick(0, Entry));
        await bot.HandleAsync(Predict(1));

        await Assert.ThrowsAsync<InvalidOperationException>(() => bot.HandleAsync(Tick(2, 0.0002m)));
        Assert.Single(bot.OpenPositions);
        Assert.Equal(8m, bot.Balance);
        Assert.Equal(0, bot.ClosedTradeCount);
    }

    [Fact]
    public async Task Replaying_the_same_events_gives_the_same_trades()
    {
        BotEvent[] events =
        [
            Tick(0, Entry), Tick(0, 0.002m, mint: "b"), Predict(1), Predict(2, mint: "b"),
            Tick(10, 0.00016m), Tick(20, 0.0019m, mint: "b"), Predict(30, SellSignal(), mint: "b"),
        ];

        async Task<IReadOnlyList<ClosedTrade>> Replay()
        {
            var store = new InMemoryTradeStore();
            var bot = new PositionManager(PolicyFixtures.Options(), store);
            foreach (BotEvent e in events)
            {
                await bot.HandleAsync(e);
            }

            return store.Closed;
        }

        IReadOnlyList<ClosedTrade> first = await Replay();
        Assert.Equal(2, first.Count);
        Assert.Equal(first, await Replay());
    }

    [Fact]
    public void Constructor_rejects_null_dependencies()
    {
        Assert.Throws<ArgumentNullException>(() => new PositionManager(null!, _store));
        Assert.Throws<ArgumentNullException>(() => new PositionManager(PolicyFixtures.Options(), null!));
    }

    [Fact]
    public async Task Open_and_close_log_rows_share_camel_case_keys_so_they_correlate_by_trade_id()
    {
        var sink = new ContextSink();
        var bot = new PositionManager(PolicyFixtures.Options(), _store, sink);

        foreach (BotEvent e in new[] { Tick(0, Entry), Predict(1), Tick(2, 0.00015m) })
        {
            await bot.HandleAsync(e);
        }

        Assert.Equal(["position opened", "position closed"], sink.Rows.Select(r => r.Message));
        using JsonDocument opened = JsonDocument.Parse(sink.Rows[0].Json!);
        using JsonDocument closed = JsonDocument.Parse(sink.Rows[1].Json!);
        Assert.Equal(opened.RootElement.GetProperty("tradeId").GetInt64(), closed.RootElement.GetProperty("tradeId").GetInt64());
        Assert.Equal(Mint, opened.RootElement.GetProperty("mint").GetString());
        Assert.Equal(Mint, closed.RootElement.GetProperty("mint").GetString());
        Assert.Equal("tp", closed.RootElement.GetProperty("reason").GetString());
    }

    /// <summary>Captures each entry's context exactly as the Postgres sink would store it.</summary>
    private sealed class ContextSink : IBotLogSink
    {
        public List<(string Message, string? Json)> Rows { get; } = [];

        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
        {
            Rows.Add((message, PostgresLogSink.SerializeContext(context)));
            return Task.CompletedTask;
        }
    }

    /// <summary>Fails the <paramref name="failingCall"/>-th close once, then behaves.</summary>
    private sealed class FlakyCloseStore(int failingCall) : ITradeStore
    {
        private readonly InMemoryTradeStore _inner = new();
        private int _closeCalls;

        public IReadOnlyList<ClosedTrade> Closed => _inner.Closed;

        public Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default) =>
            _inner.OpenAsync(opening, cancellationToken);

        public Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default) =>
            ++_closeCalls == failingCall
                ? throw new InvalidOperationException("db blip")
                : _inner.CloseAsync(trade, cancellationToken);
    }

    // --- Realistic fills (#165): the production default ---

    private static BotEvent PoolTick(int sec, decimal price, string? venue, decimal? liquidity, string mint = Mint) =>
        new BotEvent.Price(new PriceTick { Mint = mint, Ts = T0.AddSeconds(sec), Price = price, Venue = venue, Liquidity = liquidity });

    private static (PositionManager Bot, InMemoryTradeStore Store) Realistic()
    {
        var store = new InMemoryTradeStore();
        return (new PositionManager(PolicyFixtures.RealisticOptions(), store), store);
    }

    private static async Task Run(PositionManager bot, params BotEvent[] events)
    {
        foreach (BotEvent e in events)
        {
            await bot.HandleAsync(e);
        }
    }

    [Fact]
    public async Task Realistic_entry_fills_above_the_mark_with_levels_from_the_mark()
    {
        (PositionManager bot, InMemoryTradeStore store) = Realistic();
        await Run(bot, PoolTick(0, Entry, "pumpswap", 100m), Predict(1));

        OpenPosition p = Assert.Single(bot.OpenPositions.Values);
        Assert.Equal(2m, p.Entry.Size); // 2 % of the 100 SOL pool = the 0.2 · 10 balance cap
        // n = 2 · 0.997 swapped into R = 100: P · (R + n) / (R · 0.997).
        Assert.Equal(Entry * (100m + 1.994m) / (100m * 0.997m), p.Entry.EntryPrice);
        Assert.Equal(0.00015m, p.Entry.Levels.TakeProfitPrice);
        Assert.Equal(0.00007m, p.Entry.Levels.StopLossPrice);
        Assert.Equal(8m, bot.Balance);
        Assert.True(bot.Equity < 10m); // the entry's fee and impact are already lost at the mark
        Assert.Same(p.Entry, store.Open[p.TradeId]);
    }

    [Theory]
    [InlineData("raydium", 0.03)]
    [InlineData("pumpswap", null)]
    public async Task Realistic_fills_never_enter_a_dust_or_unknown_pool(string venue, double? liquidity)
    {
        (PositionManager bot, InMemoryTradeStore store) = Realistic();
        await Run(bot, PoolTick(0, Entry, venue, (decimal?)liquidity), Predict(1));

        Assert.Empty(bot.OpenPositions);
        Assert.Empty(store.Open);
        Assert.Equal(10m, bot.Balance);
    }

    [Fact]
    public async Task A_curve_token_with_almost_no_real_sol_enters_with_a_pool_sized_position()
    {
        (PositionManager bot, _) = Realistic();
        await Run(bot, PoolTick(0, Entry, "pumpfun", 0.05m), Predict(1));

        Assert.Equal(0.601m, Assert.Single(bot.OpenPositions.Values).Entry.Size); // 2 % of 30.05 SOL
        Assert.Equal(10m - 0.601m, bot.Balance);
    }

    [Fact]
    public async Task A_realistic_take_profit_pays_impact_and_fees_on_both_sides()
    {
        (PositionManager bot, InMemoryTradeStore store) = Realistic();
        await Run(bot, PoolTick(0, Entry, "pumpswap", 100m), Predict(1), PoolTick(10, 0.00015m, "pumpswap", 100m));

        ClosedTrade t = Assert.Single(store.Closed);
        Assert.Equal(ExitReason.TakeProfit, t.Reason);
        TradeOpening entry = t.Entry;
        decimal value = entry.ValueAt(0.00015m);
        Assert.Equal(0.00015m * 100m * 0.997m / (100m + value), t.ExitPrice);
        Assert.Equal(Money.Round(entry.ValueAt(t.ExitPrice) - 2m), t.Pnl);
        Assert.InRange(t.Pnl, 0.80m, 0.90m); // the ideal fill would book +1.00
        Assert.Equal(10m + t.Pnl, bot.Balance);
        Assert.Equal(t.Pnl, bot.RealizedPnl);
    }

    [Fact]
    public async Task A_round_trip_at_an_unchanged_price_loses_the_fees_and_impact()
    {
        (PositionManager bot, InMemoryTradeStore store) = Realistic();
        await Run(bot, PoolTick(0, Entry, "pumpswap", 1_000_000m), Predict(1), PoolTick(890, Entry, "pumpswap", 1_000_000m)); // fresh at the deadline
        await bot.AdvanceToAsync(T0.AddSeconds(901));

        ClosedTrade t = Assert.Single(store.Closed);
        Assert.Equal(ExitReason.Timeout, t.Reason);
        // 0.30 % per side on 2 SOL, plus an impact of ~2e-6 per side (~1e-5 SOL in all).
        decimal feesOnly = -2m * (1m - (0.997m * 0.997m));
        Assert.InRange(t.Pnl, feesOnly - 0.00002m, feesOnly);
    }

    [Fact]
    public async Task An_exit_into_a_drained_pool_realizes_nothing_whatever_the_mark()
    {
        (PositionManager bot, InMemoryTradeStore store) = Realistic();
        // The mark jumps +656 % as the pool drains to 0 SOL (the 2026-10-30 artifact): the take-profit fires, but the
        // pool has nothing to pay.
        await Run(bot, PoolTick(0, Entry, "raydium", 100m), Predict(1), PoolTick(10, Entry * 7.56m, "raydium", 0m));

        ClosedTrade t = Assert.Single(store.Closed);
        Assert.Equal(ExitReason.TakeProfit, t.Reason);
        Assert.Equal(0m, t.ExitPrice);
        Assert.Equal(-t.Entry.Size, t.Pnl);
        Assert.Equal(-1.0, t.PnlPct);
    }

    [Fact]
    public async Task A_stale_timeout_exit_fills_at_the_stop_against_the_last_known_pool()
    {
        (PositionManager bot, InMemoryTradeStore store) = Realistic();
        await Run(bot, PoolTick(0, Entry, "pumpswap", 100m), Predict(1));
        await bot.AdvanceToAsync(T0.AddSeconds(901)); // last mark is 901 s old: stale

        ClosedTrade t = Assert.Single(store.Closed);
        decimal value = t.Entry.ValueAt(0.00007m);
        Assert.Equal(0.00007m * 100m * 0.997m / (100m + value), t.ExitPrice);
    }

    private sealed class FailingStore(bool failOpen) : ITradeStore
    {
        private readonly InMemoryTradeStore _inner = new();

        public Task<long> OpenAsync(TradeOpening opening, CancellationToken cancellationToken = default) =>
            failOpen ? throw new InvalidOperationException("db down") : _inner.OpenAsync(opening, cancellationToken);

        public Task CloseAsync(ClosedTrade trade, CancellationToken cancellationToken = default) =>
            throw new InvalidOperationException("db down");
    }
}
