using System.ComponentModel.DataAnnotations;
using Solia.Bot.Backtest;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Trading;
using static Solia.Bot.Tests.Backtest.ReplayFixtures;

namespace Solia.Bot.Tests.Backtest;

/// <summary>
/// End-to-end replay of the canonical <see cref="ReplayFixtures.Scenario"/> on known numbers, plus the
/// determinism and tie-break guarantees the harness documents. This is the Sprint 3 Definition of Done:
/// the four exit reasons are all exercised and every trade's exit, P&amp;L and the run's balance /
/// win-rate are asserted on fixed values, not recomputed from the engine.
/// </summary>
public class ReplayHarnessTests
{
    private static Task<ReplayResult> Replay(IEnumerable<BotEvent> events, ReplayOrder order = ReplayOrder.Arrival) =>
        ReplayHarness.RunAsync(ReplayFixtures.Options(), events, order);

    [Fact]
    public async Task The_scenario_closes_four_trades_one_per_exit_reason()
    {
        ReplayResult result = await Replay(Scenario());

        Assert.Equal(
            [ExitReason.TakeProfit, ExitReason.StopLoss, ExitReason.Signal, ExitReason.Timeout],
            result.Trades.Select(t => t.Reason));
        Assert.Equal(["tp", "sl", "sig", "to"], result.Trades.Select(t => t.Entry.Mint));
    }

    [Theory]
    // mint, entryPrice, size, exitPrice, pnl, pnlPct, durationSec
    [InlineData("tp", 0.0001, 2.0, 0.00015, 1.0, 0.5, 1)]
    [InlineData("sl", 0.0001, 2.2, 0.00007, -0.66, -0.3, 1)]
    [InlineData("sig", 0.0001, 2.068, 0.00011, 0.2068, 0.1, 2)]
    [InlineData("to", 0.0001, 2.10936, 0.00012, 0.421872, 0.2, 900)]
    public async Task Each_trade_settles_on_its_known_numbers(
        string mint, decimal entry, decimal size, decimal exit, decimal pnl, double pnlPct, int durationSec)
    {
        ReplayResult result = await Replay(Scenario());

        ClosedTrade trade = result.Trades.Single(t => t.Entry.Mint == mint);
        Assert.Equal(entry, trade.Entry.EntryPrice);
        Assert.Equal(size, trade.Entry.Size);
        Assert.Equal(exit, trade.ExitPrice);
        Assert.Equal(pnl, trade.Pnl);
        Assert.Equal(pnlPct, trade.PnlPct, 12);
        Assert.Equal(durationSec, trade.DurationSec);
    }

    [Fact]
    public async Task The_run_reconciles_balance_equity_and_total_pnl()
    {
        ReplayResult result = await Replay(Scenario());

        // 1 − 0.66 + 0.2068 + 0.421872; the final balance adds the starting 10.
        Assert.Equal(0.968672m, result.RealizedPnl);
        Assert.Equal(0.968672m, result.TotalPnl);
        Assert.Equal(10m, result.StartingBalance);
        Assert.Equal(10.968672m, result.FinalBalance);
        Assert.Equal(result.FinalBalance, result.FinalEquity); // nothing left open
        Assert.Equal(result.StartingBalance + result.TotalPnl, result.FinalBalance);
        Assert.Empty(result.OpenPositions);
        Assert.Equal(0, result.DroppedEventCount);
    }

    [Fact]
    public async Task The_run_reports_three_wins_one_loss_and_a_75_percent_win_rate()
    {
        ReplayResult result = await Replay(Scenario());

        Assert.Equal(4, result.ClosedTradeCount);
        Assert.Equal(3, result.Wins);
        Assert.Equal(1, result.Losses);
        Assert.Equal(0, result.BreakEven);
        Assert.Equal(0.75, result.WinRate, 12);
    }

    [Theory]
    [InlineData(ReplayOrder.Arrival)]
    [InlineData(ReplayOrder.Timestamp)]
    public async Task Replaying_the_same_events_twice_gives_an_equal_result(ReplayOrder order)
    {
        ReplayResult first = await Replay(Scenario(), order);
        ReplayResult second = await Replay(Scenario(), order);

        Assert.Equal(first, second);
        Assert.True(first == second);
        Assert.Equal(first.GetHashCode(), second.GetHashCode());
    }

    [Fact]
    public async Task Results_with_different_trades_are_not_equal()
    {
        ReplayResult full = await Replay(Scenario());
        ReplayResult partial = await Replay(Scenario().Take(3));

        Assert.NotEqual(full, partial);
    }

    [Fact]
    public async Task In_timestamp_order_the_input_order_does_not_matter()
    {
        // Every timestamp in the scenario is distinct, so any permutation sorts back to the same stream.
        BotEvent[] shuffled = [.. Scenario().Reverse()];

        Assert.Equal(await Replay(Scenario()), await Replay(shuffled, ReplayOrder.Timestamp));
    }

    [Fact]
    public async Task In_arrival_order_a_slightly_late_prediction_is_applied_at_the_clock_like_live()
    {
        // Live sees the t=10 tick, then a prediction stamped t=8 (2 s late, within the 5 s lag): it is
        // applied at t=10 against the fresh mark and opens the position at t=10.
        ReplayResult result = await Replay([Tick("m", 10, 0.0001m), Predict("m", 8)]);

        OpenPosition open = Assert.Single(result.OpenPositions);
        Assert.Equal(T0.AddSeconds(10), open.Entry.OpenedAt);
        Assert.Equal(0, result.DroppedEventCount);
    }

    [Fact]
    public async Task In_timestamp_order_the_same_late_prediction_runs_first_and_sees_no_market()
    {
        // Sorting moves the prediction ahead of the tick, so at t=8 there is no mark to fill against.
        ReplayResult result = await Replay([Tick("m", 10, 0.0001m), Predict("m", 8)], ReplayOrder.Timestamp);

        Assert.Empty(result.OpenPositions);
    }

    [Fact]
    public async Task In_arrival_order_an_event_later_than_the_allowed_lag_is_dropped_and_counted()
    {
        // The prediction is 90 s behind the clock, past the 5 s lag: dropped, never acted on.
        ReplayResult result = await Replay([Tick("m", 100, 0.0001m), Predict("m", 10)]);

        Assert.Equal(1, result.DroppedEventCount);
        Assert.Empty(result.OpenPositions);
    }

    [Fact]
    public async Task In_arrival_order_a_prediction_before_a_price_at_an_equal_timestamp_sees_no_market()
    {
        ReplayResult result = await Replay([Predict("m", 0), Tick("m", 0, 0.0001m)]);

        Assert.Empty(result.OpenPositions);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task In_timestamp_order_a_price_goes_before_a_prediction_at_an_equal_timestamp(bool priceFirst)
    {
        // The tie-break is by event kind, not by input order: both enumerations open the position.
        BotEvent[] events = priceFirst
            ? [Tick("m", 0, 0.0001m), Predict("m", 0)]
            : [Predict("m", 0), Tick("m", 0, 0.0001m)];

        ReplayResult result = await Replay(events, ReplayOrder.Timestamp);

        Assert.Single(result.OpenPositions);
    }

    [Fact]
    public async Task Open_positions_are_listed_in_opening_order()
    {
        // A and B open, A closes, C opens — C may reuse A's dictionary slot, but the list stays time-ordered.
        ReplayResult result = await Replay(
        [
            Tick("a", 0, 0.0001m), Predict("a", 1),
            Tick("b", 2, 0.0001m), Predict("b", 3),
            Predict("a", 4, SellSignal()),
            Tick("c", 5, 0.0001m), Predict("c", 6),
        ]);

        Assert.Equal(["b", "c"], result.OpenPositions.Select(p => p.Mint));
    }

    [Fact]
    public async Task Total_pnl_includes_the_unrealized_pnl_of_open_positions()
    {
        // Opens 2 at 0.0001, marked +20% and still held: 0.4 unrealized, nothing realized.
        ReplayResult result = await Replay([Tick("m", 0, 0.0001m), Predict("m", 1), Tick("m", 2, 0.00012m)]);

        Assert.Single(result.OpenPositions);
        Assert.Equal(0m, result.RealizedPnl);
        Assert.Equal(10.4m, result.FinalEquity);
        Assert.Equal(0.4m, result.TotalPnl);
    }

    [Theory]
    [InlineData(60, 0.00012)] // stale 900 s mark, live feed: the quiet mint's price is its last mark
    [InlineData(30, 0.00007)] // stale mark and a feed silent for 31 s: the timeout assumes the stop was hit
    public async Task The_timeout_trade_on_a_stale_mark_depends_on_the_feed_being_live_at_the_deadline(
        int maxFeedSilenceSeconds, decimal exit)
    {
        // Pins the fixture's MaxMarkAgeSeconds/MaxFeedSilenceSeconds dependency: with 30 s the 900 s mark is stale
        // at the 931 s deadline, and no other mint is marked in between.
        BotOptions options = ReplayFixtures.Options();
        options.MaxMarkAgeSeconds = 30;
        options.MaxFeedSilenceSeconds = maxFeedSilenceSeconds;

        ReplayResult result = await ReplayHarness.RunAsync(options, Scenario());

        ClosedTrade timeout = result.Trades.Single(t => t.Entry.Mint == "to");
        Assert.Equal(ExitReason.Timeout, timeout.Reason);
        Assert.Equal(exit, timeout.ExitPrice);
    }

    [Fact]
    public async Task A_position_whose_deadline_is_past_the_last_event_is_left_open_not_force_closed()
    {
        // No event ever crosses the 901 s deadline, so the replay keeps the position open on event time alone.
        ReplayResult result = await Replay([Tick("m", 0, 0.0001m), Predict("m", 1)]);

        Assert.Empty(result.Trades);
        OpenPosition open = Assert.Single(result.OpenPositions);
        Assert.Equal("m", open.Mint);
        Assert.Equal(8m, result.FinalBalance);  // 2 locked in the open position
        Assert.Equal(10m, result.FinalEquity);  // marked flat at entry
    }

    [Fact]
    public async Task An_empty_stream_closes_nothing_and_keeps_the_starting_balance()
    {
        ReplayResult result = await Replay([]);

        Assert.Empty(result.Trades);
        Assert.Empty(result.OpenPositions);
        Assert.Equal(0, result.ClosedTradeCount);
        Assert.Equal(0.0, result.WinRate);
        Assert.Equal(10m, result.FinalBalance);
        Assert.Equal(10m, result.FinalEquity);
        Assert.Equal(0m, result.TotalPnl);
    }

    [Fact]
    public async Task RunAsync_rejects_null_arguments()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() => ReplayHarness.RunAsync(null!, []));
        await Assert.ThrowsAsync<ArgumentNullException>(() => ReplayHarness.RunAsync(new BotOptions(), null!));
    }

    public static TheoryData<Action<BotOptions>> InvalidOptions() =>
    [
        o => o.PositionTimeoutSeconds = -1,
        o => o.StartingBalance = 0m,
        o => o.MaxMarkAgeSeconds = 0,
        o => o.MaxEventLagSeconds = 3_601,
        o => o.MinStopLossPct = 0.5, // cross-field: above StopLossPct
    ];

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public async Task RunAsync_rejects_options_the_live_host_would_reject(Action<BotOptions> breakRule)
    {
        BotOptions options = ReplayFixtures.Options();
        breakRule(options);

        await Assert.ThrowsAsync<ValidationException>(() => ReplayHarness.RunAsync(options, Scenario()));
    }

    [Fact]
    public async Task RunAsync_rejects_an_unknown_order()
    {
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(
            () => ReplayHarness.RunAsync(ReplayFixtures.Options(), Scenario(), (ReplayOrder)42));
    }
}
