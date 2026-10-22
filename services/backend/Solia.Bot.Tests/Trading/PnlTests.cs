using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

/// <summary>The P&amp;L math of <see cref="ClosedTrade.Settle"/> on known numbers.</summary>
public class PnlTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static OpenPosition Position(decimal size, decimal entryPrice)
    {
        var opening = new TradeOpening(
            "mint1", T0, entryPrice, size,
            ExitLevels.Compute(PolicyFixtures.Buyable(), entryPrice, PolicyFixtures.Options()),
            PolicyFixtures.Buyable());
        return new OpenPosition(1, opening, T0.AddSeconds(900));
    }

    [Fact]
    public void A_win_realizes_the_price_gain_on_the_size()
    {
        ClosedTrade t = ClosedTrade.Settle(Position(2m, 0.0001m), T0.AddSeconds(42), 0.00015m, ExitReason.TakeProfit);

        Assert.Equal(1m, t.Pnl);
        Assert.Equal(0.5, t.PnlPct, 12);
        Assert.Equal(3m, t.Proceeds);
        Assert.Equal(42, t.DurationSec);
    }

    [Fact]
    public void A_loss_realizes_the_price_drop_on_the_size()
    {
        ClosedTrade t = ClosedTrade.Settle(Position(2m, 0.0001m), T0.AddSeconds(10), 0.00007m, ExitReason.StopLoss);

        Assert.Equal(-0.6m, t.Pnl);
        Assert.Equal(-0.3, t.PnlPct, 12);
        Assert.Equal(1.4m, t.Proceeds);
    }

    [Theory]
    [InlineData("2", "0.0001")]
    [InlineData("1.6", "0.0003")]     // size/entry isn't a terminating decimal — still exactly zero
    [InlineData("0.123", "0.0000007")]
    public void Break_even_is_exactly_zero(string size, string entry)
    {
        decimal entryPrice = decimal.Parse(entry, System.Globalization.CultureInfo.InvariantCulture);
        decimal sizeValue = decimal.Parse(size, System.Globalization.CultureInfo.InvariantCulture);

        ClosedTrade t = ClosedTrade.Settle(Position(sizeValue, entryPrice), T0.AddSeconds(5), entryPrice, ExitReason.Signal);

        Assert.Equal(0m, t.Pnl);
        Assert.Equal(0.0, t.PnlPct);
        Assert.Equal(sizeValue, t.Proceeds);
    }

    [Fact]
    public void Break_even_is_exactly_zero_even_when_size_times_price_exceeds_decimal_precision()
    {
        // 26 + 15 significant digits: size · entry can't be held exactly in a decimal.
        decimal size = 12345.678901234567890123456m;
        decimal entry = 0.000123456789012m;

        ClosedTrade t = ClosedTrade.Settle(Position(size, entry), T0.AddSeconds(5), entry, ExitReason.Signal);

        Assert.Equal(0m, t.Pnl);
        Assert.Equal(0.0, t.PnlPct);
        Assert.Equal(size, t.Proceeds);
    }

    [Fact]
    public void Pnl_is_rounded_to_money_precision()
    {
        ClosedTrade t = ClosedTrade.Settle(Position(1m, 0.0003m), T0.AddSeconds(5), 0.0004m, ExitReason.TakeProfit);

        Assert.Equal(0.333333333m, t.Pnl); // 1 · 4/3 − 1, to 9 places
    }

    [Fact]
    public void Timeout_at_entry_price_is_break_even_over_the_full_timeout()
    {
        ClosedTrade t = ClosedTrade.Settle(Position(2m, 0.0001m), T0.AddSeconds(900), 0.0001m, ExitReason.Timeout);

        Assert.Equal(0m, t.Pnl);
        Assert.Equal(900, t.DurationSec);
        Assert.Equal(ExitReason.Timeout, t.Reason);
    }

    [Fact]
    public void Duration_is_whole_seconds_truncated()
    {
        ClosedTrade t = ClosedTrade.Settle(Position(2m, 0.0001m), T0.AddMilliseconds(1999), 0.0001m, ExitReason.Signal);
        Assert.Equal(1, t.DurationSec);
    }

    [Theory]
    [InlineData(ExitReason.TakeProfit, "tp")]
    [InlineData(ExitReason.StopLoss, "sl")]
    [InlineData(ExitReason.Signal, "signal")]
    [InlineData(ExitReason.Timeout, "timeout")]
    public void Exit_reasons_map_to_the_schema_codes(ExitReason reason, string code)
    {
        Assert.Equal(code, reason.ToCode());
    }

    [Fact]
    public void An_unknown_exit_reason_has_no_code()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => ((ExitReason)99).ToCode());
    }
}
