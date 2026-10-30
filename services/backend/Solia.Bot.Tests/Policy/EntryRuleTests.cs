using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Policy;

namespace Solia.Bot.Tests.Policy;

public class EntryRuleTests
{
    private static readonly BotOptions Opts = PolicyFixtures.Options();

    /// <summary>A portfolio with room to open the trade: empty book, cash on hand, a live price.</summary>
    private static PortfolioState HasCapacity() => new(
        Balance: 10m,
        OpenPositionCount: 0,
        HoldsMint: false,
        MarketPrice: 0.0001m);

    [Fact]
    public void A_buyable_prediction_with_capacity_enters()
    {
        EntryDecision decision = EntryRule.Evaluate(PolicyFixtures.Buyable(), HasCapacity(), Opts);

        Assert.True(decision.Enter);
        Assert.Equal(EntryRejectReason.None, decision.Reason);
    }

    [Fact]
    public void An_accept_carries_the_size_computed_from_the_same_portfolio()
    {
        EntryDecision decision = EntryRule.Evaluate(PolicyFixtures.Buyable(), HasCapacity(), Opts);

        Assert.True(decision.Size > 0m);
        Assert.Equal(PositionSizer.Size(PolicyFixtures.Buyable(), HasCapacity(), Opts), decision.Size);
    }

    [Fact]
    public void A_reject_carries_a_zero_size()
    {
        PortfolioState held = HasCapacity() with { HoldsMint = true };
        Assert.Equal(0m, EntryRule.Evaluate(PolicyFixtures.Buyable(), held, Opts).Size);
    }

    [Theory]
    [MemberData(nameof(PolicyFixtures.Malformed), MemberType = typeof(PolicyFixtures))]
    public void A_malformed_prediction_is_rejected_as_invalid(ModelPrediction malformed)
    {
        EntryDecision decision = EntryRule.Evaluate(malformed, HasCapacity(), Opts);

        Assert.False(decision.Enter);
        Assert.Equal(EntryRejectReason.InvalidPrediction, decision.Reason);
    }

    [Fact]
    public void Confidence_exactly_at_the_threshold_passes_the_gate()
    {
        ModelPrediction atMin = PolicyFixtures.Buyable() with { Confidence = Opts.MinConfidence };
        Assert.True(EntryRule.Evaluate(atMin, HasCapacity(), Opts).Enter);
    }

    [Fact]
    public void Low_confidence_is_rejected_before_the_signal_is_even_checked()
    {
        ModelPrediction unsure = PolicyFixtures.Buyable() with { Confidence = 10 };
        Assert.Equal(EntryRejectReason.LowConfidence, EntryRule.Evaluate(unsure, HasCapacity(), Opts).Reason);
    }

    [Fact]
    public void A_non_buy_signal_is_rejected()
    {
        // Confident enough to pass the confidence gate, but the pump head is too weak to be a Buy.
        ModelPrediction hold = PolicyFixtures.Buyable() with { PumpProb = 0.1 };
        EntryDecision decision = EntryRule.Evaluate(hold, HasCapacity(), Opts);

        Assert.False(decision.Enter);
        Assert.Equal(EntryRejectReason.NotABuySignal, decision.Reason);
    }

    [Fact]
    public void A_full_book_is_rejected()
    {
        PortfolioState full = HasCapacity() with { OpenPositionCount = Opts.MaxConcurrentPositions };
        Assert.Equal(EntryRejectReason.AtMaxConcurrentPositions, EntryRule.Evaluate(PolicyFixtures.Buyable(), full, Opts).Reason);
    }

    [Fact]
    public void An_already_held_mint_is_rejected()
    {
        PortfolioState held = HasCapacity() with { HoldsMint = true };
        Assert.Equal(EntryRejectReason.AlreadyHoldingMint, EntryRule.Evaluate(PolicyFixtures.Buyable(), held, Opts).Reason);
    }

    [Theory]
    [InlineData(null)]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    public void A_missing_or_non_positive_price_is_rejected(double? price)
    {
        PortfolioState noPrice = HasCapacity() with { MarketPrice = price is null ? null : (decimal)price.Value };
        Assert.Equal(EntryRejectReason.NoMarketPrice, EntryRule.Evaluate(PolicyFixtures.Buyable(), noPrice, Opts).Reason);
    }

    [Fact]
    public void A_balance_below_the_dust_floor_is_rejected()
    {
        PortfolioState broke = HasCapacity() with { Balance = Opts.MinPositionSize / 2m };
        Assert.Equal(EntryRejectReason.InsufficientBalance, EntryRule.Evaluate(PolicyFixtures.Buyable(), broke, Opts).Reason);
    }

    [Fact]
    public void A_zero_balance_is_rejected_even_with_the_dust_floor_disabled()
    {
        var noFloor = new BotOptions { MinPositionSize = 0m, RealisticFills = false };
        PortfolioState broke = HasCapacity() with { Balance = 0m };
        Assert.Equal(EntryRejectReason.InsufficientBalance, EntryRule.Evaluate(PolicyFixtures.Buyable(), broke, noFloor).Reason);
    }

    [Theory]
    [InlineData(-0.2)] // expected loss
    [InlineData(0.0)]  // no expected gain
    [InlineData(0.01)] // below the MinTakeProfitPct floor
    public void A_buy_signal_without_a_positive_edge_is_rejected(double expectedReturn)
    {
        // The heads say Buy, but the sized bet has no edge — an explicit reason, not an Accept of size 0.
        ModelPrediction noEdge = PolicyFixtures.Buyable() with { ExpectedReturn = expectedReturn };
        Assert.Equal(TradeSignal.Buy, SignalClassifier.Classify(noEdge, Opts));

        EntryDecision decision = EntryRule.Evaluate(noEdge, HasCapacity(), Opts);
        Assert.False(decision.Enter);
        Assert.Equal(EntryRejectReason.NoPositiveEdge, decision.Reason);
    }

    [Fact]
    public void A_size_below_the_dust_floor_is_rejected()
    {
        // Balance clears the floor (0.004 ≥ 0.001) but the capped size (0.004 · 0.2 = 0.0008) does not.
        PortfolioState small = HasCapacity() with { Balance = 0.004m };
        EntryDecision decision = EntryRule.Evaluate(PolicyFixtures.Buyable(), small, Opts);

        Assert.False(decision.Enter);
        Assert.Equal(EntryRejectReason.BelowMinPositionSize, decision.Reason);
    }

    [Fact]
    public void A_malformed_prediction_takes_precedence_over_every_other_gate()
    {
        ModelPrediction broken = PolicyFixtures.Buyable() with { Confidence = double.NaN };
        PortfolioState bad = new(Balance: 0m, OpenPositionCount: 999, HoldsMint: true, MarketPrice: null);
        Assert.Equal(EntryRejectReason.InvalidPrediction, EntryRule.Evaluate(broken, bad, Opts).Reason);
    }

    [Fact]
    public void Earlier_gates_take_precedence_over_later_ones()
    {
        // Low confidence AND a full book AND no price: the first gate (confidence) is the reported reason.
        ModelPrediction unsure = PolicyFixtures.Buyable() with { Confidence = 0 };
        PortfolioState bad = new(Balance: 0m, OpenPositionCount: 999, HoldsMint: true, MarketPrice: null);
        Assert.Equal(EntryRejectReason.LowConfidence, EntryRule.Evaluate(unsure, bad, Opts).Reason);
    }

    [Fact]
    public void Null_arguments_throw()
    {
        Assert.Throws<ArgumentNullException>(() => EntryRule.Evaluate(null!, HasCapacity(), Opts));
        Assert.Throws<ArgumentNullException>(() => EntryRule.Evaluate(PolicyFixtures.Buyable(), HasCapacity(), null!));
    }

    [Theory]
    [InlineData("raydium", 0.03)]   // AMM dust pool
    [InlineData("pumpswap", 0.99)]  // just below the 1 SOL floor
    [InlineData(null, 0.5)]         // unknown venue: the default floor
    [InlineData("pumpfun", null)]   // unknown liquidity, even on a curve
    public void With_realistic_fills_a_pool_below_its_venue_floor_is_rejected(string? venue, double? liquidity)
    {
        PortfolioState thin = HasCapacity() with { Pool = new Pool(venue, (decimal?)liquidity) };
        EntryDecision decision = EntryRule.Evaluate(PolicyFixtures.Buyable(), thin, PolicyFixtures.RealisticOptions());

        Assert.Equal(EntryRejectReason.InsufficientLiquidity, decision.Reason);
    }

    [Theory]
    [InlineData("pumpswap", 1000.0, 2.0)] // the balance cap binds
    [InlineData("pumpswap", 25.0, 0.5)]   // the pool cap binds: 2 % of 25
    [InlineData("pumpfun", 0.02, 0.6004)] // curve with ~0 real SOL: 2 % of its 30.02 SOL reserve
    public void With_realistic_fills_a_pool_at_or_above_its_floor_enters_with_a_pool_sized_position(
        string venue, double liquidity, double size)
    {
        PortfolioState pool = HasCapacity() with { Pool = new Pool(venue, (decimal)liquidity) };
        EntryDecision decision = EntryRule.Evaluate(PolicyFixtures.Buyable(), pool, PolicyFixtures.RealisticOptions());

        Assert.True(decision.Enter);
        Assert.Equal((decimal)size, decision.Size);
    }

    [Fact]
    public void A_pool_too_shallow_for_the_dust_floor_is_skipped()
    {
        BotOptions options = PolicyFixtures.RealisticOptions();
        options.LiquidityFloorSol["raydium"] = 0m;
        PortfolioState shallow = HasCapacity() with { Pool = new Pool("raydium", 0.04m) }; // cap 0.0008 < 0.001

        Assert.Equal(EntryRejectReason.BelowMinPositionSize, EntryRule.Evaluate(PolicyFixtures.Buyable(), shallow, options).Reason);
    }

    [Fact]
    public void The_liquidity_gate_follows_the_market_price_gate()
    {
        PortfolioState noMarket = HasCapacity() with { MarketPrice = null, Pool = new Pool("raydium", 0.01m) };
        EntryDecision decision = EntryRule.Evaluate(PolicyFixtures.Buyable(), noMarket, PolicyFixtures.RealisticOptions());

        Assert.Equal(EntryRejectReason.NoMarketPrice, decision.Reason);
    }

    [Fact]
    public void Idealized_fills_skip_the_liquidity_gate()
    {
        PortfolioState dust = HasCapacity() with { Pool = new Pool("raydium", 0.01m) };
        Assert.True(EntryRule.Evaluate(PolicyFixtures.Buyable(), dust, Opts).Enter);
    }
}
