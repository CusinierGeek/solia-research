using System.Globalization;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Solia.Bot.Configuration;

namespace Solia.Bot.Tests;

public class BotOptionsValidationTests
{
    // The validators AddBotOptions registers — so these tests exercise production's rule set, not a copy.
    private static readonly IValidateOptions<BotOptions>[] Validators = RegisteredValidators();

    private static IValidateOptions<BotOptions>[] RegisteredValidators()
    {
        var services = new ServiceCollection();
        services.AddBotOptions(new ConfigurationBuilder().Build());
        return services.BuildServiceProvider().GetServices<IValidateOptions<BotOptions>>().ToArray();
    }

    private static bool IsValid(BotOptions o) =>
        Validators.All(v => v.Validate(Options.DefaultName, o).Succeeded);

    private static decimal Dec(string s) =>
        decimal.Parse(s, NumberStyles.Float, CultureInfo.InvariantCulture);

    [Fact]
    public void Defaults_are_valid()
    {
        Assert.NotEmpty(Validators);
        Assert.True(IsValid(new BotOptions()));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1001)] // above range
    public void Out_of_range_max_concurrent_positions_is_rejected(int maxPositions)
    {
        Assert.False(IsValid(new BotOptions { MaxConcurrentPositions = maxPositions }));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(100.1)]
    public void Out_of_range_confidence_is_rejected(double confidence)
    {
        Assert.False(IsValid(new BotOptions { MinConfidence = confidence }));
    }

    [Theory]
    [InlineData(-0.1, 0.2)]  // pump prob below range
    [InlineData(0.0, 0.2)]   // a zero pump gate would enter on every eligible token
    [InlineData(1.1, 0.2)]   // pump prob above range
    [InlineData(double.NaN, 0.2)]
    [InlineData(0.6, -0.1)]  // rug prob below range
    [InlineData(0.6, 1.1)]   // rug prob above range
    public void Out_of_range_entry_probabilities_are_rejected(double pump, double rug)
    {
        Assert.False(IsValid(new BotOptions { MinPumpProb = pump, MaxRugProb = rug }));
    }

    [Theory]
    [InlineData(0.3, 0.35)] // modest pump signal, looser rug gate — a coherent policy
    [InlineData(0.6, 0.6)]
    [InlineData(0.01, 1.0)] // gates nearly fully open
    public void Entry_thresholds_are_independent_of_each_other(double pump, double rug)
    {
        Assert.True(IsValid(new BotOptions { MinPumpProb = pump, MaxRugProb = rug }));
    }

    [Fact]
    public void Default_entry_rule_is_the_swept_one()
    {
        // The entry-rule sweep's choice on the champion's held-out windows (docs/ML-MODEL.md
        // "Choosing the entry rule"): pump ≥ 0.4 at 15m, no rug gate, no confidence gate; the
        // ai-service's ReferencePolicy defaults are the same values.
        var options = new BotOptions();

        Assert.Equal(0.4, options.MinPumpProb);
        Assert.Equal(1.0, options.MaxRugProb);
        Assert.Equal(0, options.MinConfidence);
        Assert.True(IsValid(options));
    }

    [Theory]
    [InlineData(0.4, 0.0)]
    [InlineData(0.4, 0.5)]
    [InlineData(0.5, 0.5)]
    [InlineData(0.9, 1.0)]
    public void Default_gates_admit_any_rug_prob_once_pump_clears_the_threshold(double pump, double rug)
    {
        // No rug gate and no confidence gate: even pump and rug both at 0.5 (confidence 0) enter.
        var options = new BotOptions();
        double confidence = 200 * Math.Max(Math.Abs(pump - 0.5), Math.Abs(rug - 0.5));

        Assert.True(pump >= options.MinPumpProb);
        Assert.True(rug <= options.MaxRugProb);
        Assert.True(confidence >= options.MinConfidence);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(86_401)] // above range
    public void Out_of_range_position_timeout_is_rejected(int timeout)
    {
        Assert.False(IsValid(new BotOptions { PositionTimeoutSeconds = timeout }));
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-5")]
    [InlineData("1000000.01")] // above the sanity cap
    [InlineData("1e20")]
    public void Out_of_range_starting_balance_is_rejected(string balance)
    {
        Assert.False(IsValid(new BotOptions { StartingBalance = Dec(balance) }));
    }

    [Theory]
    [InlineData("0.01")]
    [InlineData("1000000")] // the cap itself is allowed
    public void In_range_starting_balance_is_accepted(string balance)
    {
        // The dust floor off: a 0.01 balance can't open a 0.01 position.
        Assert.True(IsValid(new BotOptions { StartingBalance = Dec(balance), MinPositionSize = 0m }));
    }

    [Fact]
    public void Zero_sizing_cap_is_rejected()
    {
        Assert.False(IsValid(new BotOptions { MaxPositionFraction = 0 }));
        Assert.True(IsValid(new BotOptions { MaxPositionFraction = 1 }));
    }

    [Theory]
    [InlineData(0)]   // zero would make every position size to nothing
    [InlineData(1.1)] // above full Kelly
    public void Out_of_range_risk_fraction_is_rejected(double riskFraction)
    {
        Assert.False(IsValid(new BotOptions { RiskFraction = riskFraction }));
    }

    [Fact]
    public void Risk_fraction_bounds_are_accepted()
    {
        Assert.True(IsValid(new BotOptions { RiskFraction = 1.0 }));
        Assert.True(IsValid(new BotOptions { RiskFraction = 0.01 }));
    }

    [Theory]
    [InlineData("-0.01")]
    [InlineData("1000000.01")] // above the sanity cap
    public void Out_of_range_min_position_size_is_rejected(string minSize)
    {
        Assert.False(IsValid(new BotOptions { MinPositionSize = Dec(minSize) }));
    }

    [Fact]
    public void Zero_min_position_size_is_accepted()
    {
        Assert.True(IsValid(new BotOptions { MinPositionSize = 0m })); // dust floor disabled
    }

    [Fact]
    public void A_min_position_size_above_the_largest_possible_position_is_rejected()
    {
        // Largest position = 10 · 0.2 = 2: a floor of 5 would make every trade dust.
        Assert.False(IsValid(new BotOptions { StartingBalance = 10m, MaxPositionFraction = 0.2, MinPositionSize = 5m }));
    }

    [Fact]
    public void A_min_position_size_equal_to_the_largest_possible_position_is_accepted()
    {
        Assert.True(IsValid(new BotOptions { StartingBalance = 10m, MaxPositionFraction = 0.2, MinPositionSize = 2m }));
    }

    [Theory]
    [InlineData(0, 0.3)]
    [InlineData(0.5, 0)]
    public void Zero_take_profit_or_stop_loss_is_rejected(double takeProfit, double stopLoss)
    {
        Assert.False(IsValid(new BotOptions { TakeProfitPct = takeProfit, StopLossPct = stopLoss }));
    }

    [Theory]
    [InlineData(1.0)] // stop at a zero price: unreachable
    [InlineData(1.5)] // stop below zero: unreachable
    public void Stop_loss_of_100_percent_or_more_is_rejected(double stopLoss)
    {
        Assert.False(IsValid(new BotOptions { StopLossPct = stopLoss }));
    }

    [Fact]
    public void Take_profit_above_100_percent_is_accepted()
    {
        Assert.True(IsValid(new BotOptions { TakeProfitPct = 2.0, StopLossPct = 0.99 }));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(100.1)]
    public void Out_of_range_stop_loss_volatility_multiple_is_rejected(double multiple)
    {
        Assert.False(IsValid(new BotOptions { StopLossVolatilityMultiple = multiple }));
    }

    [Fact]
    public void Zero_stop_loss_volatility_multiple_disables_the_dynamic_stop_and_is_valid()
    {
        Assert.True(IsValid(new BotOptions { StopLossVolatilityMultiple = 0 }));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void Out_of_range_min_stop_loss_is_rejected(double minStop)
    {
        Assert.False(IsValid(new BotOptions { MinStopLossPct = minStop, StopLossPct = 0.99 }));
    }

    [Fact]
    public void Min_stop_loss_above_the_stop_loss_cap_is_rejected()
    {
        Assert.False(IsValid(new BotOptions { MinStopLossPct = 0.4, StopLossPct = 0.3 }));
    }

    [Fact]
    public void Min_stop_loss_equal_to_the_cap_is_a_fixed_stop_and_is_valid()
    {
        Assert.True(IsValid(new BotOptions { MinStopLossPct = 0.3, StopLossPct = 0.3 }));
    }

    [Theory]
    [InlineData(-0.01)]
    [InlineData(10.01)]
    public void Out_of_range_min_take_profit_is_rejected(double minTakeProfit)
    {
        Assert.False(IsValid(new BotOptions { MinTakeProfitPct = minTakeProfit, TakeProfitPct = 10 }));
    }

    [Fact]
    public void Min_take_profit_above_the_take_profit_cap_is_rejected()
    {
        Assert.False(IsValid(new BotOptions { MinTakeProfitPct = 0.6, TakeProfitPct = 0.5 }));
    }

    [Theory]
    [InlineData(0.0)] // floor disabled
    [InlineData(0.5)] // floor at the cap
    public void Min_take_profit_within_the_cap_is_valid(double minTakeProfit)
    {
        Assert.True(IsValid(new BotOptions { MinTakeProfitPct = minTakeProfit, TakeProfitPct = 0.5 }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(86_401)]
    public void Out_of_range_max_mark_age_is_rejected(int seconds)
    {
        Assert.False(IsValid(new BotOptions { MaxMarkAgeSeconds = seconds }));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(86_401)]
    public void Out_of_range_max_feed_silence_is_rejected(int seconds)
    {
        Assert.False(IsValid(new BotOptions { MaxFeedSilenceSeconds = seconds }));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(3_601)]
    public void Out_of_range_max_event_lag_is_rejected(int seconds)
    {
        Assert.False(IsValid(new BotOptions { MaxEventLagSeconds = seconds }));
    }

    [Fact]
    public void Zero_event_lag_means_strict_ordering_and_is_valid()
    {
        Assert.True(IsValid(new BotOptions { MaxEventLagSeconds = 0 }));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(3_601)]
    public void Out_of_range_timeout_sweep_is_rejected(int seconds)
    {
        Assert.False(IsValid(new BotOptions { TimeoutSweepSeconds = seconds }));
    }

    [Theory]
    [InlineData(0)] // disabled: deterministic replay
    [InlineData(5)]
    [InlineData(3_600)]
    public void In_range_timeout_sweep_is_valid(int seconds)
    {
        Assert.True(IsValid(new BotOptions { TimeoutSweepSeconds = seconds }));
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.01)]
    [InlineData(double.NaN)]
    public void Out_of_range_max_pool_fraction_is_rejected(double fraction)
    {
        Assert.False(IsValid(new BotOptions { MaxPoolFraction = fraction }));
    }

    [Theory]
    [InlineData("LiquidityFloorSol", "default", "-1")]
    [InlineData("LiquidityFloorSol", "raydium", "1000001")]
    [InlineData("LiquidityFloorSol", " ", "1")]          // empty venue key
    [InlineData("VirtualSol", "pumpfun", "-30")]
    [InlineData("FeePct", "pumpswap", "-0.001")]
    [InlineData("FeePct", "default", "0.6")]               // above 50 % per side: a typo (60 for 0.6 %)
    [InlineData("FeePct", "default", "NaN")]
    public void Out_of_range_venue_tables_are_rejected(string table, string venue, string value)
    {
        var options = new BotOptions();
        switch (table)
        {
            case "LiquidityFloorSol":
                options.LiquidityFloorSol[venue] = Dec(value);
                break;
            case "VirtualSol":
                options.VirtualSol[venue] = Dec(value);
                break;
            default:
                options.FeePct[venue] = double.Parse(value, CultureInfo.InvariantCulture);
                break;
        }

        Assert.False(IsValid(options));
    }

    [Fact]
    public void A_missing_venue_table_is_rejected_but_an_empty_one_is_valid()
    {
        Assert.False(IsValid(new BotOptions { FeePct = null! }));
        Assert.True(IsValid(new BotOptions { FeePct = [], LiquidityFloorSol = [], VirtualSol = [] }));
    }

    [Fact]
    public void Idealized_fills_are_valid()
    {
        Assert.True(IsValid(new BotOptions { RealisticFills = false }));
    }

    // --- Execution realism (#175) ---

    [Theory]
    [InlineData(-1, 0)]
    [InlineData(60_001, 0)]
    [InlineData(0, -1)]
    [InlineData(0, 60_001)]
    public void Out_of_range_latencies_are_rejected(int entryMs, int exitMs)
    {
        Assert.False(IsValid(new BotOptions { EntryLatencyMs = entryMs, ExitLatencyMs = exitMs }));
    }

    [Theory]
    [InlineData("-0.0001", "0")]
    [InlineData("1.5", "0")]
    [InlineData("0", "-0.001")]
    [InlineData("0", "2")]
    public void Out_of_range_network_costs_are_rejected(string fee, string rent)
    {
        Assert.False(IsValid(new BotOptions { NetworkFeeSol = Dec(fee), AccountRentSol = Dec(rent) }));
    }

    [Theory]
    [InlineData(-0.01, 3)]
    [InlineData(1.0, 3)] // a 100 % tolerance would accept any fill: disable it with 0 instead
    [InlineData(0.15, -1)]
    [InlineData(0.15, 101)]
    public void Out_of_range_slippage_tolerance_or_retries_are_rejected(double slippage, int retries)
    {
        Assert.False(IsValid(new BotOptions { MaxSlippagePct = slippage, MaxExitRetries = retries }));
    }

    [Fact]
    public void Every_execution_knob_at_zero_is_valid()
    {
        Assert.True(IsValid(new BotOptions
        {
            EntryLatencyMs = 0,
            ExitLatencyMs = 0,
            NetworkFeeSol = 0m,
            AccountRentSol = 0m,
            MaxSlippagePct = 0,
            MaxExitRetries = 0,
        }));
    }

    [Theory]
    [InlineData(60_000, 0, 60)]
    [InlineData(0, 30_000, 30)]
    public void A_latency_as_long_as_the_position_timeout_is_rejected(int entryMs, int exitMs, int timeoutSeconds)
    {
        Assert.False(IsValid(new BotOptions { EntryLatencyMs = entryMs, ExitLatencyMs = exitMs, PositionTimeoutSeconds = timeoutSeconds }));
        Assert.True(IsValid(new BotOptions
        {
            EntryLatencyMs = Math.Max(0, entryMs - 1),
            ExitLatencyMs = Math.Max(0, exitMs - 1),
            PositionTimeoutSeconds = timeoutSeconds,
        }));
    }

    [Fact]
    public void A_dust_floor_below_the_round_trip_fixed_cost_is_rejected()
    {
        // 2 · 0.001 + 0.002 = 0.004 > 0.003: the smallest trade would lose more than its size.
        Assert.False(IsValid(new BotOptions { MinPositionSize = 0.003m, NetworkFeeSol = 0.001m, AccountRentSol = 0.002m }));
        Assert.True(IsValid(new BotOptions { MinPositionSize = 0.004m, NetworkFeeSol = 0.001m, AccountRentSol = 0.002m }));
        Assert.True(IsValid(new BotOptions { MinPositionSize = 0m, NetworkFeeSol = 0.001m, AccountRentSol = 0.002m })); // floor off
    }

    [Fact]
    public void The_default_round_trip_fixed_cost_is_two_fees_and_the_rent()
    {
        Assert.Equal(2m * 0.0002m + 0.00203928m, new BotOptions().RoundTripFixedCostSol());
    }
}
