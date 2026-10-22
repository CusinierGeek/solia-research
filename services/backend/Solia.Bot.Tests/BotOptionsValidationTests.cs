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
    [InlineData(1.1, 0.2)]   // pump prob above range
    [InlineData(0.6, -0.1)]  // rug prob below range
    [InlineData(0.6, 1.1)]   // rug prob above range
    public void Out_of_range_entry_probabilities_are_rejected(double pump, double rug)
    {
        Assert.False(IsValid(new BotOptions { MinPumpProb = pump, MaxRugProb = rug }));
    }

    [Theory]
    [InlineData(0.3, 0.35)] // modest pump signal, looser rug gate — a coherent policy
    [InlineData(0.6, 0.6)]
    [InlineData(0.0, 1.0)]  // gates fully open
    public void Entry_thresholds_are_independent_of_each_other(double pump, double rug)
    {
        Assert.True(IsValid(new BotOptions { MinPumpProb = pump, MaxRugProb = rug }));
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
        Assert.True(IsValid(new BotOptions { StartingBalance = Dec(balance) }));
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
}
