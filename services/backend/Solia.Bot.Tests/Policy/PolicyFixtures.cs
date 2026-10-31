using Solia.Bot.Configuration;
using Solia.Bot.Domain;

namespace Solia.Bot.Tests.Policy;

/// <summary>Shared builders so each policy test states only the fields it cares about.</summary>
public static class PolicyFixtures
{
    /// <summary>
    /// Options with the production defaults (MinConfidence 60, MinPumpProb 0.25, MaxRugProb 0.2, …) but idealized
    /// fills (<see cref="BotOptions.RealisticFills"/> off: at the mark, no pool) and no execution realism (see
    /// <see cref="WithoutExecution"/>), so policy scenarios keep round numbers. <see cref="ProductionOptions"/> is the
    /// production default.
    /// </summary>
    internal static BotOptions Options() => WithoutExecution(new() { RealisticFills = false });

    /// <summary>The production pool fills (#165) without the execution realism of #175 (see <see cref="WithoutExecution"/>).</summary>
    internal static BotOptions RealisticOptions() => WithoutExecution(new());

    /// <summary>The production defaults, every realism knob included.</summary>
    internal static BotOptions ProductionOptions() => new();

    /// <summary>
    /// <paramref name="options"/> with every execution-realism knob at 0 — no latency, network fee, rent or slippage
    /// tolerance — which is the engine as it was before them (the regression baseline).
    /// </summary>
    internal static BotOptions WithoutExecution(BotOptions options)
    {
        options.EntryLatencyMs = 0;
        options.ExitLatencyMs = 0;
        options.NetworkFeeSol = 0m;
        options.AccountRentSol = 0m;
        options.MaxSlippagePct = 0;
        return options;
    }

    /// <summary>A prediction that clears every default gate: strong pump, low rug, high confidence, positive edge.</summary>
    internal static ModelPrediction Buyable() => new()
    {
        Mint = "mint1",
        Ts = new DateTimeOffset(2026, 1, 1, 0, 0, 0, TimeSpan.Zero),
        Horizon = "1m",
        PumpProb = 0.8,
        RugProb = 0.1,
        ExpectedReturn = 0.5,
        Confidence = 80,
    };

    /// <summary>Otherwise-buyable predictions with one field NaN, infinite or out of its contract range.</summary>
    public static TheoryData<ModelPrediction> Malformed() =>
    [
        Buyable() with { Confidence = double.NaN },
        Buyable() with { Confidence = 150 },
        Buyable() with { Confidence = -1 },
        Buyable() with { ExpectedReturn = double.NaN },
        Buyable() with { ExpectedReturn = double.PositiveInfinity },
        Buyable() with { ExpectedReturn = -1.5 },
        Buyable() with { RugProb = double.NaN },
        Buyable() with { RugProb = -0.1 },
        Buyable() with { RugProb = 1.1 },
        Buyable() with { PumpProb = double.NaN },
        Buyable() with { PumpProb = 1.5 },
    ];
}
