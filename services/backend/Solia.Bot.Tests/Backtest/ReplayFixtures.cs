using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Tests.Policy;

namespace Solia.Bot.Tests.Backtest;

/// <summary>
/// The canonical replay scenario and the event builders it is made of. Four trades on four mints, each
/// opened and closed in isolation (one position at a time) so the sizing — 20% of the balance at entry —
/// is a clean function of the running balance, and every trade exits by a different reason:
/// <c>tp</c>, <c>sl</c>, <c>signal</c>, <c>timeout</c>. With the default options a
/// <see cref="PolicyFixtures.Buyable"/> prediction filled at 0.0001 takes TP at 0.00015 (+50%), SL at
/// 0.00007 (−30%, no volatility) and times out after 900 s. The exact numbers the replay must reproduce
/// are asserted in <see cref="ReplayHarnessTests"/>.
/// </summary>
public static class ReplayFixtures
{
    /// <summary>
    /// The options the scenario's numbers are computed under: the defaults, with the values the scenario
    /// depends on pinned explicitly so a change of default can't silently move them.
    /// <see cref="BotOptions.MaxMarkAgeSeconds"/> must be at least 31 s: the <c>to</c> trade's deadline
    /// (931 s) is 31 s after its last mark (900 s); a staler mark still fills at 0.00012 while the feed is live
    /// (<see cref="BotOptions.MaxFeedSilenceSeconds"/> ≥ 31 s — the 900 s tick is also the feed's last), and at the
    /// stop once it is silent. Fills are idealized (at the mark, no pool): the scenario pins the engine's
    /// bookkeeping, the realistic fills are covered by <c>FillsTests</c> and the engine's realistic-fill tests.
    /// </summary>
    public static BotOptions Options() => PolicyFixtures.WithoutExecution(PolicyFixtures.WithFixtureGates(new()
    {
        RealisticFills = false,
        StartingBalance = 10m,
        MaxPositionFraction = 0.2,
        PositionTimeoutSeconds = 900,
        MaxMarkAgeSeconds = 60,
        MaxFeedSilenceSeconds = 60,
        MaxEventLagSeconds = 5,
    }));

    /// <summary>The replay clock's origin; every event timestamp is an offset from it.</summary>
    public static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    /// <summary>A price observation for <paramref name="mint"/> at <c>T0 + sec</c>.</summary>
    public static BotEvent Tick(string mint, int sec, decimal price) =>
        new BotEvent.Price(new PriceTick { Mint = mint, Ts = T0.AddSeconds(sec), Price = price });

    /// <summary>A buyable prediction for <paramref name="mint"/> at <c>T0 + sec</c> (unless <paramref name="prediction"/> overrides it).</summary>
    public static BotEvent Predict(string mint, int sec, ModelPrediction? prediction = null) =>
        new BotEvent.Prediction((prediction ?? PolicyFixtures.Buyable()) with { Mint = mint, Ts = T0.AddSeconds(sec) });

    /// <summary>A confident prediction with an elevated rug head — classified as a sell.</summary>
    public static ModelPrediction SellSignal() => PolicyFixtures.Buyable() with { PumpProb = 0.1, RugProb = 0.6 };

    /// <summary>
    /// The scenario, in timestamp order (all timestamps are distinct), so it replays identically in
    /// <see cref="Solia.Bot.Backtest.ReplayOrder.Arrival"/> and <see cref="Solia.Bot.Backtest.ReplayOrder.Timestamp"/>
    /// order; the late-event and tie-break rules are covered separately.
    /// </summary>
    public static IReadOnlyList<BotEvent> Scenario() =>
    [
        // tp: opens at 2 (20% of 10), tick gaps to +50% and takes profit.
        Tick("tp", 0, 0.0001m),
        Predict("tp", 1),
        Tick("tp", 2, 0.00015m),

        // sl: opens at 2.2 (20% of 11), tick hits the stop at −30%.
        Tick("sl", 10, 0.0001m),
        Predict("sl", 11),
        Tick("sl", 12, 0.00007m),

        // sig: opens at 2.068 (20% of 10.34), marks +10%, then a sell signal closes it there.
        Tick("sig", 20, 0.0001m),
        Predict("sig", 21),
        Tick("sig", 22, 0.00011m),
        Predict("sig", 23, SellSignal()),

        // to: opens at 2.10936 (20% of 10.5468), a fresh +20% mark near the deadline, then times out at it.
        Tick("to", 30, 0.0001m),
        Predict("to", 31),
        Tick("to", 900, 0.00012m),

        // A later event on an untraded mint advances the clock past the 931 s deadline and sweeps the timeout.
        Tick("end", 1000, 0.0001m),
    ];
}
