namespace Solia.Bot.Policy;

/// <summary>
/// The portfolio snapshot the <see cref="EntryRule"/> needs to decide whether a new position has
/// capacity — supplied by the engine (a later task) at decision time, which keeps the rule pure (no
/// state, no I/O). Money/price are <see cref="decimal"/> per the Solia convention.
/// </summary>
/// <param name="Balance">Cash currently available to open positions with.</param>
/// <param name="OpenPositionCount">How many positions are open right now.</param>
/// <param name="HoldsMint">Whether a position for the candidate mint is already open.</param>
/// <param name="MarketPrice">
/// Latest price the order could fill against, or null when none is known yet. A non-positive price is
/// treated as "no market".
/// </param>
public readonly record struct PortfolioState(
    decimal Balance,
    int OpenPositionCount,
    bool HoldsMint,
    decimal? MarketPrice);
