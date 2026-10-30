namespace Solia.Bot.Policy;

/// <summary>Why the <see cref="EntryRule"/> declined to enter. <see cref="None"/> accompanies an accept.</summary>
public enum EntryRejectReason
{
    None = 0,

    /// <summary>The prediction is malformed (NaN/infinite or out-of-range fields — see <see cref="PredictionValidation"/>).</summary>
    InvalidPrediction,

    /// <summary>Model confidence is below <see cref="Configuration.BotOptions.MinConfidence"/>.</summary>
    LowConfidence,

    /// <summary>The classified signal was not <see cref="TradeSignal.Buy"/>.</summary>
    NotABuySignal,

    /// <summary>Already at <see cref="Configuration.BotOptions.MaxConcurrentPositions"/>.</summary>
    AtMaxConcurrentPositions,

    /// <summary>A position for this mint is already open.</summary>
    AlreadyHoldingMint,

    /// <summary>No positive market price to fill against.</summary>
    NoMarketPrice,

    /// <summary>
    /// The pool's liquidity is unknown or below its venue's <see cref="Configuration.BotOptions.LiquidityFloorSol"/>
    /// (only with <see cref="Configuration.BotOptions.RealisticFills"/>).
    /// </summary>
    InsufficientLiquidity,

    /// <summary>Balance is not positive, or is below <see cref="Configuration.BotOptions.MinPositionSize"/>.</summary>
    InsufficientBalance,

    /// <summary>The sized bet has no positive expected edge (<see cref="PositionSizer.Fraction"/> is 0).</summary>
    NoPositiveEdge,

    /// <summary>
    /// The computed size — capped by the pool (<see cref="Configuration.BotOptions.MaxPoolFraction"/>) — falls below
    /// <see cref="Configuration.BotOptions.MinPositionSize"/> (dust).
    /// </summary>
    BelowMinPositionSize,
}

/// <summary>
/// The <see cref="EntryRule"/>'s verdict: either enter with a positive <see cref="Size"/>, or decline
/// with an explicit <see cref="EntryRejectReason"/> (and a zero size). A value type — the rule is pure.
/// </summary>
public readonly record struct EntryDecision(bool Enter, EntryRejectReason Reason, decimal Size)
{
    /// <summary>Enter the trade, committing <paramref name="size"/>.</summary>
    public static EntryDecision Accept(decimal size) => new(true, EntryRejectReason.None, size);

    /// <summary>Decline, carrying the first gate that failed.</summary>
    public static EntryDecision Reject(EntryRejectReason reason) => new(false, reason, 0m);
}
