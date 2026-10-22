namespace Solia.Bot.Trading;

/// <summary>
/// Why a position was closed — mirrors the <c>trades.exit_reason</c> CHECK constraint
/// (<c>tp</c> / <c>sl</c> / <c>signal</c> / <c>timeout</c>, docs/DATA-MODEL.md). See
/// <see cref="PositionManager"/> for the precedence between them.
/// </summary>
public enum ExitReason
{
    /// <summary>The marked price reached the take-profit level.</summary>
    TakeProfit,

    /// <summary>The marked price reached the stop-loss level.</summary>
    StopLoss,

    /// <summary>A fresh <see cref="Policy.TradeSignal.Sell"/> arrived for the held mint.</summary>
    Signal,

    /// <summary>The position reached <see cref="Configuration.BotOptions.PositionTimeoutSeconds"/>.</summary>
    Timeout,
}

public static class ExitReasonExtensions
{
    /// <summary>The value stored in <c>trades.exit_reason</c>.</summary>
    public static string ToCode(this ExitReason reason) => reason switch
    {
        ExitReason.TakeProfit => "tp",
        ExitReason.StopLoss => "sl",
        ExitReason.Signal => "signal",
        ExitReason.Timeout => "timeout",
        _ => throw new ArgumentOutOfRangeException(nameof(reason), reason, "Unknown exit reason."),
    };
}
