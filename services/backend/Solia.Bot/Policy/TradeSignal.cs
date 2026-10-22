namespace Solia.Bot.Policy;

/// <summary>
/// The directional call the policy layer derives from a <see cref="Domain.ModelPrediction"/> — a
/// <em>rule</em>, not a model output (see docs/ML-MODEL.md "Policy layer"). Only <see cref="Buy"/>
/// opens a position; <see cref="Sell"/> is the model flagging downside/rug (used by the exit policy in
/// a later task); <see cref="Hold"/> means no actionable edge.
/// </summary>
public enum TradeSignal
{
    Hold = 0,
    Buy = 1,
    Sell = 2,
}

public static class TradeSignalExtensions
{
    /// <summary>The value stored in <c>trades.strategy</c> / <c>predictions.strategy</c>.</summary>
    public static string ToCode(this TradeSignal signal) => signal switch
    {
        TradeSignal.Buy => "buy",
        TradeSignal.Sell => "sell",
        TradeSignal.Hold => "hold",
        _ => throw new ArgumentOutOfRangeException(nameof(signal), signal, "Unknown trade signal."),
    };
}
