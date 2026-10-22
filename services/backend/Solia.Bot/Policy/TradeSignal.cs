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
