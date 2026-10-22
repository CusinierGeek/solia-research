using Solia.Bot.Domain;
using Solia.Bot.Configuration;

namespace Solia.Bot.Policy;

/// <summary>
/// Turns a <see cref="ModelPrediction"/> into a <see cref="TradeSignal"/> against the configured
/// thresholds. Pure and deterministic — no I/O, no clock. The two probability heads are independent
/// gates (docs/ML-MODEL.md): a <see cref="TradeSignal.Buy"/> needs a strong pump head <em>and</em> a
/// low rug head; a confident, elevated rug head alone is a <see cref="TradeSignal.Sell"/>. A malformed
/// prediction (see <see cref="PredictionValidation"/>) or one that fails <see cref="IsConfident"/> is
/// never actionable, so it collapses to <see cref="TradeSignal.Hold"/>.
/// </summary>
public static class SignalClassifier
{
    public static TradeSignal Classify(ModelPrediction prediction, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(options);

        // Malformed or low-confidence predictions carry no actionable edge regardless of the heads.
        if (!PredictionValidation.IsWellFormed(prediction) || !IsConfident(prediction, options))
        {
            return TradeSignal.Hold;
        }

        if (prediction.PumpProb >= options.MinPumpProb && prediction.RugProb <= options.MaxRugProb)
        {
            return TradeSignal.Buy;
        }

        if (prediction.RugProb > options.MaxRugProb)
        {
            return TradeSignal.Sell;
        }

        return TradeSignal.Hold;
    }

    /// <summary>
    /// The single confidence gate (inclusive at <see cref="BotOptions.MinConfidence"/>), shared with
    /// <see cref="EntryRule"/> so the two can't drift. Fails closed: <c>&gt;=</c> is false for NaN, so a
    /// NaN confidence fails the gate rather than slipping past a <c>&lt;</c> comparison.
    /// </summary>
    public static bool IsConfident(ModelPrediction prediction, BotOptions options)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(options);

        return prediction.Confidence >= options.MinConfidence;
    }
}
