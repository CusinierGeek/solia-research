using Solia.Bot.Domain;

namespace Solia.Bot.Policy;

/// <summary>
/// The single definition of a prediction the policy layer may act on. A malformed prediction — NaN or
/// infinite fields, probabilities outside [0, 1], confidence outside [0, 100], or an expected return
/// below −100% (impossible for a long position) — is a contract drift or a broken head, and every
/// policy component fails closed on it rather than letting it reach a comparison or a decimal cast.
/// </summary>
public static class PredictionValidation
{
    public static bool IsWellFormed(ModelPrediction prediction)
    {
        ArgumentNullException.ThrowIfNull(prediction);

        return IsProbability(prediction.PumpProb)
            && IsProbability(prediction.RugProb)
            && double.IsFinite(prediction.Confidence) && prediction.Confidence is >= 0.0 and <= 100.0
            && double.IsFinite(prediction.ExpectedReturn) && prediction.ExpectedReturn >= -1.0;
    }

    private static bool IsProbability(double p) => double.IsFinite(p) && p is >= 0.0 and <= 1.0;
}
