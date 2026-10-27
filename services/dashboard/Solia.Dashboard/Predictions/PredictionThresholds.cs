namespace Solia.Dashboard.Predictions;

/// <summary>The probabilities at or above which a prediction is emphasised, shared by every prediction view.</summary>
public static class PredictionThresholds
{
    /// <summary>A pump probability at or above this is emphasised.</summary>
    public const double HighPump = 0.7;

    /// <summary>A rug probability at or above this is emphasised.</summary>
    public const double HighRug = 0.5;

    public static bool IsHighPump(double? pumpProb) => pumpProb >= HighPump;

    public static bool IsHighRug(double? rugProb) => rugProb >= HighRug;
}
