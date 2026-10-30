using Solia.Dashboard.Api;

namespace Solia.Dashboard.Stats;

/// <summary>
/// The <c>/stats/winrate</c> breakdown as donut slices — wins, losses, break-even, in that order — plus the ratio
/// shown in its centre, and how the trades exited (null when the backend didn't say). Pure, like <see cref="EquityCurve"/>.
/// </summary>
public sealed record WinRateSlices(IReadOnlyList<string> Labels, IReadOnlyList<double> Values, double Winrate, ExitBreakdown? Exits = null)
{
    public static readonly IReadOnlyList<string> SliceLabels = ["Wins", "Losses", "Break-even"];

    /// <summary>How many trades closed (the backend's denominator: break-even included).</summary>
    public int Closed => (int)Values.Sum();

    public bool IsEmpty => Closed == 0;

    public static WinRateSlices From(WinRateResponse response) =>
        new(SliceLabels, [response.Wins, response.Losses, response.BreakEven], response.Winrate, response.Exits);
}
