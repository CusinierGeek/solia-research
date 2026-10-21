namespace Solia.DataPipeline.Ingestion;

/// <summary>
/// Classifies a pumpapi <c>pool</c> value as a bonding-curve venue (pre-graduation) or an AMM
/// (post-graduation). A token on an AMM venue is "migrated". Known venues are listed explicitly;
/// unknown venues fall back to a substring heuristic so a new AMM is treated as migrated rather
/// than silently mislabeled as still-bonding.
/// </summary>
public static class VenueClassifier
{
    private static readonly HashSet<string> BondingVenues = new(StringComparer.OrdinalIgnoreCase)
    {
        "pump", "raydium-launchpad", "meteora-launchpad", "bonk", "sunrise",
    };

    private static readonly HashSet<string> AmmVenues = new(StringComparer.OrdinalIgnoreCase)
    {
        "pump-amm", "raydium-cpmm", "raydium-clmm", "raydium-v4",
        "meteora-damm-v1", "meteora-damm-v2", "meteora-dlmm", "orca-whirlpool",
    };

    private static readonly string[] AmmHints =
        ["amm", "cpmm", "clmm", "damm", "dlmm", "whirlpool"];

    /// <summary>True if the venue is a post-bonding AMM pool.</summary>
    public static bool IsAmm(string? venue)
    {
        if (string.IsNullOrWhiteSpace(venue))
        {
            return false;
        }

        if (AmmVenues.Contains(venue))
        {
            return true;
        }

        if (BondingVenues.Contains(venue))
        {
            return false;
        }

        foreach (string hint in AmmHints)
        {
            if (venue.Contains(hint, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}
