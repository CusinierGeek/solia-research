namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>
/// Maps pumpapi's provider-specific <c>pool</c> names to the canonical venue vocabulary used in
/// the database (docs/DATA-MODEL.md: <c>pumpfun</c> / <c>pumpswap</c> / <c>raydium</c> /
/// <c>meteora</c> / ...), so other data sources can emit the same values for the same venue.
/// Unknown pools pass through lower-cased rather than being dropped.
/// </summary>
public static class PumpApiVenueMap
{
    private static readonly Dictionary<string, string> Canonical = new(StringComparer.OrdinalIgnoreCase)
    {
        ["pump"] = "pumpfun",
        ["pump-amm"] = "pumpswap",
        ["raydium-launchpad"] = "raydium-launchpad",
        ["raydium-cpmm"] = "raydium",
        ["raydium-clmm"] = "raydium",
        ["raydium-v4"] = "raydium",
        ["meteora-launchpad"] = "meteora-launchpad",
        ["meteora-damm-v1"] = "meteora",
        ["meteora-damm-v2"] = "meteora",
        ["meteora-dlmm"] = "meteora",
        ["orca-whirlpool"] = "orca",
        ["bonk"] = "bonk",
        ["sunrise"] = "sunrise",
    };

    /// <summary>Canonical venue for a pumpapi pool name, or null when the pool is absent.</summary>
    public static string? ToCanonical(string? pool)
    {
        if (string.IsNullOrWhiteSpace(pool))
        {
            return null;
        }

        string trimmed = pool.Trim();
        return Canonical.TryGetValue(trimmed, out string? venue) ? venue : trimmed.ToLowerInvariant();
    }
}
