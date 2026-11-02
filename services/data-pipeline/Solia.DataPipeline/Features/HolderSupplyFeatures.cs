namespace Solia.DataPipeline.Features;

/// <summary>
/// Holder-side supply concentration of one snapshot (FE-T13, issue #197): who <i>holds</i> the token, approximated
/// from the stream by <see cref="HolderLedger"/> (each wallet's buys − sells ± token transfers on this token), as
/// opposed to the <c>concentration</c> group, which only looks at who <i>traded</i> in the window.
/// Circulating supply = the sum of the tracked positive positions, i.e. the tokens the stream saw leave the pool —
/// the bonding curve / pool account never appears in the ledger (trades credit the signer, transfers to or from a
/// known pool are ignored). Point-in-time: only events dated at or before the snapshot's ts.
/// </summary>
public sealed record HolderSupplyFeatures
{
    /// <summary>Wallets holding a positive position. Always written (0 for a token nobody holds yet).</summary>
    public int Holders { get; init; }

    /// <summary>Share of the circulating supply held by the largest holder; null while nothing circulates.</summary>
    public double? Top1Share { get; init; }

    /// <summary>Share of the circulating supply held by the 10 largest holders; null while nothing circulates.</summary>
    public double? Top10Share { get; init; }

    /// <summary>
    /// Net token flow of the current 10 largest holders over the window (buys + transfers in − sells − transfers
    /// out), as a share of the circulating supply: negative = the top holders are distributing. Null while nothing
    /// circulates.
    /// </summary>
    public double? Top10NetFlow { get; init; }

    /// <summary>Wallets holding more than <c>Pipeline:HolderLargeShare</c> of the token's total supply; null when the supply isn't known.</summary>
    public int? LargeHolders { get; init; }

    /// <summary>
    /// True when the positions can't be complete: the token's create was not seen (it predates the replay start or
    /// the live start, or its state was evicted and recreated), so earlier buys, sells and transfers are missing.
    /// </summary>
    public bool Incomplete { get; init; }
}
