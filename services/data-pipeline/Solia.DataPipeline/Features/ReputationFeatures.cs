namespace Solia.DataPipeline.Features;

/// <summary>
/// Wallet reputation features of one snapshot (FE-T10, issue #190): whether the wallets behind a token are
/// <i>good</i>, from outcomes in the cross-token <see cref="WalletStore"/> (point-in-time: only events strictly before
/// the snapshot, launch outcomes only once their horizon + label embargo is over), and whether its launch was bundled
/// (from the token's own buys). The history-dependent ones are null until the age cap is covered (replay warm-up).
/// </summary>
public sealed record ReputationFeatures
{
    /// <summary>The creator's earlier launches (within the age cap) whose outcome is recorded; 0 for an unknown creator; null without a creator or in warm-up.</summary>
    public int? CreatorSettledLaunches { get; init; }

    /// <summary>Share of those launches that rugged; null without one.</summary>
    public double? CreatorRugShare { get; init; }

    /// <summary>Share of those launches that pumped; null without one.</summary>
    public double? CreatorPumpShare { get; init; }

    /// <summary>Distinct first buyers seen (the creator excepted), at most <c>Pipeline:ReputationEarlyBuyers</c>. Always written.</summary>
    public int EarlyBuyers { get; init; }

    /// <summary>Share of the early buyers with enough round trips to be rated; null without one or in warm-up.</summary>
    public double? EarlyKnownShare { get; init; }

    /// <summary>Mean win rate of the rated early buyers; null when none is rated.</summary>
    public double? EarlyWinRate { get; init; }

    /// <summary>Mean realized P&amp;L per round trip (SOL) of the rated early buyers; null when none is rated.</summary>
    public double? EarlyPnlSol { get; init; }

    /// <summary>Share of the window's SOL buy volume from smart wallets; null without a SOL buy or in warm-up.</summary>
    public double? SmartBuyShare { get; init; }

    /// <summary>Net SOL flow (buys − sells) of smart wallets over the window; null in warm-up.</summary>
    public double? SmartNetFlowSol { get; init; }

    /// <summary>Distinct smart wallets trading in the window; null in warm-up.</summary>
    public int? SmartTraders { get; init; }

    /// <summary>Share of the first buys (with a block) landing in the creation block; null without a creation block or such a buy.</summary>
    public double? BundleCreateBlockShare { get; init; }

    /// <summary>Largest share of the first buys (with a block) landing in one single block; null without such a buy.</summary>
    public double? BundleTopBlockShare { get; init; }
}
