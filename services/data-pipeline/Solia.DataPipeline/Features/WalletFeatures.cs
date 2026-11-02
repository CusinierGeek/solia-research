namespace Solia.DataPipeline.Features;

/// <summary>
/// Wallet features of one snapshot (FE-T9, issue #189): who trades the token, from the cross-token
/// <see cref="WalletStore"/> (point-in-time: only events strictly before the snapshot), and how its creator behaves
/// on this launch (from the token's own events, like every per-token feature). Ages are in seconds, capped at
/// <c>Pipeline:WalletAgeCapHours</c>.
/// </summary>
public sealed record WalletFeatures
{
    /// <summary>Seconds of wallet history behind the lookups, capped at the age cap. Always written.</summary>
    public int CoverageSec { get; init; }

    /// <summary>Share of the window's signers first seen less than 1 h ago (unknown = fresh); null without a signer or under 1 h of history.</summary>
    public double? FreshShare1h { get; init; }

    /// <summary>Share of the window's signers first seen less than 24 h ago; null without a signer or under 24 h of history.</summary>
    public double? FreshShare24h { get; init; }

    /// <summary>Median age of the window's signers; null without a signer or before the age cap is covered.</summary>
    public double? MedianTraderAgeSec { get; init; }

    /// <summary>The creator's wallet age; null when the creator is unknown (no create seen) or before the age cap is covered.</summary>
    public int? CreatorAgeSec { get; init; }

    /// <summary>The creator's launches before this one within the age cap; null like <see cref="CreatorAgeSec"/>.</summary>
    public int? CreatorPrevLaunches { get; init; }

    /// <summary>The creator's dev buy in SOL (the create's own buy); 0 without one; null when the creator is unknown.</summary>
    public double? CreatorBuySol { get; init; }

    /// <summary>The dev buy's tokens / supply; 0 without a dev buy; null when the creator or the supply is unknown.</summary>
    public double? CreatorBuySupplyShare { get; init; }

    /// <summary>Tokens the creator sold / tokens it bought on this launch, in [0, 1]; null when it bought none (or is unknown).</summary>
    public double? CreatorSoldShare { get; init; }

    /// <summary>Seconds since the creator's first sell on this launch; null while it hasn't sold (or is unknown).</summary>
    public int? CreatorFirstSellSec { get; init; }
}
