namespace Solia.DataPipeline.Features;

/// <summary>
/// One computed feature row for a token at a point in time — the model's input unit and what
/// T7 persists to the <c>features</c> table (see docs/DATA-MODEL.md). Produced by
/// <see cref="TokenState.Snapshot"/>. Nullable fields are left null when not yet derivable
/// (e.g. growth rates before a second snapshot, RSI before enough history).
/// </summary>
public sealed record FeatureSnapshot
{
    public required string Mint { get; init; }
    public required DateTimeOffset Ts { get; init; }

    /// <summary>When the pipeline first saw the token (earliest event timestamp).</summary>
    public DateTimeOffset FirstSeenUtc { get; init; }

    /// <summary>The token's on-chain creation time, when its create event was seen.</summary>
    public DateTimeOffset? CreatedAtUtc { get; init; }

    public string? Venue { get; init; }
    public bool IsMigrated { get; init; }
    public string? PoolAddress { get; init; }

    public decimal? Price { get; init; }
    public decimal? Volume { get; init; }
    public decimal? MarketCap { get; init; }
    public decimal? Liquidity { get; init; }

    public int Holders { get; init; }
    public int BuyEvents { get; init; }
    public int SellEvents { get; init; }
    public double BuySellRatio { get; init; }

    public int TokenAgeSec { get; init; }
    public double TxCreationRate { get; init; }

    public double? TrendingScore { get; init; }
    public int? SocialMentions { get; init; }

    public double? HoldersGrowthRate { get; init; }
    public double? VolumeGrowthRate { get; init; }
    public double? Volatility { get; init; }
    public double? Ema { get; init; }
    public double? Rsi { get; init; }
    public double? PriceDerivative { get; init; }

    // SOL-weighted buy pressure over the window: only SOL-quoted trades count, so a whale buy and
    // ten dust buys no longer look alike. Amounts in SOL; null where undefined (no SOL trade/buy).

    /// <summary>Buy minus sell SOL volume (negative under net selling).</summary>
    public decimal? NetSolFlow { get; init; }

    /// <summary>Buy SOL volume / total SOL volume, in [0, 1].</summary>
    public double? BuyVolumeRatio { get; init; }

    public decimal? AvgBuySol { get; init; }
    public decimal? MaxBuySol { get; init; }

    /// <summary>Buys of at least <c>PipelineOptions.LargeBuySol</c>.</summary>
    public int LargeBuyCount { get; init; }

    // Momentum & acceleration over the configured MomentumWindows (short/medium/long). Returns are
    // simple (p_now / p_then - 1); null where the price series doesn't reach back far enough.

    public double? ReturnShort { get; init; }
    public double? ReturnMedium { get; init; }
    public double? ReturnLong { get; init; }

    /// <summary>Short return minus the short return one short window earlier (needs 2 short windows of prices).</summary>
    public double? PriceAcceleration { get; init; }

    /// <summary>Per-second EMA change as a fraction of the previous snapshot's EMA; null on the first snapshot.</summary>
    public double? EmaSlope { get; init; }

    /// <summary>
    /// Change of the per-second volume rate between the last short window and the one before it,
    /// per second (quote/s²); null until two short windows have been observed.
    /// </summary>
    public double? VolumeAcceleration { get; init; }

    /// <summary>Short-window volume rate / long-window volume rate (&gt;1 = burst); 1 when nothing traded over the long window.</summary>
    public double VolumeBurstRatio { get; init; } = 1;

    /// <summary>
    /// How far back (seconds, capped at the long window) the momentum price history reaches: a
    /// return over a longer look-back is null. Tells "no history yet" apart from a flat price.
    /// </summary>
    public double PriceHistorySec { get; init; }

    // Holder-concentration proxy over the window, from trade signers (see TraderConcentration).
    // Shares are of the signed volume in the current quote currency; unsigned trades are skipped.

    /// <summary>Distinct trade signers in the window.</summary>
    public int UniqueTraders { get; init; }

    /// <summary>Largest single-wallet share of the window's volume, in (0, 1]; null without signed volume.</summary>
    public double? TopTraderShare { get; init; }

    /// <summary>HHI of per-wallet volume shares, in (0, 1] (1 = one wallet); null without signed volume.</summary>
    public double? TraderHhi { get; init; }

    /// <summary>
    /// Share of the traders of the last short momentum window not seen earlier in the window, in
    /// [0, 1]; null without a signed trade in that short window.
    /// </summary>
    public double? NewTraderRatio { get; init; }

    /// <summary>
    /// Percent [0, 100] of the current pool's LP tokens burned, from the latest event of that pool that
    /// reported it; null when unknown (none seen yet, e.g. still on the bonding curve, malformed, or
    /// the token moved to a pool that hasn't reported it).
    /// </summary>
    public double? BurnedLiquidity { get; init; }

    /// <summary>
    /// Market-wide context of the tick (FE-T11): the same instance for every snapshot of a flush.
    /// Null only for snapshots taken outside <see cref="TokenWindowStore.SnapshotActive"/>.
    /// </summary>
    public MarketAggregates? Market { get; init; }

    /// <summary>
    /// Wallet features (FE-T9): trader freshness and creator behaviour. Null only for snapshots taken without a
    /// <see cref="WalletStore"/> (outside <see cref="TokenWindowStore.SnapshotActive"/>).
    /// </summary>
    public WalletFeatures? Wallet { get; init; }

    /// <summary>
    /// Wallet reputation features (FE-T10): creator track record, early-buyer quality, smart-money flow, launch
    /// bundling. Null only for snapshots taken without a <see cref="WalletStore"/>, like <see cref="Wallet"/>.
    /// </summary>
    public ReputationFeatures? Reputation { get; init; }

    /// <summary>
    /// Token metadata features (FE-T12): metadata host kind, social links known by ts, narrative heat. Null only for
    /// snapshots taken outside <see cref="TokenWindowStore.SnapshotActive"/>.
    /// </summary>
    public MetadataFeatures? Metadata { get; init; }

    /// <summary>Holder-side supply concentration (FE-T13) from the token's stream-derived holder ledger. Always set by <see cref="TokenState.Snapshot"/>.</summary>
    public HolderSupplyFeatures? HolderSupply { get; init; }

    public IReadOnlyDictionary<string, object?> OnchainSignals { get; init; } =
        new Dictionary<string, object?>();

    public IReadOnlyDictionary<string, object?> BehavioralSignals { get; init; } =
        new Dictionary<string, object?>();
}
