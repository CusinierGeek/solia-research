namespace Solia.Core.Data.Entities;

/// <summary>
/// Time-series feature snapshot per token — table <c>features</c>, the model's input unit.
/// Unique on <c>(mint, ts)</c> so live + replay ingestion upsert idempotently.
/// Read-oriented here: written by the data-pipeline, read by the API and the bot.
/// </summary>
public sealed class Feature
{
    public long Id { get; set; }

    /// <summary>FK → <c>tokens_history</c>.</summary>
    public string Mint { get; set; } = null!;

    /// <summary>Snapshot time (UTC).</summary>
    public DateTimeOffset Ts { get; set; }

    public string? Venue { get; set; }

    public bool IsMigrated { get; set; }

    /// <summary>Nullable; AMM pool once migrated.</summary>
    public string? PoolAddress { get; set; }

    public decimal? Price { get; set; }

    public decimal? Volume { get; set; }

    public decimal? MarketCap { get; set; }

    public int? Holders { get; set; }

    public decimal? Liquidity { get; set; }

    public int? BuyEvents { get; set; }

    public int? SellEvents { get; set; }

    public double? BuySellRatio { get; set; }

    public int? TokenAgeSec { get; set; }

    public double? TrendingScore { get; set; }

    public int? SocialMentions { get; set; }

    public double? HoldersGrowthRate { get; set; }

    public double? VolumeGrowthRate { get; set; }

    public double? TxCreationRate { get; set; }

    public double? Volatility { get; set; }

    public double? Ema { get; set; }

    public double? Rsi { get; set; }

    public double? PriceDerivative { get; set; }

    /// <summary>Buy minus sell SOL volume over the window (SOL-quoted trades only).</summary>
    public decimal? NetSolFlow { get; set; }

    public double? BuyVolumeRatio { get; set; }

    public decimal? AvgBuySol { get; set; }

    public decimal? MaxBuySol { get; set; }

    public int? LargeBuyCount { get; set; }

    /// <summary>Simple price return over the pipeline's short momentum window.</summary>
    public double? ReturnShort { get; set; }

    public double? ReturnMedium { get; set; }

    public double? ReturnLong { get; set; }

    public double? PriceAcceleration { get; set; }

    public double? EmaSlope { get; set; }

    public double? VolumeAcceleration { get; set; }

    public double? VolumeBurstRatio { get; set; }

    /// <summary>Seconds of price history behind the momentum returns (capped at the long window).</summary>
    public double? PriceHistorySec { get; set; }

    /// <summary>Distinct trade signers in the pipeline's window (holder-concentration proxy).</summary>
    public int? UniqueTraders { get; set; }

    /// <summary>Largest single-wallet share of the window's signed volume.</summary>
    public double? TopTraderShare { get; set; }

    /// <summary>Herfindahl-Hirschman index of per-wallet volume shares (1 = one wallet).</summary>
    public double? TraderHhi { get; set; }

    /// <summary>Share of the short window's traders not seen earlier in the window.</summary>
    public double? NewTraderRatio { get; set; }

    /// <summary>Pump.fun-specific signals (JSONB).</summary>
    public string? OnchainSignals { get; set; }

    /// <summary>Pump/rug pattern flags (JSONB).</summary>
    public string? BehavioralSignals { get; set; }

    /// <summary>Nullable debug payload (JSONB).</summary>
    public string? Raw { get; set; }

    public TokenHistory? Token { get; set; }
}
