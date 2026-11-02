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

    /// <summary>Percent [0, 100] of the AMM pool's LP burned; null when unknown (no pool creation seen).</summary>
    public double? BurnedLiquidity { get; set; }

    // Market-regime context (FE-T11, #195): market-wide aggregates, the same for every row of a tick.

    /// <summary>Seconds of stream history behind the market aggregates, capped at 3600 (FE-T11).</summary>
    public int? MktCoverageSec { get; set; }

    /// <summary>Market-wide Create events per minute, last 5 min.</summary>
    public double? MktLaunchRate5m { get; set; }

    /// <summary>Market-wide Create events per minute, last hour.</summary>
    public double? MktLaunchRate1h { get; set; }

    /// <summary>Market-wide graduations (Migrate events), last hour.</summary>
    public int? MktGraduations1h { get; set; }

    /// <summary>Graduations / launches, last hour; null without launches.</summary>
    public double? MktGraduationRatio1h { get; set; }

    /// <summary>Market-wide SOL buy volume, last 5 min.</summary>
    public double? MktBuySol5m { get; set; }

    /// <summary>Market-wide SOL buy volume, last hour.</summary>
    public double? MktBuySol1h { get; set; }

    /// <summary>Market-wide SOL buys minus sells, last 5 min.</summary>
    public double? MktNetSolFlow5m { get; set; }

    /// <summary>Market-wide SOL buys minus sells, last hour.</summary>
    public double? MktNetSolFlow1h { get; set; }

    /// <summary>Distinct tokens with a trade, last 5 min.</summary>
    public int? MktActiveTokens5m { get; set; }

    /// <summary>Distinct tokens with a trade, last hour.</summary>
    public int? MktActiveTokens1h { get; set; }

    // Wallet features (FE-T9, #189): trader freshness and creator behaviour, from the pipeline's wallet memory.

    /// <summary>Seconds of wallet history behind the wallet features, capped at the age cap (FE-T9).</summary>
    public int? WalCoverageSec { get; set; }

    /// <summary>Share of the window's signers first seen less than 1 h ago.</summary>
    public double? WalFresh1hShare { get; set; }

    /// <summary>Share of the window's signers first seen less than 24 h ago.</summary>
    public double? WalFresh24hShare { get; set; }

    /// <summary>Median wallet age of the window's signers, capped.</summary>
    public double? WalMedianTraderAgeSec { get; set; }

    /// <summary>Creator wallet age, capped; null when the creator is unknown or during warm-up.</summary>
    public int? WalCreatorAgeSec { get; set; }

    /// <summary>Creator's launches before this one within the age cap.</summary>
    public int? WalCreatorPrevLaunches { get; set; }

    /// <summary>Creator's dev buy, SOL.</summary>
    public double? WalCreatorBuySol { get; set; }

    /// <summary>Creator's dev buy tokens / supply.</summary>
    public double? WalCreatorBuySupplyShare { get; set; }

    /// <summary>Tokens the creator sold / bought on this launch.</summary>
    public double? WalCreatorSoldShare { get; set; }

    /// <summary>Seconds since the creator's first sell; null while it hasn't sold.</summary>
    public int? WalCreatorFirstSellSec { get; set; }

    /// <summary>Pump.fun-specific signals (JSONB).</summary>
    public string? OnchainSignals { get; set; }

    /// <summary>Pump/rug pattern flags (JSONB).</summary>
    public string? BehavioralSignals { get; set; }

    /// <summary>Nullable debug payload (JSONB).</summary>
    public string? Raw { get; set; }

    public TokenHistory? Token { get; set; }
}
