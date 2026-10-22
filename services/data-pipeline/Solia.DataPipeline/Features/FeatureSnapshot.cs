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

    public IReadOnlyDictionary<string, object?> OnchainSignals { get; init; } =
        new Dictionary<string, object?>();

    public IReadOnlyDictionary<string, object?> BehavioralSignals { get; init; } =
        new Dictionary<string, object?>();
}
