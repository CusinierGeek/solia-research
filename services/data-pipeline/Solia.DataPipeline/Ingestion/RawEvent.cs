namespace Solia.DataPipeline.Ingestion;

/// <summary>
/// The canonical, provider-agnostic shape of a single on-chain event. Every data source
/// (pumpapi today; PumpPortal / Solana RPC later) normalizes into this so nothing downstream
/// knows which provider produced it. Money/price values are <see cref="decimal"/> to avoid
/// float drift; fields absent for a given event kind are left null.
/// </summary>
public sealed record RawEvent
{
    /// <summary>Token mint address — the token identity.</summary>
    public required string Mint { get; init; }

    public required EventKind Kind { get; init; }

    /// <summary>Event time in UTC, taken from the source's timestamp (never wall-clock time).</summary>
    public required DateTimeOffset Timestamp { get; init; }

    /// <summary>
    /// Canonical trading venue (docs/DATA-MODEL.md vocabulary), e.g. "pumpfun", "pumpswap",
    /// "raydium". Null when the source didn't say.
    /// </summary>
    public string? Venue { get; init; }

    /// <summary>True when the token has graduated: the event is on a post-bonding AMM, or is the migration itself.</summary>
    public bool IsMigrated { get; init; }

    public string? PoolId { get; init; }
    public string? Signature { get; init; }

    /// <summary>Trader wallet (pumpapi <c>txSigner</c>).</summary>
    public string? Trader { get; init; }

    /// <summary>For <see cref="EventKind.Buy"/>/<see cref="EventKind.Sell"/>: true = buy; null for non-trades.</summary>
    public bool? IsBuy => Kind switch
    {
        EventKind.Buy => true,
        EventKind.Sell => false,
        _ => null,
    };

    /// <summary>Price in quote currency per token.</summary>
    public decimal? Price { get; init; }

    /// <summary>Market cap in quote currency (pumpapi <c>marketCapQuote</c>).</summary>
    public decimal? MarketCap { get; init; }

    public decimal? TokenAmount { get; init; }
    public decimal? QuoteAmount { get; init; }

    /// <summary>Real pool reserves (pumpapi <c>tokensInPool</c>/<c>quoteInPool</c>).</summary>
    public decimal? TokensInPool { get; init; }
    public decimal? QuoteInPool { get; init; }

    /// <summary>Virtual bonding-curve reserves (pumpapi <c>vTokensInBondingCurve</c>/<c>vQuoteInBondingCurve</c>).</summary>
    public decimal? VirtualTokenReserves { get; init; }
    public decimal? VirtualQuoteReserves { get; init; }

    public string? Name { get; init; }
    public string? Symbol { get; init; }
    public decimal? Supply { get; init; }

    /// <summary>Burned-liquidity percentage (pumpapi <c>burnedLiquidity</c> "100%" → 100). Rug-risk signal.</summary>
    public double? BurnedLiquidityPercent { get; init; }

    public long? Block { get; init; }
}
