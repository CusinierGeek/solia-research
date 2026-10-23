namespace Solia.Core.Data.Entities;

/// <summary>
/// Hot set of currently tracked tokens (small, frequently updated) — table <c>tokens_live</c>.
/// Read-oriented here: maintained by the data-pipeline, surfaced by the API's live reads.
/// </summary>
public sealed class TokenLive
{
    /// <summary>Token mint address (primary key, FK → <c>tokens_history</c>).</summary>
    public string Mint { get; set; } = null!;

    public string? Symbol { get; set; }

    public string? Name { get; set; }

    public DateTimeOffset? CreatedAt { get; set; }

    public decimal? LastPrice { get; set; }

    public decimal? LastMarketCap { get; set; }

    public decimal? LastLiquidity { get; set; }

    public int? Holders { get; set; }

    public double? TrendingScore { get; set; }

    /// <summary>Current venue: <c>pumpfun</c> / <c>pumpswap</c> / <c>raydium</c> / <c>meteora</c> / ...</summary>
    public string? Venue { get; set; }

    /// <summary>Graduated off the bonding curve.</summary>
    public bool IsMigrated { get; set; }

    /// <summary>Still being tracked. Initialised to the DB default (<c>TRUE</c>).</summary>
    public bool IsActive { get; set; } = true;

    /// <summary>DB default <c>now()</c>.</summary>
    public DateTimeOffset UpdatedAt { get; set; }

    public TokenHistory? Token { get; set; }
}
