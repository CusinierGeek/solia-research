using System.Globalization;
using System.Text.Json;

namespace Solia.DataPipeline.Ingestion.PumpApi;

/// <summary>
/// Maps a single pumpapi stream/replay message (JSON) to a <see cref="RawEvent"/>. The pumpapi
/// server pushes <em>all</em> events; this parser normalizes the ones we can attribute to a
/// token. It is <strong>resilient by design</strong>: malformed, partial, or unattributable
/// messages yield <c>null</c> (never an exception) so a single bad frame cannot kill the stream.
/// Schema reference: docs/DATA-SOURCES.md.
/// </summary>
public static class PumpApiEventParser
{
    private const string WrappedSolMint = RawEvent.WrappedSolMint;

    // pumpapi's native-SOL marker in transfers[] (not a real mint, flagged isSolana).
    private const string NativeSolMint = "So11111111111111111111111111111111111111111";

    private static readonly string[] SolQuotedFields =
        ["solAmount", "solInPool", "vSolInBondingCurve", "marketCapSol"];

    // Accepted event-time window. Anything outside it is a unit/format error, not a real event.
    private static readonly DateTimeOffset MinTimestamp = new(2020, 1, 1, 0, 0, 0, TimeSpan.Zero);
    private static readonly DateTimeOffset MaxTimestamp = new(2100, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static readonly Dictionary<string, EventKind> Kinds = new(StringComparer.OrdinalIgnoreCase)
    {
        ["create"] = EventKind.Create,
        ["buy"] = EventKind.Buy,
        ["sell"] = EventKind.Sell,
        ["migrate"] = EventKind.Migrate,
        ["createPool"] = EventKind.CreatePool,
        ["add"] = EventKind.AddLiquidity,
        ["remove"] = EventKind.RemoveLiquidity,
        ["transfer"] = EventKind.Transfer,
    };

    public static RawEvent? Parse(string? json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            return null;
        }

        try
        {
            using JsonDocument doc = JsonDocument.Parse(json);
            return Parse(doc.RootElement);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Parses a UTF-8 frame straight from the receive buffer, without building a string first.</summary>
    public static RawEvent? ParseUtf8(ReadOnlyMemory<byte> utf8Json)
    {
        if (utf8Json.IsEmpty)
        {
            return null;
        }

        try
        {
            using JsonDocument doc = JsonDocument.Parse(utf8Json);
            return Parse(doc.RootElement);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    public static RawEvent? Parse(JsonElement root)
    {
        try
        {
            return ParseElement(root);
        }
        catch (InvalidOperationException)
        {
            // Includes ObjectDisposedException: an element whose JsonDocument was already disposed.
            return null;
        }
    }

    private static RawEvent? ParseElement(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Object)
        {
            return null;
        }

        // Real stream/archive frames carry the type in "txType"; "action" is accepted as a fallback.
        string? action = GetString(root, "txType") ?? GetString(root, "action");
        if (action is null)
        {
            return null;
        }

        // A token-less event (e.g. createConfig) can't be attributed — skip it.
        string? mint = GetString(root, "mint");
        if (string.IsNullOrWhiteSpace(mint))
        {
            mint = ExtractTransferMint(root);
        }

        if (string.IsNullOrWhiteSpace(mint))
        {
            return null;
        }

        // Without a trustworthy event time the frame is partial: wall-clock time would corrupt replays.
        if (GetTimestamp(root) is not { } timestamp)
        {
            return null;
        }

        EventKind kind = Kinds.GetValueOrDefault(action, EventKind.Other);
        string? pool = GetString(root, "pool");

        // SOL-quoted pools (the vast majority) report sol* fields; pools quoted in another token
        // (USDC, ...) report quote* fields plus quoteMint. Either way the values are in QuoteMint.
        decimal? tokensInPool = GetDecimal(root, "tokensInPool");
        decimal? quoteInPool = GetDecimal(root, "quoteInPool") ?? GetDecimal(root, "solInPool");
        decimal? virtualTokens = GetDecimal(root, "vTokensInBondingCurve");
        decimal? virtualQuote = GetDecimal(root, "vQuoteInBondingCurve") ?? GetDecimal(root, "vSolInBondingCurve");
        string? quoteMint = GetString(root, "quoteMint");
        if (quoteMint is null && kind != EventKind.Transfer && HasAny(root, SolQuotedFields))
        {
            quoteMint = WrappedSolMint;
        }

        return new RawEvent
        {
            Mint = mint,
            Kind = kind,
            Timestamp = timestamp,
            Venue = PumpApiVenueMap.ToCanonical(pool),
            IsMigrated = kind == EventKind.Migrate || VenueClassifier.IsAmm(pool),
            PoolId = GetString(root, "poolId"),
            Signature = GetString(root, "signature"),
            Trader = GetString(root, "txSigner"),
            Price = GetDecimal(root, "price") ?? DerivePrice(quoteInPool, tokensInPool, virtualQuote, virtualTokens),
            QuoteMint = quoteMint,
            MarketCap = GetDecimal(root, "marketCapQuote") ?? GetDecimal(root, "marketCapSol"),
            TokenAmount = GetDecimal(root, "tokenAmount"),
            QuoteAmount = GetDecimal(root, "quoteAmount") ?? GetDecimal(root, "solAmount"),
            TokensInPool = tokensInPool,
            QuoteInPool = quoteInPool,
            VirtualTokenReserves = virtualTokens,
            VirtualQuoteReserves = virtualQuote,
            Name = GetString(root, "name"),
            Symbol = GetString(root, "symbol"),
            Supply = GetDecimal(root, "supply"),
            BurnedLiquidityPercent = ParsePercent(GetString(root, "burnedLiquidity")),
            Block = GetLong(root, "block"),
        };
    }

    /// <summary>
    /// Transfer events carry the token under <c>transfers[].mint</c> rather than at the root.
    /// SOL/wSOL legs are skipped; the first token leg identifies the event (one mint per event).
    /// </summary>
    private static string? ExtractTransferMint(JsonElement root)
    {
        if (!root.TryGetProperty("transfers", out JsonElement transfers) ||
            transfers.ValueKind != JsonValueKind.Array)
        {
            return null;
        }

        foreach (JsonElement transfer in transfers.EnumerateArray())
        {
            if (transfer.ValueKind != JsonValueKind.Object ||
                (transfer.TryGetProperty("isSolana", out JsonElement isSolana) && isSolana.ValueKind == JsonValueKind.True))
            {
                continue;
            }

            string? mint = GetString(transfer, "mint");
            if (!string.IsNullOrWhiteSpace(mint) && mint != WrappedSolMint && mint != NativeSolMint)
            {
                return mint;
            }
        }

        return null;
    }

    /// <summary>
    /// Price from reserves when the explicit field is absent. A bonding curve prices off its
    /// <em>virtual</em> reserves, so those win when present; real reserves are the AMM fallback.
    /// </summary>
    private static decimal? DerivePrice(decimal? quoteInPool, decimal? tokensInPool, decimal? virtualQuote, decimal? virtualTokens) =>
        Divide(virtualQuote, virtualTokens) ?? Divide(quoteInPool, tokensInPool);

    /// <summary>quote / tokens, or null when either is missing, tokens isn't positive, or the result overflows decimal.</summary>
    private static decimal? Divide(decimal? quote, decimal? tokens)
    {
        if (quote is not { } q || tokens is not { } t || t <= 0)
        {
            return null;
        }

        try
        {
            return q / t;
        }
        catch (OverflowException)
        {
            return null;
        }
    }

    /// <summary>
    /// Event time from <c>timestamp</c> (number or numeric string). pumpapi sends Unix
    /// milliseconds, but the unit is inferred from magnitude so a seconds/micro/nanosecond value
    /// can't silently land in 1970 or the far future. Null when absent or implausible.
    /// </summary>
    private static DateTimeOffset? GetTimestamp(JsonElement root)
    {
        long? raw = GetLong(root, "timestamp");
        if (raw is null &&
            long.TryParse(GetString(root, "timestamp"), NumberStyles.Integer, CultureInfo.InvariantCulture, out long parsed))
        {
            raw = parsed;
        }

        if (raw is not { } value || value <= 0)
        {
            return null;
        }

        long ms = value switch
        {
            < 100_000_000_000L => value * 1_000,         // seconds
            < 100_000_000_000_000L => value,             // milliseconds
            < 100_000_000_000_000_000L => value / 1_000, // microseconds
            _ => value / 1_000_000,                      // nanoseconds
        };

        DateTimeOffset ts = DateTimeOffset.FromUnixTimeMilliseconds(ms);
        return ts >= MinTimestamp && ts < MaxTimestamp ? ts : null;
    }

    private static bool HasAny(JsonElement root, string[] names)
    {
        foreach (string name in names)
        {
            if (root.TryGetProperty(name, out JsonElement el) && el.ValueKind == JsonValueKind.Number)
            {
                return true;
            }
        }

        return false;
    }

    private static string? GetString(JsonElement root, string name) =>
        root.TryGetProperty(name, out JsonElement el) && el.ValueKind == JsonValueKind.String
            ? el.GetString()
            : null;

    private static decimal? GetDecimal(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out JsonElement el) || el.ValueKind != JsonValueKind.Number)
        {
            return null;
        }

        if (el.TryGetDecimal(out decimal dec))
        {
            return dec;
        }

        // Scientific notation / magnitudes outside decimal range come back via double.
        if (el.TryGetDouble(out double dbl) && !double.IsNaN(dbl) && !double.IsInfinity(dbl))
        {
            try
            {
                return (decimal)dbl;
            }
            catch (OverflowException)
            {
                return null;
            }
        }

        return null;
    }

    private static long? GetLong(JsonElement root, string name)
    {
        if (!root.TryGetProperty(name, out JsonElement el) || el.ValueKind != JsonValueKind.Number)
        {
            return null;
        }

        if (el.TryGetInt64(out long v))
        {
            return v;
        }

        return el.TryGetDouble(out double d) && !double.IsNaN(d) && !double.IsInfinity(d) ? (long)d : null;
    }

    /// <summary>Percent in [0, 100]; NaN, infinities and out-of-range values are unknown (null).</summary>
    private static double? ParsePercent(string? raw)
    {
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        string trimmed = raw.Trim().TrimEnd('%').Trim();
        return double.TryParse(trimmed, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) &&
               double.IsFinite(v) && v is >= 0 and <= 100
            ? v
            : null;
    }
}
