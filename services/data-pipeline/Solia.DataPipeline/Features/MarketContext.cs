using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Market-wide aggregates attached to every snapshot of one tick (FE-T11, issue #195): the regime the
/// token trades in. Computed once per tick by <see cref="MarketContext.Aggregate"/>. Rates are per
/// minute; SOL amounts count SOL-quoted trades only. An aggregate is null until its whole window has
/// been observed (<see cref="CoverageSec"/>), so a fresh process or replay never reports a partial
/// window as a quiet market.
/// </summary>
public sealed record MarketAggregates
{
    /// <summary>Seconds of stream history behind the aggregates, capped at the longest window.</summary>
    public int CoverageSec { get; init; }

    /// <summary><see cref="EventKind.Create"/> events per minute over the short window.</summary>
    public double? LaunchRateShort { get; init; }

    /// <summary><see cref="EventKind.Create"/> events per minute over the long window.</summary>
    public double? LaunchRateLong { get; init; }

    /// <summary><see cref="EventKind.Migrate"/> events (graduations) in the long window.</summary>
    public int? GraduationsLong { get; init; }

    /// <summary>Graduations / launches over the long window; null when nothing launched.</summary>
    public double? GraduationRatioLong { get; init; }

    public double? BuySolShort { get; init; }
    public double? BuySolLong { get; init; }

    /// <summary>Buy minus sell SOL volume across all tokens.</summary>
    public double? NetSolFlowShort { get; init; }

    public double? NetSolFlowLong { get; init; }

    /// <summary>Distinct tokens with a trade in the window.</summary>
    public int? ActiveTokensShort { get; init; }

    public int? ActiveTokensLong { get; init; }
}

/// <summary>
/// Maintains market-wide rolling aggregates over every token of the stream (FE-T11, issue #195),
/// derived for free from the events already ingested. Memory is bounded by time, not by traffic:
/// counters live in one-second buckets of a ring buffer spanning the longest window, plus the last
/// two distinct trade seconds of each token traded within it (for the active-token counts).
/// <para>
/// <b>Point-in-time:</b> <see cref="Aggregate"/> at <c>ts</c> sums the whole seconds strictly before
/// the second containing <c>ts</c>, i.e. <c>[floor(ts) - W, floor(ts))</c>, so nothing at or after the
/// snapshot leaks in, and the result depends only on event timestamps — not on arrival order or on the
/// flush cadence — so live and replay agree. Events are bucketed by their own timestamp against an
/// <see cref="EventClock"/> (same skew rules as the engine): one far-future timestamp is dropped
/// rather than overwriting a bucket of the past; an event older than the longest window is dropped.
/// </para>
/// Not thread-safe: fed and queried from the pipeline's single loop.
/// </summary>
public sealed class MarketContext
{
    public static readonly TimeSpan ShortWindow = TimeSpan.FromMinutes(5);
    public static readonly TimeSpan LongWindow = TimeSpan.FromHours(1);

    private static readonly long ShortSec = (long)ShortWindow.TotalSeconds;
    private static readonly long LongSec = (long)LongWindow.TotalSeconds;
    private static readonly long SkewSec = (long)TokenWindowStore.MaxFutureSkew.TotalSeconds;

    // Accepted events span (clock - long - skew, clock + skew]: a ring larger than that never lets two
    // live seconds share a slot.
    private static readonly int RingSize = (int)(LongSec + (3 * SkewSec) + 1);

    private readonly Bucket[] _ring = new Bucket[RingSize];
    private readonly EventClock _clock = new(TokenWindowStore.MaxFutureSkew, PipelineEngine.ClockJumpConfirmations);
    private readonly Dictionary<string, TradeSeconds> _lastTrades = new(StringComparer.Ordinal);
    private long? _firstSecond;

    public MarketContext()
    {
        for (int i = 0; i < _ring.Length; i++)
        {
            _ring[i].Second = long.MinValue;
        }
    }

    /// <summary>Tokens remembered for the active-token counts (bounded by the long window).</summary>
    public int TrackedTokens => _lastTrades.Count;

    public void Apply(RawEvent e)
    {
        long now = Seconds(_clock.Observe(e.Timestamp));
        long sec = Seconds(e.Timestamp);
        if (sec > now + SkewSec || sec <= now - LongSec - SkewSec)
        {
            return; // implausibly far ahead (unconfirmed jump) or older than any window
        }

        _firstSecond = _firstSecond is { } first && first <= sec ? first : sec;
        switch (e.Kind)
        {
            case EventKind.Create:
                Slot(sec).Launches++;
                break;
            case EventKind.Migrate:
                Slot(sec).Graduations++;
                break;
            case EventKind.Buy or EventKind.Sell:
                RecordTrade(e, sec);
                break;
        }
    }

    /// <summary>
    /// The aggregates for snapshots taken at <paramref name="ts"/>, from events strictly before the
    /// second containing it. Also forgets tokens that last traded before the long window.
    /// </summary>
    public MarketAggregates Aggregate(DateTimeOffset ts)
    {
        long to = Seconds(ts);
        long coverage = _firstSecond is { } first ? Math.Clamp(to - first, 0, LongSec) : 0;

        Sums shortSums = Sum(to - ShortSec, to);
        Sums longSums = Sum(to - LongSec, to);
        int activeShort = 0;
        int activeLong = 0;
        List<string>? stale = null;
        foreach ((string mint, TradeSeconds t) in _lastTrades)
        {
            if (t.Last < to - LongSec - SkewSec)
            {
                (stale ??= []).Add(mint);
                continue;
            }

            if (t.TradedIn(to - ShortSec, to))
            {
                activeShort++;
            }

            if (t.TradedIn(to - LongSec, to))
            {
                activeLong++;
            }
        }

        stale?.ForEach(mint => _lastTrades.Remove(mint));

        bool hasShort = coverage >= ShortSec;
        bool hasLong = coverage >= LongSec;
        return new MarketAggregates
        {
            CoverageSec = (int)coverage,
            LaunchRateShort = hasShort ? shortSums.Launches / ShortWindow.TotalMinutes : null,
            LaunchRateLong = hasLong ? longSums.Launches / LongWindow.TotalMinutes : null,
            GraduationsLong = hasLong ? longSums.Graduations : null,
            GraduationRatioLong = hasLong && longSums.Launches > 0
                ? (double)longSums.Graduations / longSums.Launches
                : null,
            BuySolShort = hasShort ? (double)shortSums.BuySol : null,
            BuySolLong = hasLong ? (double)longSums.BuySol : null,
            NetSolFlowShort = hasShort ? (double)(shortSums.BuySol - shortSums.SellSol) : null,
            NetSolFlowLong = hasLong ? (double)(longSums.BuySol - longSums.SellSol) : null,
            ActiveTokensShort = hasShort ? activeShort : null,
            ActiveTokensLong = hasLong ? activeLong : null,
        };
    }

    private void RecordTrade(RawEvent e, long sec)
    {
        // SOL-quoted only, like the per-token net_sol_flow: other quotes aren't comparable amounts.
        if (e.QuoteMint == RawEvent.WrappedSolMint && e.QuoteAmount is { } sol)
        {
            ref Bucket bucket = ref Slot(sec);
            if (e.Kind == EventKind.Buy)
            {
                bucket.BuySol += sol;
            }
            else
            {
                bucket.SellSol += sol;
            }
        }

        _lastTrades[e.Mint] = _lastTrades.TryGetValue(e.Mint, out TradeSeconds t) ? t.With(sec) : new TradeSeconds(sec, null);
    }

    private ref Bucket Slot(long sec)
    {
        ref Bucket bucket = ref _ring[(int)(((sec % RingSize) + RingSize) % RingSize)];
        if (bucket.Second != sec)
        {
            bucket = new Bucket { Second = sec }; // the slot held a second outside the accepted span
        }

        return ref bucket;
    }

    private Sums Sum(long from, long to)
    {
        var sums = default(Sums);
        for (long sec = from; sec < to; sec++)
        {
            Bucket b = _ring[(int)(((sec % RingSize) + RingSize) % RingSize)];
            if (b.Second == sec)
            {
                sums.Launches += b.Launches;
                sums.Graduations += b.Graduations;
                sums.BuySol += b.BuySol;
                sums.SellSol += b.SellSol;
            }
        }

        return sums;
    }

    private static long Seconds(DateTimeOffset ts) => ts.ToUnixTimeSeconds();

    private struct Bucket
    {
        public long Second;
        public int Launches;
        public int Graduations;
        public decimal BuySol;
        public decimal SellSol;
    }

    private struct Sums
    {
        public int Launches;
        public int Graduations;
        public decimal BuySol;
        public decimal SellSol;
    }

    /// <summary>
    /// A token's two latest distinct trade seconds. Two suffice: a trade in the current (excluded)
    /// second must not hide the one before it, and no accepted trade lies past the current second
    /// by more than the clock skew.
    /// </summary>
    private readonly record struct TradeSeconds(long Last, long? Previous)
    {
        public TradeSeconds With(long sec) =>
            sec > Last ? new(sec, Last)
            : sec == Last ? this
            : Previous is not { } p || sec > p ? new(Last, sec)
            : this;

        public bool TradedIn(long from, long to) =>
            (Last >= from && Last < to) || (Previous is { } p && p >= from && p < to);
    }
}
