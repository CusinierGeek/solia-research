using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Rolling per-token state. Ingests <see cref="RawEvent"/>s into a time-bounded window and keeps
/// the latest token facts, a bounded unique-trader count (the free holders proxy), a windowed
/// price history for volatility/RSI, an EMA, and the previous snapshot's values so growth
/// rates/derivatives can be computed. <see cref="Snapshot"/> produces one <see cref="FeatureSnapshot"/>.
/// <para>
/// Tolerates out-of-order events: trades are kept sorted by timestamp, and only an event at least
/// as new as the latest one seen updates the "latest" facts (price, market cap, venue, ...).
/// Prices are series in one quote currency: when the quote mint changes (e.g. a migration to a
/// USDC pool) the price series and EMA restart, volume only sums trades in the current quote
/// currency, and growth rates aren't computed across the change. The SOL buy-pressure features
/// only ever count SOL-quoted trades, whatever the current quote currency.
/// </para>
/// <para>
/// Momentum features read a separate per-second price history (the last price of each second, with
/// the EMA as of that price), bounded by the window but keeping the one price still in force at its
/// start: a busy token's look-backs aren't lost to the volatility series' point cap, and the price
/// in force at <c>now - w</c> survives pruning. The EMA and its slope are taken as of the snapshot
/// time, never from prices after it. Volume rates and accelerations only span the time since the
/// current quote currency began, like the volume they're summed from.
/// </para>
/// <para>
/// The holder-concentration proxy (<see cref="TraderConcentration"/>) is rebuilt each snapshot from
/// the window's trades and their signers, so per-wallet volume never outlives the window; whether a
/// recent trader is new or returning looks up its first-seen time on the token.
/// </para>
/// </summary>
internal sealed class TokenState
{
    private const int MaxPriceHistory = 512; // per-window cap (memory bound)

    // QuoteAmount is null when the frame didn't report it: counted as a trade, never as an amount.
    // Trader (txSigner) is null when the source didn't report it.
    private readonly record struct Observation(DateTimeOffset Ts, EventKind Kind, decimal? QuoteAmount, string? QuoteMint, string? Trader);

    private readonly record struct PricePoint(DateTimeOffset Ts, double Price);

    // The last price of a second (ts rounded up) and the EMA right after it.
    private readonly record struct Anchor(DateTimeOffset Ts, double Price, double Ema);

    private readonly int _windowSeconds;
    private readonly int _emaSpan;
    private readonly int _rsiPeriod;
    private readonly decimal _largeBuySol;
    private readonly MomentumWindows _momentum;

    private readonly List<Observation> _window = []; // sorted by Ts
    private readonly Queue<PricePoint> _prices = new(); // in Ts order (late prices are not appended)
    private readonly List<Anchor> _anchors = []; // in Ts order, at most one per second
    private readonly HolderCounter _holders = new();
    private readonly TraderConcentration _concentration;

    private decimal? _price;
    private decimal? _marketCap;
    private decimal? _liquidity;
    private decimal? _supply;
    private string? _venue;
    private string? _poolId;
    private string? _quoteMint;
    private bool _isMigrated;
    private double? _burned;
    private double? _ema;

    private DateTimeOffset _firstSeen;
    private DateTimeOffset _quoteSince = DateTimeOffset.MinValue; // when the current quote currency began
    private DateTimeOffset _createdAt;
    private bool _hasCreate;

    private DateTimeOffset? _prevSnapshotTs;
    private decimal _prevVolume;
    private int _prevHolders;
    private decimal? _prevPrice;
    private double? _prevEma;
    private string? _prevQuoteMint;

    public TokenState(string mint, int windowSeconds, int emaSpan, int rsiPeriod, DateTimeOffset firstSeen,
        decimal largeBuySol = PipelineOptions.DefaultLargeBuySol, MomentumWindows? momentum = null)
    {
        Mint = mint;
        _windowSeconds = windowSeconds;
        _emaSpan = emaSpan;
        _rsiPeriod = rsiPeriod;
        _largeBuySol = largeBuySol;
        _momentum = momentum ?? MomentumWindows.Default;
        _concentration = new TraderConcentration(_holders);
        _firstSeen = firstSeen;
        _createdAt = firstSeen;
        LastActivity = firstSeen;
    }

    public string Mint { get; }

    /// <summary>Timestamp of the newest event applied.</summary>
    public DateTimeOffset LastActivity { get; private set; }

    public void Apply(RawEvent e)
    {
        bool isLatest = e.Timestamp >= LastActivity;
        if (isLatest)
        {
            LastActivity = e.Timestamp;
        }

        if (e.Timestamp < _firstSeen)
        {
            _firstSeen = e.Timestamp;
        }

        if (e.Kind == EventKind.Create)
        {
            _hasCreate = true;
            _createdAt = e.Timestamp;
        }

        if (e.IsMigrated)
        {
            _isMigrated = true;
        }

        if (e.Trader is not null)
        {
            _holders.Add(e.Trader, e.Timestamp);
        }

        if (e.Kind is EventKind.Buy or EventKind.Sell)
        {
            Insert(new Observation(e.Timestamp, e.Kind, e.QuoteAmount, e.QuoteMint, e.Trader));
        }

        // A late event's facts are already stale: keep only what's newest.
        if (!isLatest)
        {
            return;
        }

        if (e.QuoteMint is { } quoteMint)
        {
            if (_quoteMint is not null && quoteMint != _quoteMint)
            {
                RestartPriceSeries();
                _quoteSince = e.Timestamp;
            }

            _quoteMint = quoteMint;
        }

        if (e.Price is { } price)
        {
            _price = price;
            double p = (double)price;
            _prices.Enqueue(new PricePoint(e.Timestamp, p));
            if (_prices.Count > MaxPriceHistory)
            {
                _prices.Dequeue();
            }

            _ema = _ema is null ? p : Indicators.Ema(_ema.Value, p, Indicators.AlphaFromSpan(_emaSpan));

            var anchor = new Anchor(e.Timestamp, p, _ema.Value);
            if (_anchors.Count > 0 && CeilingSecond(_anchors[^1].Ts) == CeilingSecond(e.Timestamp))
            {
                _anchors[^1] = anchor;
            }
            else
            {
                _anchors.Add(anchor);
            }
        }

        _marketCap = e.MarketCap ?? _marketCap;
        _liquidity = e.QuoteInPool ?? _liquidity;
        _supply = e.Supply ?? _supply;
        _venue = e.Venue ?? _venue;
        _poolId = e.PoolId ?? _poolId;
        _burned = e.BurnedLiquidityPercent ?? _burned;
    }

    public bool HasActivityInWindow(DateTimeOffset now)
    {
        Prune(now);
        foreach (Observation o in _window)
        {
            if (o.Ts <= now)
            {
                return true;
            }
        }

        return false;
    }

    public FeatureSnapshot Snapshot(DateTimeOffset now)
    {
        Prune(now);

        int buys = 0;
        int sells = 0;
        decimal volume = 0;
        int solBuys = 0;
        decimal buySol = 0;
        decimal sellSol = 0;
        decimal maxBuySol = 0;
        int largeBuys = 0;
        DateTimeOffset shortStart = now - TimeSpan.FromSeconds(_momentum.ShortSeconds);
        DateTimeOffset prevShortStart = shortStart - TimeSpan.FromSeconds(_momentum.ShortSeconds);
        DateTimeOffset longStart = now - TimeSpan.FromSeconds(_momentum.LongSeconds);
        decimal shortVolume = 0;     // (now - short, now]
        decimal prevShortVolume = 0; // (now - 2 short, now - short]
        decimal longVolume = 0;      // (now - long, now]
        _concentration.Reset(shortStart);
        foreach (Observation o in _window)
        {
            if (o.Ts > now)
            {
                break; // sorted: the rest are after the snapshot time
            }

            if (o.Kind == EventKind.Buy)
            {
                buys++;
            }
            else
            {
                sells++;
            }

            // Only amounts in the current quote currency can be summed.
            bool currentQuote = o.QuoteMint is null || _quoteMint is null || o.QuoteMint == _quoteMint;
            _concentration.Add(o.Trader, o.Ts, currentQuote ? o.QuoteAmount ?? 0 : null);
            if (currentQuote)
            {
                decimal amount = o.QuoteAmount ?? 0;
                volume += amount;
                if (o.Ts > shortStart)
                {
                    shortVolume += amount;
                }
                else if (o.Ts > prevShortStart)
                {
                    prevShortVolume += amount;
                }

                if (o.Ts > longStart)
                {
                    longVolume += amount;
                }
            }

            // SOL-weighted features: a USDC (or unknown-quote) amount isn't SOL, and a trade with
            // no reported amount would drag the averages towards 0.
            if (o.QuoteMint != RawEvent.WrappedSolMint || o.QuoteAmount is not { } sol)
            {
                continue;
            }

            if (o.Kind == EventKind.Buy)
            {
                solBuys++;
                buySol += sol;
                maxBuySol = Math.Max(maxBuySol, sol);
                if (sol >= _largeBuySol)
                {
                    largeBuys++;
                }
            }
            else
            {
                sellSol += sol;
            }
        }

        int trades = buys + sells;
        double buySellRatio = trades == 0 ? 0 : (double)buys / trades;
        int holders = _holders.Count;
        (double? topTraderShare, double? traderHhi) = _concentration.Shares();
        decimal solVolume = buySol + sellSol;

        // Trades are only seen since first sight, so a token younger than the window is rated over its observed span.
        double observedSeconds = Math.Clamp((now - _firstSeen).TotalSeconds, 1, _windowSeconds);
        double txRate = trades / observedSeconds;
        int ageSec = (int)Math.Max(0, (now - (_hasCreate ? _createdAt : _firstSeen)).TotalSeconds);

        double[] windowPrices = _prices.Where(p => p.Ts <= now).Select(p => p.Price).ToArray();
        double volatility = Indicators.Volatility(windowPrices);
        double? rsi = Indicators.Rsi(windowPrices, _rsiPeriod);

        // Momentum: returns over each look-back, and the short return's change versus the short
        // window before it. Null when the price history doesn't reach back that far, which
        // PriceHistorySec tells apart from a flat price.
        int nowIndex = AnchorAt(now);
        double? current = nowIndex < 0 ? null : _anchors[nowIndex].Price;
        double? emaNow = nowIndex < 0 ? null : _anchors[nowIndex].Ema;
        double? shortReturn = Indicators.Return(PriceAt(shortStart), current);
        double? priceAcceleration = shortReturn - Indicators.Return(PriceAt(prevShortStart), PriceAt(shortStart));
        double priceHistorySec = nowIndex < 0
            ? 0
            : Math.Min((now - _anchors[0].Ts).TotalSeconds, _momentum.LongSeconds);

        // Volume: change of the per-second rate between the last two short windows (needs both
        // observed), and the short/long rate ratio (rated over the observed span, like the tx rate).
        // Volume is only summed in the current quote currency, so the span starts at its switch.
        double observedSec = (now - (_quoteSince > _firstSeen ? _quoteSince : _firstSeen)).TotalSeconds;
        double shortSec = _momentum.ShortSeconds;
        double? volumeAcceleration = observedSec < 2 * shortSec
            ? null
            : (double)(shortVolume - prevShortVolume) / shortSec / shortSec;
        double volumeBurst = longVolume == 0
            ? 1 // nothing traded over the long window: no burst either way
            : (double)shortVolume / Math.Clamp(observedSec, 1, shortSec)
              / ((double)longVolume / Math.Clamp(observedSec, 1, _momentum.LongSeconds));

        double? holdersGrowth = null;
        double? volumeGrowth = null;
        double? priceDerivative = null;
        double? emaSlope = null;
        if (_prevSnapshotTs is { } prevTs)
        {
            double dt = (now - prevTs).TotalSeconds;
            holdersGrowth = Indicators.GrowthRate(_prevHolders, holders, dt);
            if (_prevQuoteMint == _quoteMint)
            {
                volumeGrowth = Indicators.GrowthRate((double)_prevVolume, (double)volume, dt);
                if (_price is { } price && _prevPrice is { } previous)
                {
                    priceDerivative = Indicators.GrowthRate((double)previous, (double)price, dt);
                }

                if (emaNow is { } ema && _prevEma is { } prevEma)
                {
                    emaSlope = Indicators.RelativeGrowthRate(prevEma, ema, dt);
                }
            }
        }

        var snapshot = new FeatureSnapshot
        {
            Mint = Mint,
            Ts = now,
            FirstSeenUtc = _firstSeen,
            CreatedAtUtc = _hasCreate ? _createdAt : null,
            Venue = _venue,
            IsMigrated = _isMigrated,
            PoolAddress = _poolId,
            Price = _price,
            Volume = volume,
            MarketCap = _marketCap,
            Liquidity = _liquidity,
            Holders = holders,
            BuyEvents = buys,
            SellEvents = sells,
            BuySellRatio = buySellRatio,
            TokenAgeSec = ageSec,
            TxCreationRate = txRate,
            TrendingScore = volumeGrowth * buySellRatio, // null until volume growth is derivable
            SocialMentions = null,
            HoldersGrowthRate = holdersGrowth,
            VolumeGrowthRate = volumeGrowth,
            Volatility = volatility,
            Ema = emaNow,
            Rsi = rsi,
            PriceDerivative = priceDerivative,
            NetSolFlow = buySol - sellSol,
            BuyVolumeRatio = solVolume == 0 ? null : (double)(buySol / solVolume),
            AvgBuySol = solBuys == 0 ? null : buySol / solBuys,
            MaxBuySol = solBuys == 0 ? null : maxBuySol,
            LargeBuyCount = largeBuys,
            ReturnShort = shortReturn,
            ReturnMedium = Indicators.Return(PriceAt(now - TimeSpan.FromSeconds(_momentum.MediumSeconds)), current),
            ReturnLong = Indicators.Return(PriceAt(longStart), current),
            PriceAcceleration = priceAcceleration,
            EmaSlope = emaSlope,
            VolumeAcceleration = volumeAcceleration,
            VolumeBurstRatio = volumeBurst,
            PriceHistorySec = priceHistorySec,
            UniqueTraders = _concentration.UniqueTraders,
            TopTraderShare = topTraderShare,
            TraderHhi = traderHhi,
            NewTraderRatio = _concentration.NewTraderRatio,
            OnchainSignals = new Dictionary<string, object?>
            {
                ["burnedLiquidityPercent"] = _burned,
                ["poolId"] = _poolId,
                ["supply"] = _supply,
                ["quoteMint"] = _quoteMint,
            },
            BehavioralSignals = new Dictionary<string, object?>
            {
                ["rugRisk"] = _isMigrated && _burned is <= 0,
                ["hasCreate"] = _hasCreate,
            },
        };

        _prevSnapshotTs = now;
        _prevVolume = volume;
        _prevHolders = holders;
        _prevPrice = _price;
        _prevEma = emaNow;
        _prevQuoteMint = _quoteMint;
        return snapshot;
    }

    /// <summary>The price in force at <paramref name="ts"/>: the last one at or before it; null when the history starts later.</summary>
    private double? PriceAt(DateTimeOffset ts) => AnchorAt(ts) is var i and >= 0 ? _anchors[i].Price : null;

    /// <summary>Index of the last anchor at or before <paramref name="ts"/> (binary search); -1 if none.</summary>
    private int AnchorAt(DateTimeOffset ts)
    {
        int lo = 0;
        int hi = _anchors.Count - 1;
        int found = -1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) / 2);
            if (_anchors[mid].Ts <= ts)
            {
                found = mid;
                lo = mid + 1;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return found;
    }

    // Rounded up, so a second's last price is the one in force at that whole second.
    private static long CeilingSecond(DateTimeOffset ts) =>
        (ts.UtcTicks + TimeSpan.TicksPerSecond - 1) / TimeSpan.TicksPerSecond;

    /// <summary>Inserts keeping timestamp order; in-order events (the common case) append in O(1).</summary>
    private void Insert(Observation observation)
    {
        int i = _window.Count;
        while (i > 0 && _window[i - 1].Ts > observation.Ts)
        {
            i--;
        }

        _window.Insert(i, observation);
    }

    /// <summary>Prices in a new quote currency aren't comparable with the old ones.</summary>
    private void RestartPriceSeries()
    {
        _prices.Clear();
        _anchors.Clear();
        _ema = null;
        _price = null;
        _prevPrice = null;
        _prevEma = null;
        _marketCap = null;
        _liquidity = null;
    }

    private void Prune(DateTimeOffset now)
    {
        DateTimeOffset cutoff = now - TimeSpan.FromSeconds(_windowSeconds);

        int stale = 0;
        while (stale < _window.Count && _window[stale].Ts < cutoff)
        {
            stale++;
        }

        _window.RemoveRange(0, stale);

        while (_prices.Count > 0 && _prices.Peek().Ts < cutoff)
        {
            _prices.Dequeue();
        }

        // Keep the anchor in force at the cutoff: a look-back of the full window still has its price.
        int superseded = 0;
        while (superseded + 1 < _anchors.Count && _anchors[superseded + 1].Ts <= cutoff)
        {
            superseded++;
        }

        _anchors.RemoveRange(0, superseded);
    }
}
