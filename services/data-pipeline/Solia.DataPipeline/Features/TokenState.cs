using Solia.DataPipeline.Ingestion;

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
/// currency, and growth rates aren't computed across the change.
/// </para>
/// </summary>
internal sealed class TokenState
{
    private const int MaxPriceHistory = 512; // per-window cap (memory bound)

    private readonly record struct Observation(DateTimeOffset Ts, EventKind Kind, decimal QuoteAmount, string? QuoteMint);

    private readonly record struct PricePoint(DateTimeOffset Ts, double Price);

    private readonly int _windowSeconds;
    private readonly int _emaSpan;
    private readonly int _rsiPeriod;

    private readonly List<Observation> _window = []; // sorted by Ts
    private readonly Queue<PricePoint> _prices = new(); // in Ts order (late prices are not appended)
    private readonly HolderCounter _holders = new();

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
    private DateTimeOffset _createdAt;
    private bool _hasCreate;

    private DateTimeOffset? _prevSnapshotTs;
    private decimal _prevVolume;
    private int _prevHolders;
    private decimal? _prevPrice;
    private string? _prevQuoteMint;

    public TokenState(string mint, int windowSeconds, int emaSpan, int rsiPeriod, DateTimeOffset firstSeen)
    {
        Mint = mint;
        _windowSeconds = windowSeconds;
        _emaSpan = emaSpan;
        _rsiPeriod = rsiPeriod;
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
            _holders.Add(e.Trader);
        }

        if (e.Kind is EventKind.Buy or EventKind.Sell)
        {
            Insert(new Observation(e.Timestamp, e.Kind, e.QuoteAmount ?? 0, e.QuoteMint));
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
            if (o.QuoteMint is null || _quoteMint is null || o.QuoteMint == _quoteMint)
            {
                volume += o.QuoteAmount;
            }
        }

        int trades = buys + sells;
        double buySellRatio = trades == 0 ? 0 : (double)buys / trades;
        int holders = _holders.Count;

        // Trades are only seen since first sight, so a token younger than the window is rated over its observed span.
        double observedSeconds = Math.Clamp((now - _firstSeen).TotalSeconds, 1, _windowSeconds);
        double txRate = trades / observedSeconds;
        int ageSec = (int)Math.Max(0, (now - (_hasCreate ? _createdAt : _firstSeen)).TotalSeconds);

        double[] windowPrices = _prices.Where(p => p.Ts <= now).Select(p => p.Price).ToArray();
        double volatility = Indicators.Volatility(windowPrices);
        double? rsi = Indicators.Rsi(windowPrices, _rsiPeriod);

        double? holdersGrowth = null;
        double? volumeGrowth = null;
        double? priceDerivative = null;
        if (_prevSnapshotTs is { } prevTs)
        {
            double dt = (now - prevTs).TotalSeconds;
            holdersGrowth = Indicators.GrowthRate(_prevHolders, holders, dt);
            if (_prevQuoteMint == _quoteMint)
            {
                volumeGrowth = Indicators.GrowthRate((double)_prevVolume, (double)volume, dt);
                if (_price is { } current && _prevPrice is { } previous)
                {
                    priceDerivative = Indicators.GrowthRate((double)previous, (double)current, dt);
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
            Ema = _ema,
            Rsi = rsi,
            PriceDerivative = priceDerivative,
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
        _prevQuoteMint = _quoteMint;
        return snapshot;
    }

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
        _ema = null;
        _price = null;
        _prevPrice = null;
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
    }
}
