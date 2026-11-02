using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Rolling per-token state. Ingests <see cref="RawEvent"/>s into a time-bounded window and keeps
/// the latest token facts, a bounded unique-trader count (the free holders proxy), a windowed
/// price history for volatility/RSI, an EMA, and the previous snapshot's values so growth
/// rates/derivatives can be computed. <see cref="Snapshot"/> produces one <see cref="FeatureSnapshot"/>.
/// <para>
/// Causal: <see cref="Apply"/> only queues an event; a snapshot at <c>now</c> first applies the
/// queued events dated at or before <c>now</c>, in timestamp order, so no field — not even the
/// "latest" facts — sees an event dated after the snapshot. Later ones stay queued until their time
/// is reached (each applied exactly once).
/// Tolerates out-of-order events: trades are kept sorted by timestamp, and only an event at least
/// as new as the latest one applied updates the "latest" facts (price, market cap, venue, ...).
/// Prices are a SOL series: a migrated token often trades on several pools at once, each quoted in
/// its own currency (SOL, USDC, another token...), and taking every trade's price would jump by
/// orders of magnitude between pools. Only SOL-quoted events (and events reporting no quote mint)
/// set the price, EMA, market cap, liquidity, venue and pool, and only their amounts are summed in
/// volume and the concentration shares; a trade in another currency still counts as a buy/sell and
/// a holder. A token never traded against SOL has no price. The SOL buy-pressure features only
/// count SOL-quoted trades.
/// </para>
/// <para>
/// Momentum features read a separate per-second price history (the last price of each second, with
/// the EMA as of that price), bounded by the window but keeping the one price still in force at its
/// start: a busy token's look-backs aren't lost to the volatility series' point cap, and the price
/// in force at <c>now - w</c> survives pruning. The EMA and its slope are taken as of the snapshot
/// time, never from prices after it.
/// </para>
/// <para>
/// The holder-concentration proxy (<see cref="TraderConcentration"/>) is rebuilt each snapshot from
/// the window's trades and their signers, so per-wallet volume never outlives the window; whether a
/// recent trader is new or returning looks up its first-seen time on the token.
/// </para>
/// <para>
/// Wallet features (FE-T9): the window's signers are looked up in the cross-token <see cref="WalletStore"/>
/// handed to <see cref="Snapshot"/>; the creator (signer of the create) and its dev buy, later buys and sells on
/// this launch are kept for the token's lifetime.
/// </para>
/// <para>
/// Wallet reputation (FE-T10): the token's first distinct buyers and first buys (with their block) are kept for its
/// lifetime; the buyers' round-trip record, the creator's launch outcomes and the window traders' "smart" status are
/// looked up in the same <see cref="WalletStore"/>.
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

    private readonly List<RawEvent> _pending = []; // received, not yet applied; sorted by Timestamp
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

    private DateTimeOffset? _latestApplied;
    private DateTimeOffset _firstSeen;
    private DateTimeOffset _createdAt;
    private bool _hasCreate;

    // The creator (signer of the create) and its trades on this launch (FE-T9).
    private string? _creator;
    private decimal? _creatorBuySol;     // the create's dev buy, in SOL
    private decimal? _creatorBuyTokens;  // the create's dev buy, in tokens
    private decimal? _createSupply;
    private decimal _creatorBoughtTokens;
    private decimal _creatorSoldTokens;
    private DateTimeOffset? _creatorFirstSell;

    // Wallet reputation (FE-T10): the first distinct buyers (one spare, so the creator can be left out once known)
    // and the first buys with their block, both sorted by Ts.
    private readonly int _earlyCount;
    private readonly List<(DateTimeOffset Ts, string Wallet)> _earlyBuyers = [];
    private readonly List<(DateTimeOffset Ts, long? Block)> _firstBuys = [];
    private readonly HashSet<string> _smartTraders = new(StringComparer.Ordinal);
    private readonly Dictionary<string, bool> _smartCache = new(StringComparer.Ordinal);
    private long? _createBlock;

    // Holder-supply ledger (FE-T13): every wallet's net token position on this token, for its lifetime.
    private readonly HolderLedger _ledger;
    private readonly double _holderLargeShare;

    private DateTimeOffset? _prevSnapshotTs;
    private decimal _prevVolume;
    private int _prevHolders;
    private decimal? _prevPrice;
    private double? _prevEma;

    public TokenState(string mint, int windowSeconds, int emaSpan, int rsiPeriod, DateTimeOffset firstSeen,
        decimal largeBuySol = PipelineOptions.DefaultLargeBuySol, MomentumWindows? momentum = null, int earlyBuyers = 20,
        double holderLargeShare = PipelineOptions.DefaultHolderLargeShare, int holderMaxWallets = HolderLedger.DefaultMaxWallets)
    {
        _holderLargeShare = holderLargeShare;
        _ledger = new HolderLedger(holderMaxWallets);
        _earlyCount = earlyBuyers;
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

    /// <summary>Normalized name / ticker keys from the create event (FE-T12 narrative heat); empty without one.</summary>
    internal string[] NameKeys { get; private set; } = [];

    /// <summary>Whether the create's metadata URI is content-addressed (FE-T12); null without a create or a URI.</summary>
    internal bool? UriContentAddressed { get; private set; }

    /// <summary>Timestamp of the newest event received (applied or still queued).</summary>
    public DateTimeOffset LastActivity { get; private set; }

    /// <summary>Queues an event; it is applied by the first snapshot at or after its timestamp.</summary>
    public void Apply(RawEvent e)
    {
        if (e.Timestamp > LastActivity)
        {
            LastActivity = e.Timestamp;
        }

        // In-order events (the common case) append in O(1).
        int i = _pending.Count;
        while (i > 0 && _pending[i - 1].Timestamp > e.Timestamp)
        {
            i--;
        }

        _pending.Insert(i, e);
    }

    /// <summary>Applies the queued events dated at or before <paramref name="now"/>, oldest first.</summary>
    private void Advance(DateTimeOffset now)
    {
        int due = 0;
        while (due < _pending.Count && _pending[due].Timestamp <= now)
        {
            ApplyNow(_pending[due]);
            due++;
        }

        _pending.RemoveRange(0, due);
    }

    private void ApplyNow(RawEvent e)
    {
        bool isLatest = _latestApplied is not { } latest || e.Timestamp >= latest;
        if (isLatest)
        {
            _latestApplied = e.Timestamp;
        }

        if (e.Timestamp < _firstSeen)
        {
            _firstSeen = e.Timestamp;
        }

        if (e.Kind == EventKind.Create)
        {
            _hasCreate = true;
            _createdAt = e.Timestamp;
            _createBlock = e.Block ?? _createBlock;
            NameKeys = NarrativeKeys.Of(e.Name, e.Symbol);
            UriContentAddressed = Metadata.MetadataUri.IsContentAddressed(e.MetadataUri);
            ApplyCreator(e);
        }
        else if (e.Kind is EventKind.Buy or EventKind.Sell && _creator is not null && e.Trader == _creator)
        {
            ApplyCreatorTrade(e);
        }

        if (e.IsMigrated)
        {
            _isMigrated = true;
        }

        ApplyPositions(e);

        if (e.Trader is not null)
        {
            _holders.Add(e.Trader, e.Timestamp);
        }

        if (e.Kind is EventKind.Buy or EventKind.Sell)
        {
            Insert(new Observation(e.Timestamp, e.Kind, e.QuoteAmount, e.QuoteMint, e.Trader));
        }

        if (e.Kind == EventKind.Buy)
        {
            AddFirstBuy(e.Timestamp, e.Block);
            if (e.Trader is { } buyer)
            {
                AddEarlyBuyer(e.Timestamp, buyer);
            }
        }

        // A late event's facts are already stale: keep only what's newest.
        if (!isLatest)
        {
            return;
        }

        _supply = e.Supply ?? _supply;

        // Another pool's currency: its price, cap, liquidity and pool don't belong to the SOL series.
        if (!IsSolQuoted(e.QuoteMint))
        {
            return;
        }

        _quoteMint = e.QuoteMint ?? _quoteMint;

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
        _venue = e.Venue ?? _venue;
        // Burned liquidity belongs to a pool: a move to another pool drops the old pool's reading
        // (unknown until that pool reports its own), so a 100%-burned first pool can't vouch for it.
        if (e.PoolId is { } poolId && _poolId is not null && poolId != _poolId)
        {
            _burned = null;
        }

        _poolId = e.PoolId ?? _poolId;
        _burned = e.BurnedLiquidityPercent ?? _burned;
    }

    public bool HasActivityInWindow(DateTimeOffset now)
    {
        Advance(now);
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

    /// <param name="now">Snapshot time.</param>
    /// <param name="wallets">
    /// Cross-token wallet memory, already advanced to <paramref name="now"/>; null leaves <see cref="FeatureSnapshot.Wallet"/> null.
    /// </param>
    public FeatureSnapshot Snapshot(DateTimeOffset now, WalletStore? wallets = null)
    {
        Advance(now);
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

            // Only SOL amounts can be summed.
            bool inSol = IsSolQuoted(o.QuoteMint);
            _concentration.Add(o.Trader, o.Ts, inSol ? o.QuoteAmount ?? 0 : null);
            if (inSol)
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
        double observedSec = (now - _firstSeen).TotalSeconds;
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
            BurnedLiquidity = _burned,
            Wallet = wallets is null ? null : WalletFeatures(now, wallets),
            Reputation = wallets is null ? null : ReputationFeatures(now, wallets),
            HolderSupply = _ledger.Snapshot(_supply ?? _createSupply, _holderLargeShare, incomplete: !_hasCreate),
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
        return snapshot;
    }

    /// <summary>
    /// Moves the event's tokens in the holder ledger (FE-T13): the create's dev buy and buys to the signer, sells and
    /// liquidity added out of it, liquidity removed back in, transfer legs between wallets. Pool accounts are excluded
    /// first, so a transfer into a pool never reads as a holder.
    /// </summary>
    private void ApplyPositions(RawEvent e)
    {
        _ledger.Exclude(Mint);
        if (e.PoolId is { } pool)
        {
            _ledger.Exclude(pool);
        }

        if (e.Kind == EventKind.Transfer)
        {
            foreach (TokenTransfer leg in (IEnumerable<TokenTransfer>?)e.TokenTransfers ?? [])
            {
                _ledger.Add(leg.From, -leg.Amount, e.Timestamp);
                _ledger.Add(leg.To, leg.Amount, e.Timestamp);
            }

            return;
        }

        if (e.Trader is not { } trader || e.TokenAmount is not { } tokens || tokens <= 0)
        {
            return;
        }

        decimal delta = e.Kind switch
        {
            EventKind.Create or EventKind.Buy or EventKind.RemoveLiquidity => tokens,
            EventKind.Sell or EventKind.AddLiquidity => -tokens,
            _ => 0,
        };
        _ledger.Add(trader, delta, e.Timestamp);
    }

    private void ApplyCreator(RawEvent e)
    {
        if (e.Trader is null)
        {
            return;
        }

        _creator = e.Trader;
        _createSupply = e.Supply ?? _createSupply;
        _creatorBuySol = IsSolQuoted(e.QuoteMint) ? e.QuoteAmount ?? 0 : null;
        _creatorBuyTokens = e.TokenAmount ?? 0;
        _creatorBoughtTokens += _creatorBuyTokens.Value;
    }

    private void ApplyCreatorTrade(RawEvent e)
    {
        decimal tokens = e.TokenAmount ?? 0;
        if (e.Kind == EventKind.Buy)
        {
            _creatorBoughtTokens += tokens;
            return;
        }

        _creatorSoldTokens += tokens;
        if (_creatorFirstSell is not { } first || e.Timestamp < first)
        {
            _creatorFirstSell = e.Timestamp;
        }
    }

    /// <summary>Freshness of the window's signers (from <see cref="TraderConcentration"/>) and the creator's behaviour.</summary>
    private WalletFeatures WalletFeatures(DateTimeOffset now, WalletStore wallets)
    {
        (double? fresh1h, double? fresh24h, double? medianAge) = wallets.TraderStats(_concentration.Traders, now);
        var features = new WalletFeatures
        {
            CoverageSec = wallets.CoverageSec(now),
            FreshShare1h = fresh1h,
            FreshShare24h = fresh24h,
            MedianTraderAgeSec = medianAge,
        };

        if (_creator is null)
        {
            return features;
        }

        (int? creatorAge, int? previousLaunches) = wallets.CreatorStats(_creator, _createdAt, now);
        decimal? supply = _createSupply ?? _supply;
        return features with
        {
            CreatorAgeSec = creatorAge,
            CreatorPrevLaunches = previousLaunches,
            CreatorBuySol = (double?)_creatorBuySol,
            CreatorBuySupplyShare = _creatorBuyTokens is { } tokens && supply is > 0 ? (double)(tokens / supply.Value) : null,
            CreatorSoldShare = _creatorBoughtTokens > 0 ? Math.Min(1, (double)(_creatorSoldTokens / _creatorBoughtTokens)) : null,
            CreatorFirstSellSec = _creatorFirstSell is { } sold ? (int)Math.Max(0, (now - sold).TotalSeconds) : null,
        };
    }

    /// <summary>Creator track record, early-buyer quality, smart-money flow over the window and launch bundling.</summary>
    private ReputationFeatures ReputationFeatures(DateTimeOffset now, WalletStore wallets)
    {
        var buyers = new List<string>(_earlyCount);
        foreach ((DateTimeOffset ts, string wallet) in _earlyBuyers)
        {
            if (ts <= now && wallet != _creator && buyers.Count < _earlyCount)
            {
                buyers.Add(wallet);
            }
        }

        (double? knownShare, double? winRate, double? pnl) = wallets.EarlyBuyerStats(buyers, now);
        (int? settled, double? rugShare, double? pumpShare) = _creator is null
            ? (null, null, null)
            : wallets.CreatorReputation(_creator, _createdAt, now);
        (double? createShare, double? topShare) = Bundling(now);
        var features = new ReputationFeatures
        {
            CreatorSettledLaunches = settled,
            CreatorRugShare = rugShare,
            CreatorPumpShare = pumpShare,
            EarlyBuyers = buyers.Count,
            EarlyKnownShare = knownShare,
            EarlyWinRate = winRate,
            EarlyPnlSol = pnl,
            BundleCreateBlockShare = createShare,
            BundleTopBlockShare = topShare,
        };

        return wallets.ReputationCovered(now) ? SmartFlow(now, wallets, features) : features;
    }

    /// <summary>The window's SOL-quoted volume from smart wallets (as buy_pressure: SOL-quoted, reported amounts only).</summary>
    private ReputationFeatures SmartFlow(DateTimeOffset now, WalletStore wallets, ReputationFeatures features)
    {
        _smartCache.Clear();
        _smartTraders.Clear();
        decimal buySol = 0;
        decimal smartBuySol = 0;
        decimal smartNet = 0;
        foreach (Observation o in _window)
        {
            if (o.Ts > now)
            {
                break;
            }

            if (o.QuoteMint != RawEvent.WrappedSolMint || o.QuoteAmount is not { } sol)
            {
                continue;
            }

            bool buy = o.Kind == EventKind.Buy;
            buySol += buy ? sol : 0;
            if (o.Trader is not { } trader)
            {
                continue;
            }

            if (!_smartCache.TryGetValue(trader, out bool smart))
            {
                smart = wallets.IsSmart(trader, now);
                _smartCache[trader] = smart;
            }

            if (!smart)
            {
                continue;
            }

            _smartTraders.Add(trader);
            smartBuySol += buy ? sol : 0;
            smartNet += buy ? sol : -sol;
        }

        return features with
        {
            SmartBuyShare = buySol == 0 ? null : (double)(smartBuySol / buySol),
            SmartNetFlowSol = (double)smartNet,
            SmartTraders = _smartTraders.Count,
        };
    }

    /// <summary>Shares of the first buys (those reporting a block) in the creation block, and in the busiest single block.</summary>
    private (double? CreateBlockShare, double? TopBlockShare) Bundling(DateTimeOffset now)
    {
        var perBlock = new Dictionary<long, int>();
        int withBlock = 0;
        foreach ((DateTimeOffset ts, long? block) in _firstBuys)
        {
            if (ts > now)
            {
                break;
            }

            if (block is not { } b)
            {
                continue;
            }

            withBlock++;
            perBlock[b] = perBlock.GetValueOrDefault(b) + 1;
        }

        if (withBlock == 0)
        {
            return (null, null);
        }

        double? inCreate = _createBlock is { } created ? (double)perBlock.GetValueOrDefault(created) / withBlock : null;
        return (inCreate, (double)perBlock.Values.Max() / withBlock);
    }

    private void AddFirstBuy(DateTimeOffset ts, long? block)
    {
        if (_firstBuys.Count >= _earlyCount && ts >= _firstBuys[^1].Ts)
        {
            return;
        }

        int i = _firstBuys.Count;
        while (i > 0 && _firstBuys[i - 1].Ts > ts)
        {
            i--;
        }

        _firstBuys.Insert(i, (ts, block));
        if (_firstBuys.Count > _earlyCount)
        {
            _firstBuys.RemoveAt(_firstBuys.Count - 1);
        }
    }

    /// <summary>Keeps the first <c>N + 1</c> distinct buyers by their first buy (late events included).</summary>
    private void AddEarlyBuyer(DateTimeOffset ts, string wallet)
    {
        int capacity = _earlyCount + 1;
        int existing = _earlyBuyers.FindIndex(b => b.Wallet == wallet);
        if (existing >= 0)
        {
            if (_earlyBuyers[existing].Ts <= ts)
            {
                return;
            }

            _earlyBuyers.RemoveAt(existing);
        }
        else if (_earlyBuyers.Count >= capacity && ts >= _earlyBuyers[^1].Ts)
        {
            return;
        }

        int i = _earlyBuyers.Count;
        while (i > 0 && _earlyBuyers[i - 1].Ts > ts)
        {
            i--;
        }

        _earlyBuyers.Insert(i, (ts, wallet));
        if (_earlyBuyers.Count > capacity)
        {
            _earlyBuyers.RemoveAt(_earlyBuyers.Count - 1);
        }
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

    /// <summary>SOL, or no quote mint reported (a source that only ever quotes in SOL).</summary>
    private static bool IsSolQuoted(string? quoteMint) => quoteMint is null || quoteMint == RawEvent.WrappedSolMint;

    private void Prune(DateTimeOffset now)
    {
        DateTimeOffset cutoff = now - TimeSpan.FromSeconds(_windowSeconds);

        int stale = 0;
        while (stale < _window.Count && _window[stale].Ts < cutoff)
        {
            stale++;
        }

        _window.RemoveRange(0, stale);
        _ledger.Prune(cutoff);

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
