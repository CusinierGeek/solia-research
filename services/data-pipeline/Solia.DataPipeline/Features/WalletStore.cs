using System.Runtime.InteropServices;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Features;

/// <summary>One wallet's persisted state (<c>wallet_state</c> row). Times are Unix seconds.</summary>
/// <param name="RecentLaunches">Seconds of the wallet's latest token creations, ascending, at most <see cref="WalletStore.MaxRecentLaunches"/>.</param>
/// <param name="Trips">Its latest closed round trips (FE-T10), by opening time, at most <see cref="WalletStore.MaxRecentTrips"/>; null = none.</param>
/// <param name="Outcomes">Its latest launches' recorded outcomes (FE-T10), by launch time, at most <see cref="WalletStore.MaxRecentLaunches"/>; null = none.</param>
public sealed record WalletRecord(
    string Wallet, long FirstSeen, long LastSeen, int TradeCount, int TokensTraded, int TokensCreated, long[] RecentLaunches,
    RoundTrip[]? Trips = null, LaunchOutcome[]? Outcomes = null);

/// <summary>The whole persisted wallet state, as loaded at startup.</summary>
/// <param name="HistoryStart">Start of the wallet history behind the state (feature coverage).</param>
/// <param name="LastEvent">Newest event applied when it was saved.</param>
public sealed record WalletStateSnapshot(DateTimeOffset HistoryStart, DateTimeOffset LastEvent, IReadOnlyList<WalletRecord> Wallets);

/// <summary>What changed since the last save: wallets to upsert, evicted wallets to delete, and the history bounds.</summary>
public sealed record WalletStateChanges(
    DateTimeOffset HistoryStart, DateTimeOffset LastEvent, IReadOnlyList<WalletRecord> Upserts, IReadOnlyList<string> Deletes);

/// <summary>
/// Cross-token wallet memory (FE-T9, issue #189), derived for free from the signers (<see cref="RawEvent.Trader"/>,
/// pumpapi <c>txSigner</c>) of the stream's trades and creates. Per wallet: first / last seen, trade count, tokens
/// traded, tokens created and the times of its latest launches. The model never sees an address, only the
/// behavioural statistics built from them (<see cref="TraderStats"/>, <see cref="CreatorStats"/>), so an unknown
/// wallet simply counts as fresh.
/// <para>
/// <b>Point-in-time:</b> <see cref="Apply"/> only queues an event; <see cref="Advance"/> to <c>now</c> applies the
/// queued events dated <b>strictly before</b> <c>now</c>, in timestamp order, so a snapshot's lookups never see an
/// event at or after its <c>ts</c>. An event older than what was already advanced to is applied at once (it is in
/// the past of every later snapshot). The result depends on event timestamps and arrival order only — the same in
/// live and in replay.
/// </para>
/// <para>
/// <b>Bounded:</b> a wallet inactive for longer than <see cref="PipelineOptions.WalletTtlHours"/> is forgotten:
/// logically at once (lookups and updates treat it as absent, so the result never depends on when a sweep ran),
/// physically by an hourly sweep. Features only look back <see cref="PipelineOptions.WalletAgeCapHours"/>: ages are
/// capped at it and launches counted within it, and the history-dependent ones stay null until that much history
/// has been observed (<see cref="CoverageSec"/>), so a replay with that much warm-up reproduces them.
/// </para>
/// <para>
/// <b>Persistence</b> (live only, <paramref name="trackChanges"/>): changed and evicted wallets are collected for
/// <see cref="DrainChanges"/>; <see cref="Load"/> restores a saved state. A restored state whose first new event comes
/// more than <see cref="StaleGap"/> after the saved one restarts the coverage: wallets that traded during the outage
/// would otherwise look fresher than they are.
/// </para>
/// Not thread-safe: fed and queried from the pipeline's single loop.
/// </summary>
public sealed partial class WalletStore
{
    public static readonly TimeSpan FreshShort = TimeSpan.FromHours(1);
    public static readonly TimeSpan FreshLong = TimeSpan.FromHours(24);
    public static readonly TimeSpan StaleGap = TimeSpan.FromHours(1);
    internal static readonly TimeSpan SweepInterval = TimeSpan.FromHours(1);
    public const int MaxRecentLaunches = 16;

    private static readonly long FreshShortSec = (long)FreshShort.TotalSeconds;
    private static readonly long FreshLongSec = (long)FreshLong.TotalSeconds;
    private static readonly long StaleGapSec = (long)StaleGap.TotalSeconds;
    private static readonly long SweepSec = (long)SweepInterval.TotalSeconds;

    private struct Wallet
    {
        public long FirstSeen;
        public long LastSeen;
        public int TradeCount;
        public int TokensTraded;
        public int TokensCreated;
        public int Mint0, Mint1, Mint2, Mint3; // stable hashes of the latest distinct mints traded (0 = empty)
        public long[]? Launches;
        public RoundTrip[]? Trips;        // FE-T10, ascending by Opened
        public LaunchOutcome[]? Outcomes; // FE-T10, ascending by Launch
    }

    // Wallet is null for an unsigned trade: it only feeds the launch outcomes (FE-T10).
    // Price: SOL-series price (SOL or unreported quote); Sol / Tokens: SOL-quoted amounts only, as buy_pressure.
    private readonly record struct Pending(
        DateTimeOffset Ts, string? Wallet, string Mint, EventKind Kind, double? Price, double? Sol, double? Tokens)
    {
        public bool IsCreate => Kind == EventKind.Create;
    }

    private readonly long _ttlSec;
    private readonly long _capSec;
    private readonly Dictionary<string, Wallet> _wallets = new(StringComparer.Ordinal);
    private readonly List<Pending> _pending = []; // sorted by Ts
    private readonly HashSet<string>? _dirty;
    private readonly HashSet<string>? _deleted;
    private readonly List<long> _ages = [];
    private DateTimeOffset? _advancedTo;
    private long? _historyStart;
    private long? _lastEvent;
    private long _nextSweep = long.MinValue;
    private bool _restored;

    public WalletStore(PipelineOptions options, bool trackChanges = false)
    {
        _ttlSec = options.WalletTtlHours * 3600L;
        _capSec = options.WalletAgeCapHours * 3600L;
        _positionMaxSec = options.ReputationPositionMaxHours * 3600L;
        _minTrips = options.ReputationMinTrips;
        _smartWinRate = options.ReputationSmartWinRate;
        _outcomeHorizonSec = options.ReputationOutcomeHorizonSeconds;
        _outcomeEmbargoSec = options.ReputationOutcomeEmbargoSeconds;
        _pumpMultiple = 1 + options.ReputationPumpThreshold;
        _rugKeep = 1 - options.ReputationRugThreshold;
        if (trackChanges)
        {
            _dirty = new HashSet<string>(StringComparer.Ordinal);
            _deleted = new HashSet<string>(StringComparer.Ordinal);
        }
    }

    /// <summary>Wallets held in memory (including logically expired ones not swept yet).</summary>
    public int Count => _wallets.Count;

    /// <summary>Events queued, waiting for a snapshot at or after their time.</summary>
    internal int PendingCount => _pending.Count;

    /// <summary>Whether changes are collected for persistence.</summary>
    public bool TracksChanges => _dirty is not null;

    /// <summary>
    /// Queues a trade's or a create's signer; any other event is ignored, and an unsigned trade only feeds the launch
    /// outcomes (an unsigned create is ignored). An event dated more than the TTL away from the last advance (a bogus
    /// timestamp) is dropped: ahead, it would sit in the queue for that long; behind, it would only describe wallets
    /// already forgotten.
    /// </summary>
    public void Apply(RawEvent e)
    {
        if (e.Kind is not (EventKind.Buy or EventKind.Sell or EventKind.Create)
            || (e.Trader is null && (e.Kind == EventKind.Create || e.Price is null)))
        {
            return;
        }

        var ttl = TimeSpan.FromSeconds(_ttlSec);
        if (_advancedTo is { } reference && (e.Timestamp > reference + ttl || e.Timestamp < reference - ttl))
        {
            return;
        }

        var pending = ToPending(e);
        if (_advancedTo is { } advanced && e.Timestamp < advanced)
        {
            ApplyNow(pending); // already in the past of every later snapshot
            return;
        }

        int i = _pending.Count; // in-order events (the common case) append in O(1)
        while (i > 0 && _pending[i - 1].Ts > e.Timestamp)
        {
            i--;
        }

        _pending.Insert(i, pending);
    }

    /// <summary>
    /// Applies the queued events dated strictly before <paramref name="now"/>, oldest first, and sweeps the expired
    /// wallets at most once an hour. Idempotent; call before the lookups of a snapshot at <paramref name="now"/>.
    /// </summary>
    public void Advance(DateTimeOffset now)
    {
        int due = 0;
        while (due < _pending.Count && _pending[due].Ts < now)
        {
            ApplyNow(_pending[due]);
            due++;
        }

        _pending.RemoveRange(0, due);
        if (_advancedTo is not { } advanced || now > advanced)
        {
            _advancedTo = now;
        }

        long nowSec = Seconds(now);
        SettleLaunches(nowSec);
        if (nowSec >= _nextSweep)
        {
            Sweep(nowSec);
            SweepPositions(nowSec);
            _nextSweep = nowSec + SweepSec;
        }
    }

    /// <summary>Seconds of wallet history behind the state at <paramref name="now"/>, capped at the age cap.</summary>
    public int CoverageSec(DateTimeOffset now) =>
        _historyStart is { } start ? (int)Math.Clamp(Seconds(now) - start, 0, _capSec) : 0;

    /// <summary>
    /// Freshness of a token's window traders at <paramref name="now"/>: the share first seen less than 1 h / 24 h
    /// ago (an unknown wallet is fresh: age 0), and their median age in seconds, capped at the age cap. Each is null
    /// without a trader, or while the history doesn't cover its horizon yet (1 h; 24 h; the age cap).
    /// </summary>
    public (double? FreshShare1h, double? FreshShare24h, double? MedianAgeSec) TraderStats(
        IEnumerable<string> traders, DateTimeOffset now)
    {
        long nowSec = Seconds(now);
        _ages.Clear();
        int fresh1h = 0;
        int fresh24h = 0;
        foreach (string trader in traders)
        {
            long age = Age(trader, nowSec);
            _ages.Add(age);
            fresh1h += age < FreshShortSec ? 1 : 0;
            fresh24h += age < FreshLongSec ? 1 : 0;
        }

        int n = _ages.Count;
        if (n == 0)
        {
            return (null, null, null);
        }

        int coverage = CoverageSec(now);
        return (
            coverage >= FreshShortSec ? (double)fresh1h / n : null,
            coverage >= FreshLongSec ? (double)fresh24h / n : null,
            coverage >= _capSec ? Median() : null);
    }

    /// <summary>
    /// The creator's age (seconds, capped at the age cap) and its launches before this token's creation within the
    /// age cap (at most <see cref="MaxRecentLaunches"/>). Both null while the history doesn't cover the age cap; a
    /// creator not seen before <paramref name="now"/> is a fresh wallet with no launch.
    /// </summary>
    public (int? AgeSec, int? PreviousLaunches) CreatorStats(string creator, DateTimeOffset createdAt, DateTimeOffset now)
    {
        if (CoverageSec(now) < _capSec)
        {
            return (null, null);
        }

        long nowSec = Seconds(now);
        if (!TryGetLive(creator, nowSec, out Wallet w))
        {
            return (0, 0);
        }

        long createdSec = Seconds(createdAt);
        int previous = 0;
        foreach (long launch in w.Launches ?? [])
        {
            if (launch >= nowSec - _capSec && launch < createdSec)
            {
                previous++;
            }
        }

        return ((int)Math.Clamp(nowSec - w.FirstSeen, 0, _capSec), previous);
    }

    /// <summary>The wallet's state as last applied, or null when unknown or expired at <paramref name="now"/>.</summary>
    public WalletRecord? Get(string wallet, DateTimeOffset now) =>
        TryGetLive(wallet, Seconds(now), out Wallet w) ? ToRecord(wallet, w) : null;

    /// <summary>
    /// Collects the wallets changed and evicted since the last call (live persistence), and stops tracking them.
    /// Null when changes aren't tracked or nothing has been observed yet.
    /// </summary>
    public WalletStateChanges? DrainChanges()
    {
        if (_dirty is null || _deleted is null || _historyStart is not { } start || _lastEvent is not { } last)
        {
            return null;
        }

        var upserts = new List<WalletRecord>(_dirty.Count);
        foreach (string wallet in _dirty)
        {
            if (_wallets.TryGetValue(wallet, out Wallet w))
            {
                upserts.Add(ToRecord(wallet, w));
            }
        }

        var deletes = _deleted.ToList();
        _dirty.Clear();
        _deleted.Clear();
        return new WalletStateChanges(DateTimeOffset.FromUnixTimeSeconds(start), DateTimeOffset.FromUnixTimeSeconds(last),
            upserts, deletes);
    }

    /// <summary>Puts back drained changes whose save failed, so the next save retries them (with current values).</summary>
    public void RestoreChanges(WalletStateChanges changes)
    {
        if (_dirty is null || _deleted is null)
        {
            return;
        }

        foreach (WalletRecord r in changes.Upserts)
        {
            if (_wallets.ContainsKey(r.Wallet))
            {
                _dirty.Add(r.Wallet);
            }
        }

        foreach (string wallet in changes.Deletes)
        {
            if (!_wallets.ContainsKey(wallet))
            {
                _deleted.Add(wallet);
            }
        }
    }

    /// <summary>Restores a saved state into an empty store (before any event).</summary>
    public void Load(WalletStateSnapshot snapshot)
    {
        if (_wallets.Count > 0 || _historyStart is not null)
        {
            throw new InvalidOperationException("wallet state can only be loaded into an empty store");
        }

        foreach (WalletRecord r in snapshot.Wallets)
        {
            long[] launches = r.RecentLaunches.Order().TakeLast(MaxRecentLaunches).ToArray();
            _wallets[r.Wallet] = new Wallet
            {
                FirstSeen = r.FirstSeen,
                LastSeen = r.LastSeen,
                TradeCount = r.TradeCount,
                TokensTraded = r.TokensTraded,
                TokensCreated = r.TokensCreated,
                Launches = launches.Length == 0 ? null : launches,
                Trips = r.Trips is { Length: > 0 } trips ? trips.OrderBy(t => t.Opened).TakeLast(MaxRecentTrips).ToArray() : null,
                Outcomes = r.Outcomes is { Length: > 0 } outcomes
                    ? outcomes.OrderBy(o => o.Launch).TakeLast(MaxRecentLaunches).ToArray()
                    : null,
            };
        }

        _historyStart = Seconds(snapshot.HistoryStart);
        _lastEvent = Seconds(snapshot.LastEvent);
        _restored = true;
    }

    private void ApplyNow(Pending p)
    {
        long sec = Seconds(p.Ts);
        if (!p.IsCreate && p.Price is { } price)
        {
            UpdateLaunch(p.Mint, sec, price);
        }

        if (p.Wallet is null)
        {
            return; // an unsigned trade only feeds the launch outcomes
        }

        if (_restored)
        {
            _restored = false;
            if (_lastEvent is { } saved && sec - saved > StaleGapSec)
            {
                _historyStart = sec; // the outage left a hole in the history: start covering again
            }
        }

        _historyStart = _historyStart is { } start && start <= sec ? start : sec;
        _lastEvent = _lastEvent is { } last && last >= sec ? last : sec;

        ref Wallet w = ref CollectionsMarshal.GetValueRefOrAddDefault(_wallets, p.Wallet, out bool exists);
        if (!exists || w.LastSeen < sec - _ttlSec)
        {
            w = new Wallet { FirstSeen = sec, LastSeen = sec }; // new, or forgotten after the TTL (reputation too)
        }
        else
        {
            w.FirstSeen = Math.Min(w.FirstSeen, sec);
            w.LastSeen = Math.Max(w.LastSeen, sec);
        }

        if (p.IsCreate)
        {
            w.TokensCreated++;
            w.Launches = AddLaunch(w.Launches, sec);
            TrackLaunch(p.Mint, p.Wallet, sec, p.Price);
        }
        else
        {
            w.TradeCount++;
            int h = StableHash(p.Mint);
            if (h != w.Mint0 && h != w.Mint1 && h != w.Mint2 && h != w.Mint3)
            {
                w.TokensTraded++;
                (w.Mint3, w.Mint2, w.Mint1, w.Mint0) = (w.Mint2, w.Mint1, w.Mint0, h);
            }
        }

        // The create's dev buy opens the creator's position like any buy.
        if (p.Sol is { } sol && p.Tokens is { } tokens
            && UpdatePosition(p.Wallet, p.Mint, sec, p.Kind != EventKind.Sell, sol, tokens) is { } trip)
        {
            w.Trips = AddTrip(w.Trips, trip);
        }

        _dirty?.Add(p.Wallet);
        _deleted?.Remove(p.Wallet);
    }

    private void Sweep(long nowSec)
    {
        foreach ((string wallet, Wallet w) in _wallets)
        {
            if (w.LastSeen < nowSec - _ttlSec)
            {
                _wallets.Remove(wallet); // allowed while enumerating
                _dirty?.Remove(wallet);
                _deleted?.Add(wallet);
            }
        }
    }

    private bool TryGetLive(string wallet, long nowSec, out Wallet w) =>
        _wallets.TryGetValue(wallet, out w) && w.LastSeen >= nowSec - _ttlSec;

    private long Age(string wallet, long nowSec) =>
        TryGetLive(wallet, nowSec, out Wallet w) ? Math.Clamp(nowSec - w.FirstSeen, 0, _capSec) : 0;

    private double Median()
    {
        Span<long> ages = CollectionsMarshal.AsSpan(_ages);
        ages.Sort();
        int mid = ages.Length / 2;
        return ages.Length % 2 == 1 ? ages[mid] : (ages[mid - 1] + ages[mid]) / 2.0;
    }

    private static long[] AddLaunch(long[]? launches, long sec)
    {
        if (launches is null)
        {
            return [sec];
        }

        var list = new List<long>(launches) { sec };
        list.Sort();
        return list.Count > MaxRecentLaunches ? list.GetRange(list.Count - MaxRecentLaunches, MaxRecentLaunches).ToArray() : list.ToArray();
    }

    private static WalletRecord ToRecord(string wallet, Wallet w) =>
        new(wallet, w.FirstSeen, w.LastSeen, w.TradeCount, w.TokensTraded, w.TokensCreated, w.Launches ?? [], w.Trips, w.Outcomes);

    /// <summary>FNV-1a, never 0 (0 marks an empty slot); stable across processes, unlike <see cref="string.GetHashCode()"/>.</summary>
    private static int StableHash(string s)
    {
        uint h = 2166136261;
        foreach (char c in s)
        {
            h = (h ^ c) * 16777619;
        }

        return h == 0 ? 1 : (int)h;
    }

    private static long Seconds(DateTimeOffset ts) => ts.ToUnixTimeSeconds();
}
