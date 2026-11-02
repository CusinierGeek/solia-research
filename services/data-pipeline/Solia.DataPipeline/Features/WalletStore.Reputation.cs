using System.Runtime.InteropServices;
using Solia.DataPipeline.Ingestion;

namespace Solia.DataPipeline.Features;

/// <summary>A closed buy → sell round trip of one wallet on one token: when its first buy was, and its realized P&amp;L in SOL.</summary>
public readonly record struct RoundTrip(long Opened, double PnlSol)
{
    public bool IsWin => PnlSol > 0;
}

/// <summary>A launch's recorded outcome over its first hour (FE-T10): whether it pumped and / or rugged.</summary>
public readonly record struct LaunchOutcome(long Launch, bool Rugged, bool Pumped);

/// <summary>
/// Wallet reputation (FE-T10, issue #190): behaviour statistics from outcomes, not identities, so they stay valid as
/// wallets rotate. Built on the same point-in-time queue as the rest of the store (only events strictly before a
/// snapshot are applied).
/// <para>
/// <b>Round trips</b>: per (wallet, token), SOL-quoted buys open / grow a position (tokens + SOL cost) and sells
/// realize P&amp;L at the average cost; the position closes when (almost) every token bought is sold, which records
/// one <see cref="RoundTrip"/>. A position never closed stays open — never a loss. Positions live in fixed UTC buckets
/// of <c>Pipeline:ReputationPositionMaxHours</c> (aligned on the Unix epoch): one still open at a bucket boundary is
/// dropped, so the positions depend only on the events since the last boundary. A wallet's win rate / P&amp;L count
/// its trips opened within the age cap minus one bucket, which therefore all opened after a boundary the history
/// covers: a replay warmed up over the age cap reproduces them exactly, whatever was open before its start.
/// </para>
/// <para>
/// <b>Launch outcomes</b>: a create starts tracking the token's SOL price over its first
/// <c>Pipeline:ReputationOutcomeHorizonSeconds</c> (the longest label horizon). The outcome — pumped: highest price
/// ≥ (1 + pump threshold) × first price; rugged: price fell by the rug threshold from its running peak — is recorded
/// on the creator only once the horizon <b>plus the label embargo</b> is over, so a snapshot never sees an outcome
/// that wasn't known at its <c>ts</c>, and can't leak a label.
/// </para>
/// <para>
/// Open positions and launches still being tracked are not persisted: after a live restart they start over.
/// </para>
/// </summary>
public sealed partial class WalletStore
{
    public const int MaxRecentTrips = 32;

    /// <summary>A position holding at most this share of its peak tokens is closed (dust left by rounding).</summary>
    internal const double CloseResidual = 0.01;

    private struct Position
    {
        public long Opened;
        public double Tokens;
        public double Cost;
        public double Realized;
        public double Peak;
    }

    private sealed class LaunchTrack(string mint, string creator, long created, double? first)
    {
        public string Mint { get; } = mint;
        public string Creator { get; } = creator;
        public long Created { get; } = created;
        public double? First { get; set; } = first;
        public double Max { get; set; } = first ?? 0;
        public double Peak { get; set; } = first ?? 0;
        public bool Rugged { get; set; }
    }

    private readonly long _positionMaxSec;
    private readonly int _minTrips;
    private readonly double _smartWinRate;
    private readonly long _outcomeHorizonSec;
    private readonly long _outcomeEmbargoSec;
    private readonly double _pumpMultiple;
    private readonly double _rugKeep;
    private readonly Dictionary<(string Wallet, string Mint), Position> _positions = [];
    private readonly Dictionary<string, LaunchTrack> _launches = new(StringComparer.Ordinal);
    private readonly List<LaunchTrack> _launchOrder = []; // sorted by Created

    /// <summary>Open buy → sell positions held in memory.</summary>
    internal int PositionCount => _positions.Count;

    /// <summary>Launches whose outcome isn't recorded yet.</summary>
    internal int TrackedLaunchCount => _launches.Count;

    /// <summary>Whether the history covers the reputation look-backs at <paramref name="now"/> (the age cap).</summary>
    public bool ReputationCovered(DateTimeOffset now) => CoverageSec(now) >= _capSec;

    /// <summary>
    /// The creator's launches before this token's creation, within the age cap, whose outcome is recorded at
    /// <paramref name="now"/>, and the share of them that rugged / pumped. Null while the history doesn't cover the
    /// age cap; an unknown creator has 0 recorded launches; the shares are null without one.
    /// </summary>
    public (int? Settled, double? RugShare, double? PumpShare) CreatorReputation(
        string creator, DateTimeOffset createdAt, DateTimeOffset now)
    {
        if (!ReputationCovered(now))
        {
            return (null, null, null);
        }

        long nowSec = Seconds(now);
        long createdSec = Seconds(createdAt);
        int settled = 0;
        int rugged = 0;
        int pumped = 0;
        if (TryGetLive(creator, nowSec, out Wallet w))
        {
            foreach (LaunchOutcome o in w.Outcomes ?? [])
            {
                if (o.Launch >= nowSec - _capSec && o.Launch < createdSec)
                {
                    settled++;
                    rugged += o.Rugged ? 1 : 0;
                    pumped += o.Pumped ? 1 : 0;
                }
            }
        }

        return settled == 0 ? (0, null, null) : (settled, (double)rugged / settled, (double)pumped / settled);
    }

    /// <summary>
    /// A wallet's closed round trips counted at <paramref name="now"/> (opened within the age cap minus the position
    /// life), how many were wins, and their summed realized P&amp;L in SOL. An unknown wallet has none.
    /// </summary>
    public (int Trips, int Wins, double PnlSol) TripStats(string wallet, DateTimeOffset now)
    {
        long nowSec = Seconds(now);
        if (!TryGetLive(wallet, nowSec, out Wallet w) || w.Trips is null)
        {
            return (0, 0, 0);
        }

        long since = nowSec - (_capSec - _positionMaxSec);
        int trips = 0;
        int wins = 0;
        double pnl = 0;
        foreach (RoundTrip t in w.Trips)
        {
            if (t.Opened >= since)
            {
                trips++;
                wins += t.IsWin ? 1 : 0;
                pnl += t.PnlSol;
            }
        }

        return (trips, wins, pnl);
    }

    /// <summary>Whether the wallet has enough history (<c>Pipeline:ReputationMinTrips</c>) to be rated.</summary>
    public bool IsKnown(string wallet, DateTimeOffset now) => TripStats(wallet, now).Trips >= _minTrips;

    /// <summary>A known wallet whose win rate is above <c>Pipeline:ReputationSmartWinRate</c>.</summary>
    public bool IsSmart(string wallet, DateTimeOffset now)
    {
        (int trips, int wins, _) = TripStats(wallet, now);
        return trips >= _minTrips && (double)wins / trips > _smartWinRate;
    }

    /// <summary>
    /// Quality of a token's early buyers: the share of them that are known (enough round trips), and the known ones'
    /// mean win rate and mean P&amp;L per trip (SOL). Null while the history doesn't cover the age cap or without a
    /// buyer; the win rate and P&amp;L are null when no buyer is known.
    /// </summary>
    public (double? KnownShare, double? WinRate, double? PnlPerTripSol) EarlyBuyerStats(
        IReadOnlyCollection<string> buyers, DateTimeOffset now)
    {
        if (buyers.Count == 0 || !ReputationCovered(now))
        {
            return (null, null, null);
        }

        int known = 0;
        double winRate = 0;
        double pnl = 0;
        foreach (string buyer in buyers)
        {
            (int trips, int wins, double sum) = TripStats(buyer, now);
            if (trips < _minTrips)
            {
                continue;
            }

            known++;
            winRate += (double)wins / trips;
            pnl += sum / trips;
        }

        return known == 0
            ? (0, null, null)
            : ((double)known / buyers.Count, winRate / known, pnl / known);
    }

    private static Pending ToPending(RawEvent e)
    {
        bool solSeries = e.QuoteMint is null || e.QuoteMint == RawEvent.WrappedSolMint;
        bool inSol = e.QuoteMint == RawEvent.WrappedSolMint;
        return new Pending(
            e.Timestamp,
            e.Trader,
            e.Mint,
            e.Kind,
            solSeries && e.Price is > 0 ? (double)e.Price.Value : null,
            inSol && e.QuoteAmount is > 0 ? (double)e.QuoteAmount.Value : null,
            e.TokenAmount is > 0 ? (double)e.TokenAmount.Value : null);
    }

    /// <summary>Applies a SOL-quoted buy / sell to the (wallet, token) position; returns the round trip it closes, if any.</summary>
    private RoundTrip? UpdatePosition(string wallet, string mint, long sec, bool buy, double sol, double tokens)
    {
        ref Position p = ref CollectionsMarshal.GetValueRefOrAddDefault(_positions, (wallet, mint), out bool exists);
        if (exists && Bucket(p.Opened) != Bucket(sec))
        {
            exists = false; // crossed a bucket boundary: dropped, still open (never a loss)
        }

        if (buy)
        {
            if (!exists)
            {
                p = new Position { Opened = sec };
            }

            p.Tokens += tokens;
            p.Cost += sol;
            p.Peak = Math.Max(p.Peak, p.Tokens);
            return null;
        }

        if (!exists || p.Tokens <= 0)
        {
            _positions.Remove((wallet, mint)); // a sell of tokens bought before the history (or not in SOL)
            return null;
        }

        double sold = Math.Min(tokens, p.Tokens);
        double costOut = p.Cost * (sold / p.Tokens);
        p.Realized += (sol * (sold / tokens)) - costOut;
        p.Tokens -= sold;
        p.Cost -= costOut;
        if (p.Tokens > p.Peak * CloseResidual)
        {
            return null;
        }

        var trip = new RoundTrip(p.Opened, p.Realized);
        _positions.Remove((wallet, mint));
        return trip;
    }

    private void TrackLaunch(string mint, string creator, long sec, double? price)
    {
        if (_launches.ContainsKey(mint))
        {
            return;
        }

        var track = new LaunchTrack(mint, creator, sec, price);
        _launches[mint] = track;
        int i = _launchOrder.Count;
        while (i > 0 && _launchOrder[i - 1].Created > sec)
        {
            i--;
        }

        _launchOrder.Insert(i, track);
    }

    /// <summary>A SOL-series price of a launch still within its outcome horizon.</summary>
    private void UpdateLaunch(string mint, long sec, double price)
    {
        if (!_launches.TryGetValue(mint, out LaunchTrack? track) || sec < track.Created || sec > track.Created + _outcomeHorizonSec)
        {
            return;
        }

        if (track.First is null)
        {
            track.First = price;
            track.Max = price;
            track.Peak = price;
            return;
        }

        track.Max = Math.Max(track.Max, price);
        track.Peak = Math.Max(track.Peak, price);
        if (price <= track.Peak * _rugKeep)
        {
            track.Rugged = true;
        }
    }

    /// <summary>Records on their creators the outcomes whose horizon + embargo ended strictly before <paramref name="nowSec"/>.</summary>
    private void SettleLaunches(long nowSec)
    {
        int settled = 0;
        while (settled < _launchOrder.Count && _launchOrder[settled].Created + _outcomeHorizonSec + _outcomeEmbargoSec < nowSec)
        {
            LaunchTrack track = _launchOrder[settled];
            _launches.Remove(track.Mint);
            ref Wallet w = ref CollectionsMarshal.GetValueRefOrNullRef(_wallets, track.Creator);
            if (!System.Runtime.CompilerServices.Unsafe.IsNullRef(ref w) && w.LastSeen >= nowSec - _ttlSec)
            {
                bool pumped = track.First is { } first && track.Max >= first * _pumpMultiple;
                w.Outcomes = AddOutcome(w.Outcomes, new LaunchOutcome(track.Created, track.Rugged, pumped));
                _dirty?.Add(track.Creator);
            }

            settled++;
        }

        _launchOrder.RemoveRange(0, settled);
    }

    private void SweepPositions(long nowSec)
    {
        foreach (((string, string) key, Position p) in _positions)
        {
            if (Bucket(p.Opened) != Bucket(nowSec))
            {
                _positions.Remove(key); // allowed while enumerating
            }
        }
    }

    private long Bucket(long sec) => Math.DivRem(sec, _positionMaxSec, out long rem) - (rem < 0 ? 1 : 0);

    private static RoundTrip[] AddTrip(RoundTrip[]? trips, RoundTrip trip) =>
        InsertBounded(trips, trip, t => t.Opened, MaxRecentTrips);

    private static LaunchOutcome[] AddOutcome(LaunchOutcome[]? outcomes, LaunchOutcome outcome) =>
        InsertBounded(outcomes, outcome, o => o.Launch, MaxRecentLaunches);

    /// <summary>Inserts after every item with the same or an earlier key (stable), keeping the latest <paramref name="max"/>.</summary>
    private static T[] InsertBounded<T>(T[]? items, T item, Func<T, long> key, int max)
    {
        var list = new List<T>(items ?? []);
        int i = list.Count;
        while (i > 0 && key(list[i - 1]) > key(item))
        {
            i--;
        }

        list.Insert(i, item);
        return list.Count > max ? list.GetRange(list.Count - max, max).ToArray() : list.ToArray();
    }
}
