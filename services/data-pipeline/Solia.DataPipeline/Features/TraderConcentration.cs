using System.Runtime.InteropServices;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Holder-concentration proxy over one snapshot's window, from trade signers (pumpapi
/// <c>txSigner</c>): how much of the volume a few wallets drive, and how many of the recent traders
/// are new to the token. Rebuilt each snapshot (<see cref="Reset"/>) from the window's trades, so
/// its per-wallet state is bounded by the window, never by the token's lifetime; one instance is
/// reused across snapshots to spare the hot path an allocation per token per tick. Trades must be
/// added in timestamp order. A trade without a signer is skipped: lumping all of them into one
/// "unknown" wallet would fake a whale.
/// </summary>
/// <param name="holders">
/// The token's lifetime wallets: a trader first seen on the token before the recent span is
/// returning, even when that first trade has left the window. Once the counter is a sketch (it
/// keeps no wallet), the window's earliest trade stands in for the first one.
/// </param>
internal sealed class TraderConcentration(HolderCounter holders)
{
    private struct Trader
    {
        public decimal Volume;  // in the current quote currency
        public bool Counted;    // has a trade in the current quote currency
        public bool Recent;     // has a trade since the recent start
    }

    private readonly Dictionary<string, Trader> _traders = new(StringComparer.Ordinal);
    private DateTimeOffset _recentStart;
    private int _unique;
    private int _recent;
    private int _newRecent;
    private decimal _volume;

    /// <summary>Distinct signers of the window's trades in the current quote currency.</summary>
    public int UniqueTraders => _unique;

    /// <summary>Distinct signers of the window's trades, whatever their quote currency (FE-T9 wallet freshness).</summary>
    public IEnumerable<string> Traders => _traders.Keys;

    /// <summary>
    /// Share of the traders active since the recent start that are new to the token (first seen in
    /// that span, not returning); null without a recent signed trade.
    /// </summary>
    public double? NewTraderRatio => _recent == 0 ? null : (double)_newRecent / _recent;

    /// <summary>Clears the previous snapshot's state; trades after <paramref name="recentStart"/> are recent.</summary>
    public void Reset(DateTimeOffset recentStart)
    {
        _traders.Clear();
        _recentStart = recentStart;
        _unique = 0;
        _recent = 0;
        _newRecent = 0;
        _volume = 0;
    }

    /// <param name="trader">The signer; null is skipped.</param>
    /// <param name="ts">Trade time (non-decreasing across calls).</param>
    /// <param name="amount">
    /// Volume to attribute, or null when the trade is in another quote currency (it can't be summed,
    /// so it only tells whether the trader is returning). Unreported or non-positive amounts carry
    /// no volume.
    /// </param>
    public void Add(string? trader, DateTimeOffset ts, decimal? amount)
    {
        if (trader is null)
        {
            return;
        }

        ref Trader t = ref CollectionsMarshal.GetValueRefOrAddDefault(
            _traders, trader, out bool seenInWindow);

        if (ts > _recentStart && !t.Recent)
        {
            t.Recent = true;
            _recent++;
            bool isNew = holders.TryGetFirstSeen(trader, out DateTimeOffset firstSeen)
                ? firstSeen > _recentStart
                : !seenInWindow;
            if (isNew)
            {
                _newRecent++;
            }
        }

        if (amount is not { } a)
        {
            return;
        }

        if (!t.Counted)
        {
            t.Counted = true;
            _unique++;
        }

        if (a > 0)
        {
            t.Volume += a;
            _volume += a;
        }
    }

    /// <summary>
    /// The largest single-wallet share of the summed volume, and the Herfindahl-Hirschman index of
    /// the per-wallet shares (sum of squared shares; 1 = one wallet, 1/n = volume split evenly over
    /// n wallets), both in (0, 1], in one pass; both null without summed volume.
    /// </summary>
    public (double? TopShare, double? Hhi) Shares()
    {
        if (_volume <= 0)
        {
            return (null, null);
        }

        decimal max = 0;
        double hhi = 0;
        foreach (Trader t in _traders.Values)
        {
            if (t.Volume > max)
            {
                max = t.Volume;
            }

            double share = (double)(t.Volume / _volume);
            hhi += share * share;
        }

        return ((double)(max / _volume), hhi);
    }
}
