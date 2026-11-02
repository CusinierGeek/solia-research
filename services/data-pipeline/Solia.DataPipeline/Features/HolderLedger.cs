namespace Solia.DataPipeline.Features;

/// <summary>
/// Per-token net token position of every wallet seen on the token (FE-T13, issue #197), derived from the stream only:
/// the create's dev buy, buys − sells, liquidity removed − added, and token transfers in − out. Lives in the token's
/// <see cref="TokenState"/>, so it is dropped with it when the token is evicted.
/// <para>
/// Pool accounts (every pool id the token reported) and the mint itself are excluded: a position they held through a
/// transfer is dropped, and later movements to or from them are ignored, so the ledger only holds wallets.
/// Circulating supply is the sum of the positive positions (a negative one — a wallet that sold or sent tokens it got
/// before the history started — counts as 0).
/// </para>
/// <para>
/// Memory bound: at most <c>maxWallets</c> positions. Past it, the empty or negative ones are dropped first, then the
/// smallest holders; their tokens stay in the circulating total (<see cref="Truncated"/> is set), and if such a
/// wallet comes back it restarts from its later movements.
/// </para>
/// </summary>
internal sealed class HolderLedger
{
    internal const int TopK = 10;
    internal const int DefaultMaxWallets = 4_096;

    private readonly record struct Flow(DateTimeOffset Ts, string Wallet, decimal Delta);

    private readonly int _maxWallets;
    private readonly Dictionary<string, decimal> _positions = new(StringComparer.Ordinal);
    private readonly HashSet<string> _excluded = new(StringComparer.Ordinal);
    private readonly List<Flow> _flows = []; // sorted by Ts; pruned to the window
    private decimal _circulating; // Σ max(position, 0), plus the tokens of evicted holders

    public HolderLedger(int maxWallets = DefaultMaxWallets)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(maxWallets, TopK);
        _maxWallets = maxWallets;
    }

    /// <summary>Tracked positions (including empty and negative ones).</summary>
    public int Count => _positions.Count;

    /// <summary>True once holders were evicted to respect the memory bound.</summary>
    public bool Truncated { get; private set; }

    public decimal Circulating => _circulating;

    public decimal PositionOf(string wallet) => _positions.GetValueOrDefault(wallet);

    /// <summary>Marks a pool (or other non-wallet) account: its position, if any, leaves the ledger and it is never tracked again.</summary>
    public void Exclude(string account)
    {
        if (!_excluded.Add(account))
        {
            return;
        }

        if (_positions.Remove(account, out decimal position))
        {
            _circulating -= Math.Max(position, 0);
        }

        _flows.RemoveAll(f => f.Wallet == account);
    }

    /// <summary>Moves <paramref name="delta"/> tokens into (positive) or out of (negative) <paramref name="wallet"/>'s position at <paramref name="ts"/>.</summary>
    public void Add(string wallet, decimal delta, DateTimeOffset ts)
    {
        if (delta == 0 || _excluded.Contains(wallet))
        {
            return;
        }

        decimal before = _positions.GetValueOrDefault(wallet);
        decimal after = before + delta;
        _positions[wallet] = after;
        _circulating += Math.Max(after, 0) - Math.Max(before, 0);

        int i = _flows.Count;
        while (i > 0 && _flows[i - 1].Ts > ts)
        {
            i--;
        }

        _flows.Insert(i, new Flow(ts, wallet, delta));

        if (_positions.Count > _maxWallets)
        {
            Evict();
        }
    }

    /// <summary>Drops the flows dated before <paramref name="cutoff"/> (the window start); positions are kept.</summary>
    public void Prune(DateTimeOffset cutoff)
    {
        int stale = 0;
        while (stale < _flows.Count && _flows[stale].Ts < cutoff)
        {
            stale++;
        }

        _flows.RemoveRange(0, stale);
    }

    /// <param name="totalSupply">The token's total supply, for <see cref="HolderSupplyFeatures.LargeHolders"/>; null when unknown.</param>
    /// <param name="largeShare">A wallet holding more than this share of <paramref name="totalSupply"/> is a large holder.</param>
    /// <param name="incomplete">Whether the token's history is known to be partial (its create wasn't seen).</param>
    public HolderSupplyFeatures Snapshot(decimal? totalSupply, double largeShare, bool incomplete)
    {
        // The TopK largest positive positions, largest first; ties broken by wallet so live and replay agree.
        var top = new List<(string Wallet, decimal Position)>(TopK + 1);
        int holders = 0;
        int large = 0;
        decimal? largeFloor = totalSupply is > 0 ? totalSupply.Value * (decimal)largeShare : null;
        foreach ((string wallet, decimal position) in _positions)
        {
            if (position <= 0)
            {
                continue;
            }

            holders++;
            if (largeFloor is { } floor && position > floor)
            {
                large++;
            }

            if (top.Count == TopK && !Ranks(wallet, position, top[^1]))
            {
                continue;
            }

            int i = top.Count;
            while (i > 0 && Ranks(wallet, position, top[i - 1]))
            {
                i--;
            }

            top.Insert(i, (wallet, position));
            if (top.Count > TopK)
            {
                top.RemoveAt(TopK);
            }
        }

        var features = new HolderSupplyFeatures
        {
            Holders = holders,
            LargeHolders = largeFloor is null ? null : large,
            Incomplete = incomplete,
        };
        if (_circulating <= 0 || top.Count == 0)
        {
            return features;
        }

        var topWallets = new HashSet<string>(StringComparer.Ordinal);
        decimal top10 = 0;
        foreach ((string wallet, decimal position) in top)
        {
            topWallets.Add(wallet);
            top10 += position;
        }

        decimal flow = 0;
        foreach (Flow f in _flows)
        {
            if (topWallets.Contains(f.Wallet))
            {
                flow += f.Delta;
            }
        }

        return features with
        {
            Top1Share = Share(top[0].Position),
            Top10Share = Share(top10),
            Top10NetFlow = (double)(flow / _circulating),
        };
    }

    private double Share(decimal tokens) => Math.Min(1.0, (double)(tokens / _circulating));

    /// <summary>Whether (<paramref name="wallet"/>, <paramref name="position"/>) ranks above <paramref name="other"/>.</summary>
    private static bool Ranks(string wallet, decimal position, (string Wallet, decimal Position) other) =>
        position > other.Position || (position == other.Position && string.CompareOrdinal(wallet, other.Wallet) < 0);

    /// <summary>Back to 3/4 of the bound: empty / negative positions first, then the smallest holders (their tokens stay circulating).</summary>
    private void Evict()
    {
        int target = _maxWallets * 3 / 4;
        foreach (string wallet in _positions.Where(p => p.Value <= 0).Select(p => p.Key).ToList())
        {
            _positions.Remove(wallet);
        }

        if (_positions.Count > target)
        {
            Truncated = true;
            foreach (string wallet in _positions
                         .OrderBy(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal)
                         .Take(_positions.Count - target).Select(p => p.Key).ToList())
            {
                _positions.Remove(wallet);
            }
        }
    }
}
