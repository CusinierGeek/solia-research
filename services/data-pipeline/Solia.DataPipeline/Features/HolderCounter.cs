using System.Numerics;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Unique-trader count (the free holders proxy) in bounded memory. Exact up to
/// <see cref="ExactLimit"/> wallets; past that it switches to a HyperLogLog sketch (4 KB,
/// ~1.6% standard error) that keeps counting instead of saturating, so a token with millions of
/// traders costs the same memory as one with a thousand.
/// </summary>
internal sealed class HolderCounter
{
    internal const int ExactLimit = 1024;

    private const int Precision = 12;
    private const int RegisterCount = 1 << Precision;

    private HashSet<string>? _exact = new(StringComparer.Ordinal);
    private byte[]? _registers;

    public int Count => _exact?.Count ?? Estimate();

    public void Add(string trader)
    {
        if (_exact is null)
        {
            AddToSketch(trader);
            return;
        }

        if (!_exact.Add(trader) || _exact.Count <= ExactLimit)
        {
            return;
        }

        _registers = new byte[RegisterCount];
        foreach (string known in _exact)
        {
            AddToSketch(known);
        }

        _exact = null;
    }

    private void AddToSketch(string trader)
    {
        ulong hash = Hash(trader);
        int register = (int)(hash >> (64 - Precision));
        ulong rest = hash << Precision;
        byte rank = (byte)(rest == 0 ? 64 - Precision + 1 : BitOperations.LeadingZeroCount(rest) + 1);
        if (rank > _registers![register])
        {
            _registers[register] = rank;
        }
    }

    private int Estimate()
    {
        double sum = 0;
        int zeros = 0;
        foreach (byte rank in _registers!)
        {
            sum += 1.0 / (1UL << rank);
            if (rank == 0)
            {
                zeros++;
            }
        }

        double alpha = 0.7213 / (1 + (1.079 / RegisterCount));
        double estimate = alpha * RegisterCount * RegisterCount / sum;

        // Small-range correction (linear counting) while registers are still empty.
        if (estimate <= 2.5 * RegisterCount && zeros > 0)
        {
            estimate = RegisterCount * Math.Log((double)RegisterCount / zeros);
        }

        return (int)Math.Round(estimate);
    }

    /// <summary>FNV-1a + splitmix64 finalizer: well-mixed and, unlike string.GetHashCode, stable across runs.</summary>
    private static ulong Hash(string value)
    {
        ulong h = 14695981039346656037UL;
        foreach (char c in value)
        {
            h ^= c;
            h *= 1099511628211UL;
        }

        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9UL;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebUL;
        h ^= h >> 31;
        return h;
    }
}
