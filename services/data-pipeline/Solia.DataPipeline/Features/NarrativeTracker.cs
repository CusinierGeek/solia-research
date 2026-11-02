using System.Globalization;
using System.Text;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Features;

/// <summary>
/// Normalizes a token's name and ticker into the keys narratives are matched on (FE-T12): compatibility-decomposed
/// (full-width and styled letters fold to plain ones), diacritics, punctuation, spaces and emoji dropped, lower-cased.
/// "$PEPE", "Pépé" and "pepe 🐸" all give <c>pepe</c>. Keys shorter than <see cref="MinLength"/> are dropped:
/// one-letter tickers would match unrelated launches.
/// </summary>
public static class NarrativeKeys
{
    public const int MinLength = 2;

    /// <summary>The normalized key of <paramref name="text"/>, or null when nothing usable is left.</summary>
    public static string? Normalize(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return null;
        }

        string decomposed = text.Normalize(NormalizationForm.FormKD);
        var key = new StringBuilder(decomposed.Length);
        foreach (char c in decomposed)
        {
            if (CharUnicodeInfo.GetUnicodeCategory(c) is UnicodeCategory.NonSpacingMark
                or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark)
            {
                continue;
            }

            if (char.IsLetterOrDigit(c))
            {
                key.Append(char.ToLowerInvariant(c));
            }
        }

        return key.Length >= MinLength ? key.ToString() : null;
    }

    /// <summary>The distinct keys of a name and a ticker (zero, one or two).</summary>
    public static string[] Of(string? name, string? symbol)
    {
        string? n = Normalize(name);
        string? s = Normalize(symbol);
        return (n, s) switch
        {
            (null, null) => [],
            (null, _) => [s!],
            (_, null) => [n],
            _ when n == s => [n],
            _ => [n, s!],
        };
    }
}

/// <summary>
/// Narrative heat (FE-T12, issue #196), derived for free from the stream: how many other launches in the last hour
/// share a token's normalized name or ticker (<see cref="NarrativeKeys"/>), and how many tokens sharing it graduated
/// recently. A launch matches when any of its keys equals any of the token's; the token itself never counts.
/// <para>
/// Point-in-time: a snapshot at <c>ts</c> counts only events of whole seconds strictly before the second of
/// <c>ts</c> (like <see cref="MarketContext"/>), so the counts depend on event timestamps only and live and replay
/// agree. A graduation is attributed to a narrative only when the graduating token's create is at most
/// <see cref="CreationMemory"/> older than the graduation, a bound checked on timestamps (never on when memory was
/// pruned), so any history of at least <see cref="FullCoverage"/> gives the same counts. The counts stay null until
/// that much history has been observed (replay warm-up; a live restart).
/// </para>
/// Not thread-safe: fed and queried from the pipeline's single loop.
/// </summary>
public sealed class NarrativeTracker
{
    public static readonly TimeSpan MatchWindow = TimeSpan.FromHours(1);
    public static readonly TimeSpan GraduationWindow = TimeSpan.FromHours(3);
    public static readonly TimeSpan CreationMemory = TimeSpan.FromHours(3);

    /// <summary>History needed before every count is defined: a graduation at the far end of its window, of a token created <see cref="CreationMemory"/> earlier.</summary>
    public static readonly TimeSpan FullCoverage = CreationMemory + GraduationWindow;

    private static readonly long MatchSec = (long)MatchWindow.TotalSeconds;
    private static readonly long GraduationSec = (long)GraduationWindow.TotalSeconds;
    private static readonly long CreationSec = (long)CreationMemory.TotalSeconds;
    private static readonly long FullSec = (long)FullCoverage.TotalSeconds;
    private static readonly long SkewSec = (long)TokenWindowStore.MaxFutureSkew.TotalSeconds;
    private static readonly long HorizonSec = Math.Max(MatchSec, Math.Max(GraduationSec, CreationSec));

    private readonly EventClock _clock = new(TokenWindowStore.MaxFutureSkew, PipelineEngine.ClockJumpConfirmations);
    private readonly Dictionary<string, List<Entry>> _launches = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<Entry>> _graduations = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Creation> _creations = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _graduated = new(StringComparer.Ordinal);
    private long? _firstSecond;

    /// <summary>Creates remembered (bounded by <see cref="CreationMemory"/>).</summary>
    public int RememberedCreations => _creations.Count;

    public void Apply(RawEvent e)
    {
        long now = Seconds(_clock.Observe(e.Timestamp));
        long sec = Seconds(e.Timestamp);
        if (sec > now + SkewSec || sec <= now - HorizonSec - SkewSec)
        {
            return; // implausibly far ahead (unconfirmed jump) or older than any window
        }

        _firstSecond = _firstSecond is { } first && first <= sec ? first : sec;
        if (e.Kind == EventKind.Create)
        {
            string[] keys = NarrativeKeys.Of(e.Name, e.Symbol);
            if (keys.Length > 0 && _creations.TryAdd(e.Mint, new Creation(keys, sec)))
            {
                Add(_launches, keys, new Entry(sec, e.Mint));
            }
        }
        else if (e.Kind == EventKind.Migrate
                 && _creations.TryGetValue(e.Mint, out Creation created)
                 && sec - created.Second <= CreationSec
                 && _graduated.TryAdd(e.Mint, sec))
        {
            Add(_graduations, created.Keys, new Entry(sec, e.Mint));
        }
    }

    /// <summary>
    /// The narrative counts of a token with <paramref name="keys"/> for a snapshot at <paramref name="ts"/>, and the
    /// coverage behind them.
    /// </summary>
    public (int CoverageSec, int? Matches, int? Graduations) Count(string[] keys, string mint, DateTimeOffset ts)
    {
        long to = Seconds(ts);
        int coverage = (int)(_firstSecond is { } first ? Math.Clamp(to - first, 0, FullSec) : 0);
        if (keys.Length == 0)
        {
            return (coverage, null, null);
        }

        int? matches = coverage >= MatchSec ? CountDistinct(_launches, keys, mint, to - MatchSec, to) : null;
        int? graduations = coverage >= FullSec ? CountDistinct(_graduations, keys, mint, to - GraduationSec, to) : null;
        return (coverage, matches, graduations);
    }

    /// <summary>Forgets what no window can reach any more (with slack for out-of-order events). Housekeeping only: never changes a count.</summary>
    public void Prune(DateTimeOffset ts)
    {
        long to = Seconds(ts);
        Prune(_launches, to - MatchSec - (2 * SkewSec));
        Prune(_graduations, to - GraduationSec - (2 * SkewSec));
        RemoveWhere(_creations, c => c.Second < to - CreationSec - (2 * SkewSec));
        RemoveWhere(_graduated, g => g < to - GraduationSec - (2 * SkewSec));
    }

    private static int CountDistinct(Dictionary<string, List<Entry>> index, string[] keys, string mint, long from, long to)
    {
        HashSet<string>? seen = keys.Length > 1 ? new HashSet<string>(StringComparer.Ordinal) : null;
        int count = 0;
        foreach (string key in keys)
        {
            if (!index.TryGetValue(key, out List<Entry>? entries))
            {
                continue;
            }

            foreach (Entry entry in entries)
            {
                if (entry.Second >= from && entry.Second < to && entry.Mint != mint && (seen?.Add(entry.Mint) ?? true))
                {
                    count++;
                }
            }
        }

        return count;
    }

    private static void Add(Dictionary<string, List<Entry>> index, string[] keys, Entry entry)
    {
        foreach (string key in keys)
        {
            if (!index.TryGetValue(key, out List<Entry>? entries))
            {
                index[key] = entries = [];
            }

            entries.Add(entry);
        }
    }

    private static void Prune(Dictionary<string, List<Entry>> index, long before)
    {
        List<string>? empty = null;
        foreach ((string key, List<Entry> entries) in index)
        {
            entries.RemoveAll(e => e.Second < before);
            if (entries.Count == 0)
            {
                (empty ??= []).Add(key);
            }
        }

        empty?.ForEach(key => index.Remove(key));
    }

    private static void RemoveWhere<T>(Dictionary<string, T> map, Func<T, bool> stale)
    {
        foreach (string key in map.Where(kv => stale(kv.Value)).Select(kv => kv.Key).ToList())
        {
            map.Remove(key);
        }
    }

    private static long Seconds(DateTimeOffset ts) => ts.ToUnixTimeSeconds();

    private readonly record struct Entry(long Second, string Mint);

    private readonly record struct Creation(string[] Keys, long Second);
}
