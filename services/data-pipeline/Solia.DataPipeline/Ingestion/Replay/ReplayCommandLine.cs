using System.Globalization;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Parsed arguments of the <c>replay</c> command:
/// <c>replay --from &lt;utc&gt; --to &lt;utc&gt; [--mirror &lt;dir&gt;] [--mirror-only] [--rebuild] [config overrides]</c>.
/// <list type="bullet">
/// <item><c>--from</c> / <c>--to</c>: the half-open UTC window <c>[from, to)</c>, ISO 8601 (e.g. <c>2026-05-01T00:00Z</c>,
/// or a bare date = midnight UTC). Must be ordered, start at or after the archive start, and end no later than the
/// start of the current hour (an hour's archive is only published once the hour is over).</item>
/// <item><c>--mirror &lt;dir&gt;</c> (opt-in): read hours from a local archive mirror first, download the absent ones into it.</item>
/// <item><c>--mirror-only</c>: only fill the mirror (requires <c>--mirror</c>); no replay, no database.</item>
/// <item><c>--rebuild</c>: overwrite <c>features</c> rows already stored for the same (mint, ts) instead of keeping them
/// (only for tokens whose whole history is inside the window, see <c>PostgresTokenSink</c>).</item>
/// <item>Configuration overrides, handed to the host as for the live service: a key path
/// (<c>--Pipeline:SnapshotCadenceSeconds=5</c> or <c>--Pipeline:SnapshotCadenceSeconds 5</c>) or <c>--environment &lt;name&gt;</c>.</item>
/// </list>
/// </summary>
public sealed record ReplayCommandLine(
    ReplayWindow Window, string? MirrorDirectory, bool MirrorOnly, bool Rebuild, IReadOnlyList<string> ConfigArgs)
{
    public const string Usage =
        "usage: replay --from <utc> --to <utc> [--mirror <dir>] [--mirror-only] [--rebuild] [--Section:Key=value ...]\n" +
        "  --from, --to    half-open UTC window [from, to), ISO 8601 (2026-05-01T00:00Z or 2026-05-01);\n" +
        "                  must end at or before the start of the current hour\n" +
        "  --mirror <dir>  opt-in local archive mirror: read hours from <dir>, download absent ones into it\n" +
        "  --mirror-only   only download the window into the mirror (needs --mirror); no replay\n" +
        "  --rebuild       refresh features rows already stored for the window (default keeps them)\n" +
        "  --Section:Key=value, --environment <name>   configuration overrides, as for the live service";

    /// <summary>
    /// Parses and validates <paramref name="args"/> (the arguments after <c>replay</c>) against
    /// <paramref name="nowUtc"/>. Returns null with an error message on any invalid input. Only
    /// the window and flags are checked here; the mirror directory is checked by <see cref="ArchiveMirror"/>.
    /// </summary>
    public static ReplayCommandLine? TryParse(IReadOnlyList<string> args, DateTimeOffset nowUtc, out string? error)
    {
        DateTimeOffset? from = null, to = null;
        string? mirror = null;
        bool mirrorOnly = false, rebuild = false;
        var configArgs = new List<string>();

        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "--from" or "--to" or "--mirror":
                    if (i + 1 >= args.Count || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                    {
                        error = $"{arg} needs a value.";
                        return null;
                    }

                    string value = args[++i];
                    if (arg == "--mirror")
                    {
                        mirror = value;
                        break;
                    }

                    if (!TryParseUtc(value, out DateTimeOffset parsed))
                    {
                        error = $"{arg} '{value}' is not an ISO 8601 date/time.";
                        return null;
                    }

                    if (arg == "--from")
                    {
                        from = parsed;
                    }
                    else
                    {
                        to = parsed;
                    }

                    break;
                case "--mirror-only":
                    mirrorOnly = true;
                    break;
                case "--rebuild":
                    rebuild = true;
                    break;
                case not null when IsConfigOverride(arg):
                    configArgs.Add(arg);
                    if (!arg.Contains('=', StringComparison.Ordinal))
                    {
                        if (i + 1 >= args.Count || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                        {
                            error = $"{arg} needs a value.";
                            return null;
                        }

                        configArgs.Add(args[++i]);
                    }

                    break;
                default:
                    error = $"unknown argument '{arg}'.";
                    return null;
            }
        }

        if (from is null || to is null)
        {
            error = "--from and --to are required.";
            return null;
        }

        if (mirrorOnly && mirror is null)
        {
            error = "--mirror-only needs --mirror <dir>.";
            return null;
        }

        if (mirrorOnly && rebuild)
        {
            error = "--rebuild has no effect with --mirror-only (nothing is replayed).";
            return null;
        }

        var window = new ReplayWindow(from.Value, to.Value);
        try
        {
            window.EnsureValid();
        }
        catch (ArgumentException ex)
        {
            error = ex.Message;
            return null;
        }

        // The current hour's archive isn't published until the hour is over: a window reaching into it would
        // record that hour as "missing" (a complete run) and silently never ingest it.
        DateTimeOffset latestEnd = ReplayWindow.FloorToHour(nowUtc);
        if (window.EndUtc > latestEnd)
        {
            error = $"Replay window ends at {window.EndUtc:O}, after {latestEnd:O} (now {nowUtc:O}); archives only " +
                    "exist for hours that are over, so end the window at or before the start of the current hour.";
            return null;
        }

        error = null;
        return new ReplayCommandLine(window, mirror, mirrorOnly, rebuild, configArgs);
    }

    /// <summary>A configuration key path (<c>--Section:Key[=value]</c>) or the host's <c>--environment</c>.</summary>
    private static bool IsConfigOverride(string arg)
    {
        string key = arg.Split('=', 2)[0];
        return key.StartsWith("--", StringComparison.Ordinal) &&
               (key.Contains(':', StringComparison.Ordinal) || key.Equals("--environment", StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>ISO 8601; a value without an offset is taken as UTC.</summary>
    private static bool TryParseUtc(string value, out DateTimeOffset result) =>
        DateTimeOffset.TryParse(value, CultureInfo.InvariantCulture,
            DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out result);
}
