using Microsoft.Extensions.Configuration;

namespace Solia.Shared.Configuration;

/// <summary>One flat .env-style key (e.g. ENRICH_HOLDERS) and the options-section path it maps onto.</summary>
/// <param name="IsBool">Parse the value with <see cref="FlatValues.GetBool"/> (true/1/yes/on…) before binding.</param>
public readonly record struct FlatAlias(string FlatKey, string Path, bool IsBool = false);

/// <summary>
/// Maps flat keys from .env.example onto their options-section paths, so options are bound by the
/// normal configuration binder. Shared by every service that exposes flat env keys; each service only
/// declares its own alias table.
/// </summary>
public static class FlatAliases
{
    /// <summary>
    /// Adds the aliased values as a configuration source on top of the sources registered so far, so a
    /// flat key wins over appsettings. Values are trimmed (a Windows-edited .env can leave a trailing CR);
    /// unset keys are skipped. Throws, naming the key, if a flat bool is set but invalid; other values
    /// are left to the binder + options validation, which name the offending option.
    /// </summary>
    public static IConfigurationBuilder AddFlatAliases(
        this IConfigurationManager configuration, IEnumerable<FlatAlias> aliases) =>
        configuration.AddInMemoryCollection(Resolve(key => configuration[key], aliases));

    /// <summary>
    /// The aliased values currently visible through <paramref name="lookup"/>, keyed by their section path
    /// (same trimming / bool parsing / skipping rules as <see cref="AddFlatAliases"/>). Use this to apply the
    /// aliases lazily — e.g. when options are resolved — so configuration sources added later still count.
    /// </summary>
    public static Dictionary<string, string?> Resolve(Func<string, string?> lookup, IEnumerable<FlatAlias> aliases)
    {
        var values = new Dictionary<string, string?>();
        foreach ((string flatKey, string path, bool isBool) in aliases)
        {
            string? raw = lookup(flatKey);
            if (string.IsNullOrWhiteSpace(raw))
            {
                continue;
            }
            values[path] = isBool
                ? FlatValues.GetBool(lookup, flatKey)!.Value.ToString()
                : raw.Trim();
        }
        return values;
    }
}
