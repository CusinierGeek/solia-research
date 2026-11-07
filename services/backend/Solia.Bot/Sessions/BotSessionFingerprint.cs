using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Solia.Bot.Configuration;

namespace Solia.Bot.Sessions;

/// <summary>
/// One piece of the bot's regime that lives outside <see cref="BotOptions"/> — e.g. the horizon the host trades on.
/// Registered as singletons; every one registered is part of the <see cref="BotSessionFingerprint"/>, except one
/// with a null <paramref name="Value"/>: a feature that is off doesn't move the fingerprint of a running session.
/// </summary>
public sealed record BotRegimePart(string Name, string? Value);

/// <summary>
/// What identifies a bot regime: the effective options, normalized so that two configurations meaning the same
/// thing compare equal (property order, venue-key case, <c>10</c> vs <c>10.0</c>), and the champion serving it.
/// Two bot runs with the same <see cref="Hash"/> belong to the same session. <see cref="BotOptions.SessionLabel"/>
/// names a session, it isn't part of its regime, so it is left out.
/// </summary>
public static class BotSessionFingerprint
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    /// <summary>The normalized JSON of <paramref name="options"/> plus <paramref name="parts"/> (each under its name).</summary>
    public static string OptionsJson(BotOptions options, IEnumerable<BotRegimePart>? parts = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        var root = (JsonObject)JsonSerializer.SerializeToNode(options, Json)!;
        root.Remove(JsonNamingPolicy.CamelCase.ConvertName(nameof(BotOptions.SessionLabel)));
        foreach (BotRegimePart part in parts ?? [])
        {
            if (part.Value is not null)
            {
                root[part.Name] = part.Value;
            }
        }

        return Normalize(root, lowerKeys: false)!.ToJsonString();
    }

    /// <summary>SHA-256 (lower-case hex) of the normalized options and the champion's version (none = empty).</summary>
    public static string Hash(string optionsJson, string? championVersion)
    {
        ArgumentNullException.ThrowIfNull(optionsJson);
        byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes($"{optionsJson}\n{championVersion}"));
        return Convert.ToHexStringLower(digest);
    }

    // Objects get their keys sorted; nested objects are the per-venue tables, whose keys are case-insensitive, so
    // they are lower-cased too. A number keeps its value, not its spelling.
    private static JsonNode? Normalize(JsonNode? node, bool lowerKeys) => node switch
    {
        JsonObject obj => new JsonObject(obj
            .Select(kv => KeyValuePair.Create(lowerKeys ? kv.Key.ToLowerInvariant() : kv.Key, Normalize(kv.Value, lowerKeys: true)))
            .OrderBy(kv => kv.Key, StringComparer.Ordinal)),
        JsonArray array => new JsonArray(array.Select(n => Normalize(n, lowerKeys)).ToArray()),
        // Dividing by 1.000… drops a decimal's trailing zeros.
        JsonValue value when value.GetValueKind() == JsonValueKind.Number
            && decimal.TryParse(value.ToJsonString(), NumberStyles.Float, CultureInfo.InvariantCulture, out decimal number) =>
            JsonValue.Create(number / 1.000000000000000000000000000000000m),
        _ => node?.DeepClone(),
    };
}
