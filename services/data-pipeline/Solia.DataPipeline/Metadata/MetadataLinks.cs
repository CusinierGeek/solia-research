using System.Text.Json;

namespace Solia.DataPipeline.Metadata;

/// <summary>Which social links a token's metadata JSON lists (FE-T12).</summary>
public readonly record struct MetadataLinks(bool HasWebsite, bool HasTwitter, bool HasTelegram)
{
    private static readonly string[] WebsiteFields = ["website"];
    private static readonly string[] TwitterFields = ["twitter", "x"];
    private static readonly string[] TelegramFields = ["telegram"];

    /// <summary>
    /// Reads the links of a metadata JSON document: pump.fun's top-level <c>website</c> / <c>twitter</c> /
    /// <c>telegram</c>, or the same under Metaplex's <c>extensions</c>. A link counts when it is a non-blank string.
    /// Null when the document isn't a JSON object (malformed, HTML error page, array...).
    /// </summary>
    public static MetadataLinks? Parse(ReadOnlyMemory<byte> utf8Json)
    {
        try
        {
            using JsonDocument doc = JsonDocument.Parse(utf8Json);
            JsonElement root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            JsonElement? extensions = root.TryGetProperty("extensions", out JsonElement ext) && ext.ValueKind == JsonValueKind.Object
                ? ext
                : null;
            return new MetadataLinks(
                Has(root, extensions, WebsiteFields),
                Has(root, extensions, TwitterFields),
                Has(root, extensions, TelegramFields));
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static bool Has(JsonElement root, JsonElement? extensions, string[] fields) =>
        fields.Any(f => IsLink(root, f) || (extensions is { } ext && IsLink(ext, f)));

    private static bool IsLink(JsonElement obj, string field) =>
        obj.TryGetProperty(field, out JsonElement value)
        && value.ValueKind == JsonValueKind.String
        && !string.IsNullOrWhiteSpace(value.GetString());
}
