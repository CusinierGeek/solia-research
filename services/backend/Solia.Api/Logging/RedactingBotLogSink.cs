using System.Text.Json.Nodes;
using Solia.Bot.Logging;
using Solia.Secrets;
using Solia.Shared.Logging;

namespace Solia.Api.Logging;

/// <summary>
/// Scrubs the bot's operational `logs` rows. The bot writes them straight to its sink (not through
/// <c>ILogger</c>), so <see cref="RedactingLoggerFactory"/> never sees them: an <c>ex.Message</c> in a
/// "bot failed" context would otherwise reach the table verbatim. The message and every string in the
/// context are redacted value by value — on the JSON tree, not its text, so JSON escaping can't hide a match.
/// </summary>
internal sealed class RedactingBotLogSink(IBotLogSink inner, SecretRedactor redactor) : IBotLogSink
{
    public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default)
    {
        if (redactor.IsEmpty)
        {
            return inner.WriteAsync(level, message, context, ct);
        }
        return inner.WriteAsync(level, redactor.Redact(message), RedactContext(context), ct);
    }

    private object? RedactContext(object? context)
    {
        if (context is null)
        {
            return null;
        }

        // Same serialisation (camelCase keys) the sink stores, so the scrubbed tree keeps the row's shape.
        JsonNode? tree = JsonNode.Parse(PostgresLogSink.SerializeContext(context)!);
        return Scrub(tree) ? tree : context;
    }

    /// <summary>Redacts string leaves in place; true when anything changed.</summary>
    private bool Scrub(JsonNode? node)
    {
        switch (node)
        {
            case JsonObject obj:
                bool changed = false;
                foreach (KeyValuePair<string, JsonNode?> property in obj.ToList())
                {
                    if (property.Value is JsonValue leaf && TryRedact(leaf, out string? redacted))
                    {
                        obj[property.Key] = redacted;
                        changed = true;
                    }
                    else
                    {
                        changed |= Scrub(property.Value);
                    }
                }
                return changed;
            case JsonArray array:
                bool any = false;
                for (int i = 0; i < array.Count; i++)
                {
                    if (array[i] is JsonValue leaf && TryRedact(leaf, out string? redacted))
                    {
                        array[i] = redacted;
                        any = true;
                    }
                    else
                    {
                        any |= Scrub(array[i]);
                    }
                }
                return any;
            default:
                return false;
        }
    }

    private bool TryRedact(JsonValue leaf, out string? redacted)
    {
        redacted = leaf.TryGetValue(out string? text) && redactor.ContainsSecret(text) ? redactor.Redact(text) : null;
        return redacted is not null;
    }
}
