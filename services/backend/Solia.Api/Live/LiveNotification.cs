using System.Globalization;

namespace Solia.Api.Live;

/// <summary>One <c>solia_live</c> NOTIFY: which row event happened, and the id of the row to re-read.</summary>
public readonly record struct LiveNotification(string Event, long Id)
{
    /// <summary>
    /// Parses a trigger payload, <c>&lt;event&gt;:&lt;id&gt;</c>. Anything else — an unknown event, a
    /// non-positive or malformed id — is rejected rather than guessed at.
    /// </summary>
    public static bool TryParse(string? payload, out LiveNotification notification)
    {
        notification = default;
        int colon = payload?.LastIndexOf(':') ?? -1;
        if (colon <= 0)
        {
            return false;
        }

        string name = payload![..colon];
        string? known = Array.Find(LiveEvents.RowEvents, e => e == name);
        if (known is null
            || !long.TryParse(payload.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out long id)
            || id <= 0)
        {
            return false;
        }

        notification = new LiveNotification(known, id);
        return true;
    }
}
