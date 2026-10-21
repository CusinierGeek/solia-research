namespace Solia.Shared.Configuration;

/// <summary>
/// Parsing for the flat env-style keys from .env.example. Values are trimmed (a Windows-edited
/// .env can leave a trailing CR); a value that is set but invalid throws an error naming the
/// key instead of being silently replaced by the default.
/// </summary>
public static class FlatValues
{
    public static string Get(Func<string, string?> lookup, string key, string fallback)
    {
        string? v = lookup(key);
        return string.IsNullOrWhiteSpace(v) ? fallback : v.Trim();
    }

    public static int GetPort(Func<string, string?> lookup, string key, int fallback)
    {
        string? v = lookup(key);
        if (string.IsNullOrWhiteSpace(v))
        {
            return fallback;
        }
        if (int.TryParse(v.Trim(), out int port) && port is >= 1 and <= 65535)
        {
            return port;
        }
        throw new InvalidOperationException($"{key} must be a port number (1-65535), got '{v}'.");
    }

    /// <summary>Accepts true/false, 1/0, yes/no, on/off (case-insensitive); null if unset.</summary>
    public static bool? GetBool(Func<string, string?> lookup, string key)
    {
        string? v = lookup(key);
        if (string.IsNullOrWhiteSpace(v))
        {
            return null;
        }
        return v.Trim().ToLowerInvariant() switch
        {
            "true" or "1" or "yes" or "on" => true,
            "false" or "0" or "no" or "off" => false,
            _ => throw new InvalidOperationException(
                $"{key} must be a boolean (true/false, 1/0, yes/no, on/off), got '{v}'."),
        };
    }
}
