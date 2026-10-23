using Solia.Core.Http;

namespace Solia.Api.Endpoints;

/// <summary>
/// Shared query-string checks for the read endpoints. Each failure is an
/// <see cref="InvalidRequestException"/>, so it reaches the caller as a 400 <c>invalid_request</c> envelope
/// naming the offending parameter. (A value that can't even bind — <c>?page=abc</c> — is a framework 400,
/// wrapped in the same envelope by the error-shape middleware.)
/// </summary>
public static class QueryValidation
{
    /// <summary>
    /// <paramref name="value"/> lower-cased when it is one of <paramref name="allowed"/> (case-insensitive),
    /// null when absent — missing, empty or blank (<c>?status=</c>), so the endpoint's default applies.
    /// </summary>
    public static string? OneOf(string? value, string name, IReadOnlyCollection<string> allowed)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string normalized = value.Trim().ToLowerInvariant();
        return allowed.Contains(normalized)
            ? normalized
            : throw new InvalidRequestException($"{name} must be one of: {string.Join(", ", allowed)}.");
    }

    /// <summary>Rejects an inverted range (<c>from</c> after <c>to</c>); either bound may be open.</summary>
    public static void Range(DateTimeOffset? from, DateTimeOffset? to)
    {
        if (from is { } f && to is { } t && f > t)
        {
            throw new InvalidRequestException("from must not be after to.");
        }
    }

    /// <summary><paramref name="value"/> in [1, <paramref name="max"/>], or <paramref name="defaultValue"/> when absent.</summary>
    public static int Limit(int? value, int defaultValue, int max, string name = "limit")
    {
        int limit = value ?? defaultValue;
        return limit is >= 1 && limit <= max
            ? limit
            : throw new InvalidRequestException($"{name} must be between 1 and {max}.");
    }
}
