namespace Solia.Api.Health;

/// <summary>
/// The health JSON (the de-facto standard health-report shape, snake_case on the wire):
/// <c>{ "status": "Healthy", "total_duration": "00:00:00.012", "entries": { "database": { ... } } }</c>.
/// Entries carry a fixed description only — never an exception message, which can quote connection details.
/// </summary>
public sealed record HealthResponse(string Status, TimeSpan TotalDuration, IReadOnlyDictionary<string, HealthEntryDto> Entries);

/// <summary>One check's outcome in a <see cref="HealthResponse"/>.</summary>
public sealed record HealthEntryDto(string Status, string? Description, TimeSpan Duration, IReadOnlyList<string> Tags);

/// <summary>
/// The only descriptions a health response may carry. <see cref="HealthEndpoints"/> replaces any other with
/// <see cref="CheckFailed"/>: the health-check service fills an entry's description with the exception message
/// when a check throws or times out, and that text must not reach unauthenticated callers.
/// </summary>
internal static class HealthDescriptions
{
    public const string DatabaseReachable = "database reachable";
    public const string DatabaseUnreachable = "database unreachable";
    public const string AiServiceReachable = "ai-service reachable";
    public const string AiServiceUnreachable = "ai-service unreachable";
    public const string AiServiceNotHealthy = "ai-service reports it is not healthy";
    public const string CheckFailed = "check failed";

    private static readonly HashSet<string> Known =
    [
        DatabaseReachable, DatabaseUnreachable, AiServiceReachable, AiServiceUnreachable, AiServiceNotHealthy, CheckFailed,
    ];

    /// <summary>The description if it is one of the fixed ones; otherwise <see cref="CheckFailed"/> (or null for none).</summary>
    public static string? Sanitize(string? description) =>
        description is null || Known.Contains(description) ? description : CheckFailed;
}
