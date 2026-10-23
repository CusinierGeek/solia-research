namespace Solia.Api.Health;

/// <summary>
/// Logs a check's failure once, when it starts failing (Warning, with the cause), and its recovery once
/// (Information) — not on every probe. Orchestrators poll the probes every few seconds, and each logged
/// Warning is mirrored into the `logs` table, so a dependency that stays down must not write a row per poll.
/// </summary>
internal sealed class HealthTransitionLog(ILogger<HealthTransitionLog> logger)
{
    private readonly Dictionary<string, bool> _healthy = [];

    public void Healthy(string check)
    {
        if (Swap(check, healthy: true) == false)
        {
            logger.LogInformation("Health check {HealthCheck} recovered", check);
        }
    }

    public void Unhealthy(string check, string description, Exception? cause)
    {
        if (Swap(check, healthy: false) != false)
        {
            logger.LogWarning(cause, "Health check {HealthCheck} is failing: {Description}", check, description);
        }
    }

    /// <summary>Records the new state; returns the previous one (null before the first check).</summary>
    private bool? Swap(string check, bool healthy)
    {
        lock (_healthy)
        {
            bool? previous = _healthy.TryGetValue(check, out bool was) ? was : null;
            _healthy[check] = healthy;
            return previous;
        }
    }
}
