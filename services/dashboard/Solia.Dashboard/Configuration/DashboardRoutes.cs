namespace Solia.Dashboard.Configuration;

/// <summary>Host endpoints outside the Blazor views (which live in <see cref="Navigation.DashboardNav"/>).</summary>
public static class DashboardRoutes
{
    /// <summary>Dependency-free liveness endpoint, probed by the container HEALTHCHECK (<c>--healthcheck</c>).</summary>
    public const string HealthLive = "/health/live";
}
