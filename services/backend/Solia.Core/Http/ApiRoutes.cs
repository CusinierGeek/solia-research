namespace Solia.Core.Http;

/// <summary>Shared route constants so the host and tests agree on the public surface.</summary>
public static class ApiRoutes
{
    /// <summary>Versioned base path for all public endpoints (docs/API-CONTRACTS.md).</summary>
    public const string BasePath = "/api/v1";

    /// <summary>Every health check (DB + ai-service); 503 when one is Unhealthy.</summary>
    public const string Health = "/health";

    /// <summary>Readiness probe: the DB and ai-service checks.</summary>
    public const string HealthReady = "/health/ready";

    /// <summary>Liveness probe: no dependency is checked.</summary>
    public const string HealthLive = "/health/live";
}
