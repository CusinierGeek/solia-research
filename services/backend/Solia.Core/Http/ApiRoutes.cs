namespace Solia.Core.Http;

/// <summary>Shared route constants so the host and tests agree on the public surface.</summary>
public static class ApiRoutes
{
    /// <summary>Versioned base path for all public endpoints (docs/API-CONTRACTS.md). Endpoints land in T4/T5.</summary>
    public const string BasePath = "/api/v1";

    /// <summary>Liveness probe; real DB/ai-service readiness checks land in T8.</summary>
    public const string Health = "/health";
}
