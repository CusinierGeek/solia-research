namespace Solia.Api.Endpoints;

/// <summary>Maps the public read/query surface (docs/API-CONTRACTS.md) onto the versioned group.</summary>
public static class ReadEndpoints
{
    public static RouteGroupBuilder MapReadEndpoints(this RouteGroupBuilder group) =>
        group.MapTokensEndpoints()
            .MapPredictionsEndpoints()
            .MapTradesEndpoints()
            .MapBotSessionsEndpoints()
            .MapStatsEndpoints()
            .MapLogsEndpoints();
}
