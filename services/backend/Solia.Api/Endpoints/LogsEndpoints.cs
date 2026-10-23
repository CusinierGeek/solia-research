using Microsoft.EntityFrameworkCore;
using Solia.Api.OpenApi;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary><c>GET /logs/bot</c> — recent bot/system logs from the <c>logs</c> table.</summary>
public static class LogsEndpoints
{
    public const int DefaultLimit = 100;

    public const int MaxLimit = 1000;

    /// <summary>The <c>logs.level</c> values (CHECK in SQL), least to most severe.</summary>
    public static readonly string[] Levels = ["debug", "info", "warn", "error"];

    /// <summary>The services that write <c>logs</c> rows.</summary>
    public static readonly string[] Services = ["data-pipeline", "ai-service", "backend", "bot"];

    public static RouteGroupBuilder MapLogsEndpoints(this RouteGroupBuilder group)
    {
        group.MapGet("/logs/bot", GetRecentAsync)
            .WithName("GetBotLogs").WithTags("logs")
            .WithSummary("Recent bot/system log entries, newest first.")
            .ProducesError(StatusCodes.Status400BadRequest);
        return group;
    }

    /// <summary>
    /// The newest <c>limit</c> entries, newest first. <c>level</c> is a minimum severity
    /// (<c>warn</c> → warn + error); <c>service</c> narrows to one writer.
    /// </summary>
    internal static async Task<IReadOnlyList<LogDto>> GetRecentAsync(
        SoliaDbContext db, string? level, string? service, int? limit, CancellationToken cancellationToken)
    {
        string? minLevel = QueryValidation.OneOf(level, "level", Levels);
        string? writer = QueryValidation.OneOf(service, "service", Services);
        int take = QueryValidation.Limit(limit, DefaultLimit, MaxLimit);

        IQueryable<LogEntry> query = db.Logs.AsNoTracking();
        if (minLevel is not null)
        {
            string[] atLeast = Levels[Array.IndexOf(Levels, minLevel)..];
            query = query.Where(l => atLeast.Contains(l.Level));
        }

        if (writer is not null)
        {
            query = query.Where(l => l.Service == writer);
        }

        List<LogEntry> rows = await query
            .OrderByDescending(l => l.Ts).ThenByDescending(l => l.Id)
            .Take(take)
            .ToListAsync(cancellationToken);

        return rows.Select(l => new LogDto(l.Id, l.Ts, l.Service, l.Level, l.Message,
            l.Context is null ? null : new RawJson(l.Context))).ToList();
    }
}
