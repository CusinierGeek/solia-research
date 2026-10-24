using Microsoft.EntityFrameworkCore;
using Solia.Api.Endpoints;
using Solia.Core.Data;
using Solia.Core.Data.Entities;

namespace Solia.Api.Live;

/// <summary>Re-reads notified rows in their REST shape; a row that no longer exists is null / left out.</summary>
public interface ILiveRowReader
{
    Task<PredictionRecordDto?> PredictionAsync(long id, CancellationToken cancellationToken);

    Task<TradeDto?> TradeAsync(long id, CancellationToken cancellationToken);

    /// <summary>The <c>logs</c> rows among <paramref name="ids"/>, in id order — one round trip for a burst.</summary>
    Task<IReadOnlyList<LogDto>> LogsAsync(IReadOnlyCollection<long> ids, CancellationToken cancellationToken);
}

/// <summary>
/// Reads through the pooled <see cref="SoliaDbContext"/> with the read endpoints' own projections, so a live
/// event carries exactly the item the matching REST endpoint returns.
/// </summary>
internal sealed class DbLiveRowReader(IServiceScopeFactory scopes) : ILiveRowReader
{
    public async Task<PredictionRecordDto?> PredictionAsync(long id, CancellationToken cancellationToken)
    {
        await using AsyncServiceScope scope = scopes.CreateAsyncScope();
        return await Db(scope).Predictions.AsNoTracking().Where(p => p.Id == id)
            .Select(PredictionsEndpoints.ToDto)
            .SingleOrDefaultAsync(cancellationToken);
    }

    public async Task<TradeDto?> TradeAsync(long id, CancellationToken cancellationToken)
    {
        await using AsyncServiceScope scope = scopes.CreateAsyncScope();
        return await Db(scope).Trades.AsNoTracking().Where(t => t.Id == id)
            .Select(TradesEndpoints.ToDto)
            .SingleOrDefaultAsync(cancellationToken);
    }

    public async Task<IReadOnlyList<LogDto>> LogsAsync(IReadOnlyCollection<long> ids, CancellationToken cancellationToken)
    {
        await using AsyncServiceScope scope = scopes.CreateAsyncScope();
        List<LogEntry> rows = await Db(scope).Logs.AsNoTracking().Where(l => ids.Contains(l.Id))
            .OrderBy(l => l.Id)
            .ToListAsync(cancellationToken);
        return rows.ConvertAll(LogsEndpoints.ToDto);
    }

    private static SoliaDbContext Db(AsyncServiceScope scope) => scope.ServiceProvider.GetRequiredService<SoliaDbContext>();
}
