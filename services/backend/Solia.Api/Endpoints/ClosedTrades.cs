using Solia.Core.Data.Entities;

namespace Solia.Api.Endpoints;

/// <summary>
/// The one definition of a closed trade the endpoints share — <c>status = 'closed'</c> with a <c>closed_at</c> —
/// so <c>/stats/*</c> and <c>/model/status</c> always count the same set.
/// </summary>
public static class ClosedTrades
{
    public const string Status = "closed";

    /// <summary>The closed trades of <paramref name="trades"/>.</summary>
    public static IQueryable<Trade> Closed(this IQueryable<Trade> trades) =>
        trades.Where(t => t.Status == Status && t.ClosedAt != null);
}
