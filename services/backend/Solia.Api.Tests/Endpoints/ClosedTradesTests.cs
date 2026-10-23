using Solia.Api.Endpoints;
using Solia.Core.Data.Entities;

namespace Solia.Api.Tests.Endpoints;

public class ClosedTradesTests
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

    private static Trade NewTrade(long id, string status, DateTimeOffset? closedAt) =>
        new() { Id = id, Mint = "m", OpenedAt = T0, Status = status, ClosedAt = closedAt };

    [Fact]
    public void Closed_keeps_only_closed_status_with_a_closed_at()
    {
        IQueryable<Trade> trades = new[]
        {
            NewTrade(1, "closed", T0.AddMinutes(1)),
            NewTrade(2, "closed", null), // closed without a closed_at: not in the set
            NewTrade(3, "open", null),
            NewTrade(4, "open", T0.AddMinutes(2)), // a closed_at alone does not make it closed
            NewTrade(5, "closed", T0.AddMinutes(3)),
        }.AsQueryable();

        Assert.Equal([1L, 5L], trades.Closed().Select(t => t.Id).ToArray());
    }
}
