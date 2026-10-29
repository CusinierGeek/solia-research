namespace Solia.Bot.Trading;

/// <summary>
/// How many trades have closed — the input to the retrain threshold ("after N new closed trades",
/// docs/ARCHITECTURE.md). Counting only: the trigger that compares it to N and starts a retrain is
/// Sprint 7, which passes as <c>since</c> the <c>started_at</c> of the latest training run whose decision is
/// <c>promoted</c> / <c>rejected</c> — a <c>failed</c> or <c>rollback</c> run trained nothing on the new trades,
/// so it does not reset the count (same rule as <c>GET /model/status</c>, docs/API-CONTRACTS.md).
/// </summary>
public interface IClosedTradeCounter
{
    /// <summary>Trades closed strictly after <paramref name="since"/>, or ever when it is null.</summary>
    Task<long> CountClosedSinceAsync(DateTimeOffset? since, CancellationToken cancellationToken = default);
}
