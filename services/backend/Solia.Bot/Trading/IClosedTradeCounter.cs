namespace Solia.Bot.Trading;

/// <summary>
/// How many trades have closed — the input to the retrain threshold ("after N new closed trades",
/// docs/ARCHITECTURE.md). Counting only: the trigger that compares it to N and starts a retrain is
/// Sprint 7, which passes the end of the last training run as <c>since</c>.
/// </summary>
public interface IClosedTradeCounter
{
    /// <summary>Trades closed strictly after <paramref name="since"/>, or ever when it is null.</summary>
    Task<long> CountClosedSinceAsync(DateTimeOffset? since, CancellationToken cancellationToken = default);
}
