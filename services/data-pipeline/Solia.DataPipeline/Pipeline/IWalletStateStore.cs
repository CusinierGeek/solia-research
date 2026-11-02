using Solia.DataPipeline.Features;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// Durable copy of the cross-token wallet memory (<see cref="WalletStore"/>), so a live restart doesn't reset every
/// wallet to fresh. Live ingestion only: a replay rebuilds its state from a warm-up instead and must never write here.
/// </summary>
public interface IWalletStateStore
{
    /// <summary>The saved state, or null when nothing was ever saved.</summary>
    Task<WalletStateSnapshot?> LoadAsync(CancellationToken cancellationToken);

    /// <summary>Upserts the changed wallets, deletes the evicted ones and records the history bounds, atomically.</summary>
    Task SaveAsync(WalletStateChanges changes, CancellationToken cancellationToken);
}
