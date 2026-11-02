using Solia.DataPipeline.Features;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// The sink of a replay's warm-up (<c>--warmup-hours</c>): forwards only what is dated at or after the window's
/// start. Snapshots of the warm-up hours are dropped — the engine still computes them, so the state they advance
/// (growth baselines, wallet memory, market context) is exactly what live ingestion would hold — and so are the
/// deactivations of the warm-up, which would touch rows this replay never wrote.
/// </summary>
public sealed class WarmupSink(ITokenSink inner, DateTimeOffset windowStart) : ITokenSink
{
    public Task WriteAsync(FlushBatch batch, CancellationToken cancellationToken)
    {
        if (batch.Ts < windowStart)
        {
            return Task.CompletedTask;
        }

        IReadOnlyList<FeatureSnapshot> snapshots = batch.Snapshots.Where(s => s.Ts >= windowStart).ToList();
        return inner.WriteAsync(batch with { Snapshots = snapshots }, cancellationToken);
    }

    public Task DeactivateAllAsync(CancellationToken cancellationToken) => inner.DeactivateAllAsync(cancellationToken);
}
