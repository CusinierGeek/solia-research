namespace Solia.DataPipeline.Ingestion;

/// <summary>
/// A provider-agnostic source of <see cref="RawEvent"/>s. Implementations (live WebSocket in
/// T4, historical replay in T5, test fixtures) differ only in transport; everything downstream
/// consumes this stream. The sequence runs until the token is cancelled or the source ends.
/// </summary>
public interface IDataSource
{
    IAsyncEnumerable<RawEvent> StreamAsync(CancellationToken cancellationToken);
}
