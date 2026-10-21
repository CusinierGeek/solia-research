using System.Runtime.CompilerServices;
using ZstdSharp;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Replays events from a local JSON-Lines file. A <c>.zst</c> file is zstd-decompressed on the
/// fly; anything else is read plain. Used for tests/fixtures and for replaying a mirrored archive.
/// A missing file throws <see cref="FileNotFoundException"/> when enumeration starts.
/// </summary>
public sealed class FileReplaySource(string path) : IDataSource
{
    public async IAsyncEnumerable<RawEvent> StreamAsync([EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await using var file = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read, JsonlEventReader.BufferSize,
            FileOptions.Asynchronous | FileOptions.SequentialScan);

        // Disposing the file twice (here and via the decompressor) is harmless.
        await using Stream content = path.EndsWith(".zst", StringComparison.OrdinalIgnoreCase)
            ? new DecompressionStream(file)
            : file;

        await foreach (RawEvent evt in JsonlEventReader.ReadAsync(content, cancellationToken))
        {
            yield return evt;
        }
    }
}
