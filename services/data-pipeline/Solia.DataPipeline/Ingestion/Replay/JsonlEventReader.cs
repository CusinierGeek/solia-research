using System.Runtime.CompilerServices;
using System.Text;
using Solia.DataPipeline.Ingestion.PumpApi;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Decodes a decompressed JSON-Lines stream (one pumpapi message per line) into
/// <see cref="RawEvent"/>s via the same parser the live source uses, so replayed events are
/// identical to live ones. Blank lines are skipped; the parser already drops unparseable lines.
/// Does not take ownership of the stream (the caller disposes it). Cancellation always throws
/// <see cref="OperationCanceledException"/>, so a partial read is never mistaken for the end.
/// </summary>
internal static class JsonlEventReader
{
    /// <summary>Archives decompress to gigabytes; read them in large chunks.</summary>
    internal const int BufferSize = 64 * 1024;

    public static StreamReader CreateReader(Stream utf8Jsonl) =>
        new(utf8Jsonl, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, BufferSize, leaveOpen: true);

    public static async IAsyncEnumerable<RawEvent> ReadAsync(
        Stream utf8Jsonl, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using StreamReader reader = CreateReader(utf8Jsonl);
        while (true)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string? line = await reader.ReadLineAsync(cancellationToken);
            if (line is null)
            {
                yield break; // end of stream
            }

            if (ParseLine(line) is { } evt)
            {
                yield return evt;
            }
        }
    }

    public static RawEvent? ParseLine(string line) =>
        string.IsNullOrWhiteSpace(line) ? null : PumpApiEventParser.Parse(line);
}
