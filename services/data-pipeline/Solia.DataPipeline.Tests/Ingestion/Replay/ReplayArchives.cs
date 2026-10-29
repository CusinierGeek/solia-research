using System.Globalization;
using System.Text;
using Solia.DataPipeline.Ingestion.Replay;
using ZstdSharp;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

/// <summary>Builds in-memory hourly archives and a scratch directory for the replay/mirror tests.</summary>
internal static class ReplayArchives
{
    public static string BuyLine(string mint, DateTimeOffset at) =>
        $$"""{"txType":"buy","mint":"{{mint}}","pool":"pump","price":0.001,"timestamp":{{at.ToUnixTimeMilliseconds()}}}""";

    public static byte[] Zstd(string text)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        using var output = new MemoryStream();
        using (var compressor = new CompressionStream(output))
        {
            compressor.Write(bytes, 0, bytes.Length);
        }

        return output.ToArray();
    }

    /// <summary>An opener over <paramref name="archives"/> (absent or null = missing hour) that records each request.</summary>
    public static Func<DateTimeOffset, CancellationToken, Task<Stream?>> Opener(
        IReadOnlyDictionary<DateTimeOffset, byte[]?> archives, List<DateTimeOffset>? requested = null) =>
        (hour, _) =>
        {
            requested?.Add(hour);
            byte[]? bytes = archives.GetValueOrDefault(hour);
            return Task.FromResult<Stream?>(bytes is null ? null : new MemoryStream(bytes));
        };

    /// <summary>An <see cref="IReplayArchive"/> over a test opener.</summary>
    public sealed class DelegateArchive(Func<DateTimeOffset, CancellationToken, Task<Stream?>> open) : IReplayArchive
    {
        public Task<Stream?> OpenAsync(DateTimeOffset hour, CancellationToken cancellationToken) => open(hour, cancellationToken);
    }

    /// <summary>Runs <paramref name="body"/> with <paramref name="name"/> as the current culture (restored after).</summary>
    public static void WithCulture(string name, Action body)
    {
        CultureInfo previous = CultureInfo.CurrentCulture;
        CultureInfo.CurrentCulture = new CultureInfo(name);
        try
        {
            body();
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }

    public static string NewTempDir()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"solia-mirror-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>A stream that returns <paramref name="prefix"/> and then fails, like a dropped download.</summary>
    public sealed class DroppingStream(byte[] prefix) : MemoryStream(prefix)
    {
        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = base.Read(buffer, offset, count);
            return n > 0 ? n : throw new IOException("connection reset");
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            int n = await base.ReadAsync(buffer, cancellationToken);
            return n > 0 ? n : throw new IOException("connection reset");
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
            ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
    }
}
