using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.Replay;
using ZstdSharp;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public class FileReplaySourceTests
{
    private const string Wsol = "So11111111111111111111111111111111111111112";
    private const string Usdc = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

    /// <summary>20 real lines captured from replay.pumpapi.io/2026/04/20/12.jsonl.zst.</summary>
    private static readonly string FixturePath =
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "sample.jsonl");

    private static readonly EventKind[] FixtureKinds =
    [
        EventKind.Sell, EventKind.Buy, EventKind.Buy, EventKind.Buy, EventKind.Sell, EventKind.Transfer,
        EventKind.Sell, EventKind.Buy, EventKind.Buy, EventKind.Buy, EventKind.Create, EventKind.Create,
        EventKind.Buy, EventKind.CreatePool, EventKind.RemoveLiquidity, EventKind.Migrate, EventKind.CreatePool,
        EventKind.Create, EventKind.AddLiquidity,
    ];

    [Fact]
    public async Task Reads_the_real_archive_fixture()
    {
        List<RawEvent> events = await Collect(new FileReplaySource(FixturePath));

        // 20 lines; the SOL-only transfer has no token to attribute and is dropped.
        Assert.Equal(FixtureKinds, events.Select(e => e.Kind));
        Assert.All(events, e => Assert.Equal(2026, e.Timestamp.Year));

        RawEvent pumpSell = events[0];
        Assert.Equal("H7ETD67xPws33iKr3DZWsy1ZYKPgqLkzQB5n7daxpump", pumpSell.Mint);
        Assert.Equal("pumpfun", pumpSell.Venue);
        Assert.False(pumpSell.IsMigrated);
        Assert.Equal(Wsol, pumpSell.QuoteMint);
        Assert.Equal(0.224586568m, pumpSell.QuoteAmount);   // from solAmount
        Assert.Equal(42.954387564m, pumpSell.QuoteInPool);  // from solInPool
        Assert.Equal(72.954387564m, pumpSell.VirtualQuoteReserves); // from vSolInBondingCurve
        Assert.NotNull(pumpSell.MarketCap);                 // from marketCapSol
        Assert.NotNull(pumpSell.Price);

        Assert.Equal("67EnRFJ2u3ZYcgJKiaQbYujQYud2x4wAnT5BiNZApump", events[5].Mint); // token leg, SOL legs skipped
        Assert.Null(events[5].QuoteMint);

        RawEvent usdcBuy = events[9];
        Assert.Equal("meteora-launchpad", usdcBuy.Venue);
        Assert.Equal(Usdc, usdcBuy.QuoteMint);
        Assert.Equal(1.614924m, usdcBuy.QuoteAmount);       // from quoteAmount

        Assert.Equal("meteora", events[1].Venue);
        Assert.Equal("raydium", events[7].Venue);
        Assert.Equal(100.0, events[14].BurnedLiquidityPercent);

        RawEvent migrate = events[15];
        Assert.Equal("pumpswap", migrate.Venue);
        Assert.True(migrate.IsMigrated);
    }

    [Fact]
    public async Task Reads_zst_compressed_fixture_identically()
    {
        string tmp = TempPath(".jsonl.zst");
        try
        {
            await File.WriteAllBytesAsync(tmp, Zstd(await File.ReadAllBytesAsync(FixturePath)));

            List<RawEvent> plain = await Collect(new FileReplaySource(FixturePath));
            List<RawEvent> compressed = await Collect(new FileReplaySource(tmp));

            Assert.Equal(plain, compressed);
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public async Task Skips_blank_and_unparseable_lines()
    {
        string tmp = TempPath(".jsonl");
        try
        {
            await File.WriteAllTextAsync(tmp,
                "\n   \nnot json\n{\"txType\":\"buy\",\"mint\":\"OK\",\"pool\":\"pump\",\"timestamp\":1776686400000}\n");

            List<RawEvent> events = await Collect(new FileReplaySource(tmp));

            Assert.Equal("OK", Assert.Single(events).Mint);
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public async Task Pre_cancelled_token_throws_instead_of_ending_quietly()
    {
        using var cts = new CancellationTokenSource();
        await cts.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => Collect(new FileReplaySource(FixturePath), cts.Token));
    }

    [Fact]
    public async Task Missing_file_throws_file_not_found()
    {
        string missing = TempPath(".jsonl");

        await Assert.ThrowsAsync<FileNotFoundException>(() => Collect(new FileReplaySource(missing)));
    }

    [Fact]
    public async Task Corrupt_zst_file_throws()
    {
        string tmp = TempPath(".jsonl.zst");
        try
        {
            await File.WriteAllBytesAsync(tmp, "this is not zstd"u8.ToArray());

            await Assert.ThrowsAnyAsync<Exception>(() => Collect(new FileReplaySource(tmp)));
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    [Fact]
    public async Task Truncated_zst_file_throws()
    {
        string tmp = TempPath(".jsonl.zst");
        try
        {
            byte[] full = Zstd(await File.ReadAllBytesAsync(FixturePath));
            await File.WriteAllBytesAsync(tmp, full[..(full.Length / 2)]);

            await Assert.ThrowsAnyAsync<Exception>(() => Collect(new FileReplaySource(tmp)));
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    private static string TempPath(string extension) =>
        Path.Combine(Path.GetTempPath(), $"solia-{Guid.NewGuid():N}{extension}");

    private static byte[] Zstd(byte[] plain)
    {
        using var output = new MemoryStream();
        using (var compressor = new CompressionStream(output))
        {
            compressor.Write(plain, 0, plain.Length);
        }
        return output.ToArray();
    }

    private static async Task<List<RawEvent>> Collect(IDataSource source, CancellationToken ct = default)
    {
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(10)); // fail fast rather than hang
        var events = new List<RawEvent>();
        await foreach (RawEvent e in source.StreamAsync(timeout.Token))
        {
            events.Add(e);
        }
        return events;
    }
}
