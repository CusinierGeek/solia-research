using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Features;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

/// <summary>
/// End-to-end: replay the real-archive fixture through the engine into an ephemeral PostgreSQL
/// (the T1 migrations applied), and assert features land — including a post-bonding token — and
/// that a re-run is idempotent.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class PipelineEndToEndTests(PostgresFixture db) : IClassFixture<PostgresFixture>
{
    private static readonly string FixturePath =
        Path.Combine(AppContext.BaseDirectory, "Fixtures", "sample.jsonl");

    [Fact]
    public async Task Replaying_the_fixture_persists_features_including_a_migrated_token()
    {
        var options = new PipelineOptions { SnapshotCadenceSeconds = 1, WindowSeconds = 86_400 };
        var sink = new PostgresTokenSink(db.DataSource);

        await Run(options, sink);

        Assert.True(await db.ScalarAsync<long>("SELECT count(*) FROM features") > 0);
        Assert.True(await db.ScalarAsync<long>("SELECT count(*) FROM features WHERE is_migrated") > 0);
        Assert.True(await db.ScalarAsync<long>("SELECT count(*) FROM tokens_live") > 0);
        Assert.Equal(0, await db.ScalarAsync<long>("SELECT count(*) FROM tokens_history WHERE first_seen_at > last_seen_at"));

        // Idempotent: replaying the same window again inserts no new feature rows (ON CONFLICT).
        long before = await db.ScalarAsync<long>("SELECT count(*) FROM features");
        await Run(options, sink);
        long after = await db.ScalarAsync<long>("SELECT count(*) FROM features");
        Assert.Equal(before, after);
    }

    [Fact]
    public async Task Replay_command_path_is_idempotent_and_rebuild_refreshes_rows()
    {
        // The fixture spans 2026-04-20 11:59–12:06 UTC: serve it all as the 11:00 archive, 12:00 missing.
        var hour = new DateTimeOffset(2026, 4, 20, 11, 0, 0, TimeSpan.Zero);
        var window = new ReplayWindow(hour, hour.AddHours(2));
        // Plus, just before it, a token created inside the window and then traded: the only kind --rebuild
        // may overwrite (the fixture's own creates are never traded, so never snapshotted).
        var born = new DateTimeOffset(2026, 4, 20, 11, 59, 50, TimeSpan.Zero);
        string bornHere = string.Join("\n", new[]
        {
            $$"""{"txType":"create","mint":"BORNHERE","pool":"pump","price":0.001,"timestamp":{{born.ToUnixTimeMilliseconds()}}}""",
        }.Concat(Enumerable.Range(1, 3).Select(i =>
            Solia.DataPipeline.Tests.Ingestion.Replay.ReplayArchives.BuyLine("BORNHERE", born.AddSeconds(2 * i)))));
        byte[] archive = Solia.DataPipeline.Tests.Ingestion.Replay.ReplayArchives.Zstd(
            bornHere + "\n" + await File.ReadAllTextAsync(FixturePath));
        var opener = new Solia.DataPipeline.Tests.Ingestion.Replay.ReplayArchives.DelegateArchive(
            (h, _) => Task.FromResult<Stream?>(h == hour ? new MemoryStream(archive) : null));
        var options = new PipelineOptions { SnapshotCadenceSeconds = 1, WindowSeconds = 86_400 };
        var runner = new ReplayRunner(NullLoggerFactory.Instance);
        await db.ScalarAsync<int>("TRUNCATE features, tokens_live, tokens_history CASCADE; SELECT 1");

        async Task<ReplaySummary> Replay(bool rebuild)
        {
            var sink = new PostgresTokenSink(db.DataSource, rebuild);
            return await runner.ReplayAsync(window, opener, sink, () => sink.FeatureRowsWritten, options, default);
        }

        ReplaySummary first = await Replay(rebuild: false);
        long rows = await db.ScalarAsync<long>("SELECT count(*) FROM features");
        Assert.True(first.IsComplete);
        Assert.Equal(rows, first.FeatureRowsWritten);
        Assert.Equal(hour.AddHours(1), Assert.Single(first.MissingHours));

        ReplaySummary again = await Replay(rebuild: false);
        Assert.Equal(0, again.FeatureRowsWritten); // nothing new
        Assert.Equal(rows, await db.ScalarAsync<long>("SELECT count(*) FROM features"));

        // Simulate a column added after the first replay: wipe it, then rebuild fills it back.
        long filled = await db.ScalarAsync<long>("SELECT count(*) FROM features WHERE unique_traders IS NOT NULL");
        Assert.True(filled > 0);
        await db.ScalarAsync<int>("UPDATE features SET unique_traders = NULL; SELECT 1");

        // Only tokens whose creation the replay saw (whole history in the window) are refreshed; the others
        // keep their stored rows rather than being overwritten from a truncated history.
        long refreshable = await db.ScalarAsync<long>(
            "SELECT count(*) FROM features f JOIN tokens_history h USING (mint) WHERE h.created_at IS NOT NULL");
        Assert.InRange(refreshable, 1, rows - 1); // the fixture has both kinds

        ReplaySummary rebuilt = await Replay(rebuild: true);
        Assert.Equal(refreshable, rebuilt.FeatureRowsWritten);
        Assert.Equal(rows, await db.ScalarAsync<long>("SELECT count(*) FROM features"));
        Assert.Equal(0, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features f JOIN tokens_history h USING (mint) " +
            "WHERE h.created_at IS NOT NULL AND f.unique_traders IS NULL")); // refreshed rows filled back
        Assert.Equal(0, await db.ScalarAsync<long>(
            "SELECT count(*) FROM features f JOIN tokens_history h USING (mint) " +
            "WHERE h.created_at IS NULL AND f.unique_traders IS NOT NULL")); // the others kept as stored
        Assert.False(await db.ScalarAsync<bool>("SELECT bool_or(is_active) FROM tokens_live")); // replay ends inactive
    }

    private static Task Run(PipelineOptions options, ITokenSink sink) =>
        new PipelineEngine(
            new FileReplaySource(FixturePath),
            new TokenWindowStore(options),
            sink,
            Microsoft.Extensions.Options.Options.Create(options),
            NullLogger<PipelineEngine>.Instance).RunAsync(CancellationToken.None);
}
