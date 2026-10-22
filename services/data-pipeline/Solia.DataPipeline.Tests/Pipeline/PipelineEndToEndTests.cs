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

    private static Task Run(PipelineOptions options, ITokenSink sink) =>
        new PipelineEngine(
            new FileReplaySource(FixturePath),
            new TokenWindowStore(options),
            sink,
            Microsoft.Extensions.Options.Options.Create(options),
            NullLogger<PipelineEngine>.Instance).RunAsync(CancellationToken.None);
}
