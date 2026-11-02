using Solia.DataPipeline.Metadata;
using Solia.DataPipeline.Tests.Pipeline;

namespace Solia.DataPipeline.Tests.Metadata;

/// <summary>FE-T12 (#196): the metadata fetch results round-trip through <c>token_metadata</c> (migration 0016).</summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class PostgresMetadataStoreTests(PostgresFixture db) : IClassFixture<PostgresFixture>, IAsyncLifetime
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);

    private PostgresMetadataStore Store => new(db.DataSource);

    public async Task InitializeAsync()
    {
        await using var cmd = db.DataSource.CreateCommand("TRUNCATE token_metadata");
        await cmd.ExecuteNonQueryAsync();
    }

    public Task DisposeAsync() => Task.CompletedTask;

    [Fact]
    public async Task Saved_results_load_back_within_the_fetch_time_range()
    {
        await Store.SaveAsync(new TokenMetadataRecord("OK", T0, MetadataFetchStatus.Ok, HasWebsite: true, HasTwitter: false, HasTelegram: true), default);
        await Store.SaveAsync(new TokenMetadataRecord("FAIL", T0.AddHours(1), MetadataFetchStatus.Failed), default);
        await Store.SaveAsync(new TokenMetadataRecord("OLD", T0.AddDays(-3), MetadataFetchStatus.Unsupported), default);
        await Store.SaveAsync(new TokenMetadataRecord("LATER", T0.AddDays(3), MetadataFetchStatus.Ok), default);

        IReadOnlyList<TokenMetadataRecord> loaded = await Store.LoadAsync(T0.AddDays(-1), T0.AddDays(1), default);

        Assert.Equal(["FAIL", "OK"], loaded.Select(r => r.Mint).Order().ToArray());
        TokenMetadataRecord ok = loaded.Single(r => r.Mint == "OK");
        Assert.Equal(new TokenMetadataRecord("OK", T0, MetadataFetchStatus.Ok, true, false, true), ok);
        Assert.Equal(MetadataFetchStatus.Failed, loaded.Single(r => r.Mint == "FAIL").Status);
        Assert.Equal(3, (await Store.LoadAsync(T0.AddDays(-1), null, default)).Count); // live: no upper bound
    }

    [Fact]
    public async Task A_token_keeps_its_first_result()
    {
        await Store.SaveAsync(new TokenMetadataRecord("M", T0, MetadataFetchStatus.Failed), default);
        await Store.SaveAsync(new TokenMetadataRecord("M", T0.AddMinutes(1), MetadataFetchStatus.Ok, true, true, true), default);

        TokenMetadataRecord loaded = Assert.Single(await Store.LoadAsync(T0.AddDays(-1), null, default));
        Assert.Equal(MetadataFetchStatus.Failed, loaded.Status);
        Assert.Equal(T0, loaded.FetchedAt);
    }
}
