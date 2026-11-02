using Solia.DataPipeline.Features;
using Solia.DataPipeline.Pipeline;

namespace Solia.DataPipeline.Tests.Pipeline;

/// <summary>FE-T9 (#189): the live wallet memory round-trips through <c>wallet_state</c> (migration 0014).</summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class PostgresWalletStateStoreTests(PostgresFixture db) : IClassFixture<PostgresFixture>, IAsyncLifetime
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private static readonly long S0 = T0.ToUnixTimeSeconds();

    private PostgresWalletStateStore Store => new(db.DataSource);

    public async Task InitializeAsync()
    {
        await using var cmd = db.DataSource.CreateCommand("TRUNCATE wallet_state, wallet_state_meta");
        await cmd.ExecuteNonQueryAsync();
    }

    public Task DisposeAsync() => Task.CompletedTask;

    private static WalletRecord Wallet(string wallet, long first, int trades = 1, params long[] launches) =>
        new(wallet, first, first + 60, trades, 1, launches.Length, launches);

    [Fact]
    public async Task Nothing_saved_loads_as_null()
    {
        Assert.Null(await Store.LoadAsync(default));
    }

    [Fact]
    public async Task Saved_wallets_and_history_bounds_load_back()
    {
        await Store.SaveAsync(new WalletStateChanges(T0, T0.AddHours(1),
            [Wallet("a", S0), Wallet("dev", S0 + 10, 3, S0 + 100, S0 + 200)], []), default);

        WalletStateSnapshot loaded = (await Store.LoadAsync(default))!;

        Assert.Equal(T0, loaded.HistoryStart);
        Assert.Equal(T0.AddHours(1), loaded.LastEvent);
        WalletRecord dev = Assert.Single(loaded.Wallets, w => w.Wallet == "dev");
        Assert.Equal(S0 + 10, dev.FirstSeen);
        Assert.Equal(S0 + 70, dev.LastSeen);
        Assert.Equal(3, dev.TradeCount);
        Assert.Equal(2, dev.TokensCreated);
        Assert.Equal([S0 + 100, S0 + 200], dev.RecentLaunches);
        Assert.Empty(Assert.Single(loaded.Wallets, w => w.Wallet == "a").RecentLaunches);
    }

    [Fact]
    public async Task A_later_save_updates_changed_wallets_and_deletes_evicted_ones()
    {
        await Store.SaveAsync(new WalletStateChanges(T0, T0, [Wallet("a", S0), Wallet("b", S0), Wallet("c", S0)], []), default);

        await Store.SaveAsync(new WalletStateChanges(T0, T0.AddHours(2), [Wallet("a", S0, trades: 9)], ["b", "never-saved"]), default);

        WalletStateSnapshot loaded = (await Store.LoadAsync(default))!;
        Assert.Equal(["a", "c"], loaded.Wallets.Select(w => w.Wallet).Order());
        Assert.Equal(9, loaded.Wallets.Single(w => w.Wallet == "a").TradeCount);
        Assert.Equal(T0.AddHours(2), loaded.LastEvent);
        Assert.Equal(1L, await db.ScalarAsync<long>("SELECT count(*) FROM wallet_state_meta"));
    }

    [Fact]
    public async Task A_restarted_store_restores_the_wallets_of_the_previous_run()
    {
        var before = new WalletStore(new Options.PipelineOptions(), trackChanges: true);
        before.Apply(Features.WalletStoreTests.Trade("A", "veteran", T0));
        before.Advance(T0.AddSeconds(1));
        await Store.SaveAsync(before.DrainChanges()!, default);

        var after = new WalletStore(new Options.PipelineOptions(), trackChanges: true);
        after.Load((await Store.LoadAsync(default))!);

        Assert.Equal(S0, after.Get("veteran", T0.AddMinutes(1))!.FirstSeen);
    }
}
