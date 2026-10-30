using Microsoft.Extensions.Options;
using Microsoft.Extensions.Time.Testing;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Logging;
using Solia.Bot.Sessions;
using Solia.Bot.Tests.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Sessions;

/// <summary>
/// Integration tests against a real PostgreSQL (Testcontainers) — run in CI. Each test starts without any session or
/// champion.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class PostgresBotSessionStoreTests(PostgresFixture db) : IClassFixture<PostgresFixture>, IAsyncLifetime
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);
    private const string Options = "{\"minPumpProb\":0.25}";

    private PostgresBotSessionStore Store => new(db.DataSource);

    public async Task InitializeAsync()
    {
        await Exec("UPDATE trades SET session_id = NULL; DELETE FROM bot_sessions; UPDATE models SET is_champion = FALSE");
    }

    public Task DisposeAsync() => Task.CompletedTask;

    private Task Exec(string sql, params (string, object)[] parameters) => db.ScalarAsync<int>($"{sql}; SELECT 1", parameters);

    private async Task<long> SeedChampionAsync(string version)
    {
        await Exec("UPDATE models SET is_champion = FALSE");
        return await db.ScalarAsync<long>(
            "INSERT INTO models (version, path, is_champion) VALUES (@v, @p, TRUE) RETURNING id", ("v", version), ("p", $"models/{version}.pt"));
    }

    private async Task<long> SeedOpenTradeAsync()
    {
        string mint = $"mint-{Guid.NewGuid():N}";
        await Exec("INSERT INTO tokens_history (mint) VALUES (@m)", ("m", mint));
        return await db.ScalarAsync<long>("INSERT INTO trades (mint, status) VALUES (@m, 'open') RETURNING id", ("m", mint));
    }

    [Fact]
    public async Task A_first_start_records_the_regime_and_the_champion()
    {
        long model = await SeedChampionAsync($"v-{Guid.NewGuid():N}");

        BotSession session = await Store.BeginAsync(new BotSessionRequest(Options, " tuned ", NewRun: true));

        Assert.Equal(BotSessionDecision.New, session.Decision);
        Assert.Equal(model, session.ModelId);
        Assert.Equal(session.ConfigHash, await db.ScalarAsync<string>("SELECT config_hash FROM bot_sessions WHERE id = @id", ("id", session.Id)));
        Assert.Equal("tuned", await db.ScalarAsync<string>("SELECT label FROM bot_sessions WHERE id = @id", ("id", session.Id)));
        Assert.Equal(0.25, await db.ScalarAsync<double>("SELECT (options->>'minPumpProb')::float8 FROM bot_sessions WHERE id = @id", ("id", session.Id)));
        Assert.Equal(1, await db.ScalarAsync<int>("SELECT count(*) FROM bot_sessions WHERE ended_at IS NULL"));
    }

    [Fact]
    public async Task A_restart_continues_and_orphans_the_trades_left_open()
    {
        await SeedChampionAsync($"v-{Guid.NewGuid():N}");
        BotSession first = await Store.BeginAsync(new BotSessionRequest(Options, null, NewRun: true));
        long left = await SeedOpenTradeAsync();

        BotSession second = await Store.BeginAsync(new BotSessionRequest(Options, null, NewRun: true));

        Assert.Equal(BotSessionDecision.Continue, second.Decision);
        Assert.Equal(first.Id, second.Id);
        Assert.Equal(left, await db.ScalarAsync<long>("SELECT resumed_after_trade_id FROM bot_sessions WHERE id = @id", ("id", first.Id)));
    }

    [Fact]
    public async Task A_re_check_while_running_keeps_the_orphan_cut_off()
    {
        await SeedChampionAsync($"v-{Guid.NewGuid():N}");
        BotSession first = await Store.BeginAsync(new BotSessionRequest(Options, null, NewRun: true));
        long cutoff = await db.ScalarAsync<long>("SELECT resumed_after_trade_id FROM bot_sessions WHERE id = @id", ("id", first.Id));
        await SeedOpenTradeAsync(); // still held by the running bot

        await SeedChampionAsync($"v-{Guid.NewGuid():N}"); // a promotion while it runs
        BotSession second = await Store.BeginAsync(new BotSessionRequest(Options, null, NewRun: false));

        Assert.Equal(BotSessionDecision.New, second.Decision);
        Assert.Equal(cutoff, await db.ScalarAsync<long>("SELECT resumed_after_trade_id FROM bot_sessions WHERE id = @id", ("id", second.Id)));
        Assert.Equal(1, await db.ScalarAsync<int>(
            "SELECT count(*) FROM bot_sessions WHERE id = @id AND ended_at IS NOT NULL", ("id", first.Id)));
    }

    [Fact]
    public async Task Another_configuration_ends_the_current_session_and_opens_a_new_one()
    {
        await SeedChampionAsync($"v-{Guid.NewGuid():N}");
        BotSession first = await Store.BeginAsync(new BotSessionRequest(Options, null, NewRun: true));

        BotSession second = await Store.BeginAsync(new BotSessionRequest("{\"minPumpProb\":0.3}", "stricter", NewRun: true));

        Assert.Equal(BotSessionDecision.New, second.Decision);
        Assert.NotEqual(first.Id, second.Id);
        Assert.Equal(1, await db.ScalarAsync<int>("SELECT count(*) FROM bot_sessions WHERE ended_at IS NULL"));
        Assert.Equal(second.Id, await db.ScalarAsync<long>("SELECT id FROM bot_sessions WHERE ended_at IS NULL"));
    }

    [Theory]
    [InlineData(true, BotSessionDecision.Adopt)]
    [InlineData(false, BotSessionDecision.New)]
    public async Task A_backfilled_session_is_adopted_only_under_its_champion(bool sameChampion, BotSessionDecision expected)
    {
        long model = await SeedChampionAsync($"v-{Guid.NewGuid():N}");
        long backfilled = await db.ScalarAsync<long>(
            "INSERT INTO bot_sessions (started_at, model_id, label) VALUES (@t, @m, 'realistic fills (#165)') RETURNING id",
            ("t", T0), ("m", model));
        if (!sameChampion)
        {
            await SeedChampionAsync($"v-{Guid.NewGuid():N}");
        }

        BotSession session = await Store.BeginAsync(new BotSessionRequest(Options, "new label", NewRun: true));

        Assert.Equal(expected, session.Decision);
        Assert.Equal(sameChampion, session.Id == backfilled);
        Assert.Equal(session.ConfigHash, await db.ScalarAsync<string>("SELECT config_hash FROM bot_sessions WHERE id = @id", ("id", session.Id)));
        if (sameChampion)
        {
            // Adopting keeps its label: the human one given at backfill.
            Assert.Equal("realistic fills (#165)", await db.ScalarAsync<string>("SELECT label FROM bot_sessions WHERE id = @id", ("id", backfilled)));
        }
    }

    [Fact]
    public async Task The_trade_store_records_each_trade_in_the_current_session()
    {
        await SeedChampionAsync($"v-{Guid.NewGuid():N}");
        var sessions = new BotSessions(Store, Microsoft.Extensions.Options.Options.Create(new BotOptions()), [], NullBotLogSink.Instance, new FakeTimeProvider());
        BotSession session = await sessions.StartAsync();
        string mint = $"mint-{Guid.NewGuid():N}";
        await Exec("INSERT INTO tokens_history (mint) VALUES (@m)", ("m", mint));
        ModelPrediction p = PolicyFixtures.Buyable() with { Mint = mint };

        long id = await new PostgresTradeStore(db.DataSource, sessions).OpenAsync(
            new TradeOpening(mint, T0, 0.0001m, 2m, ExitLevels.Compute(p, 0.0001m, PolicyFixtures.Options()), p));

        Assert.Equal(session.Id, await db.ScalarAsync<long>("SELECT session_id FROM trades WHERE id = @id", ("id", id)));
    }
}
