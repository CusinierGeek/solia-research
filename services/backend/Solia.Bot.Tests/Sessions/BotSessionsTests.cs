using Microsoft.Extensions.Options;
using Microsoft.Extensions.Time.Testing;
using Solia.Bot.Configuration;
using Solia.Bot.Logging;
using Solia.Bot.Sessions;

namespace Solia.Bot.Tests.Sessions;

/// <summary>The bot's session tracker over the in-memory store: start/continue rules and a mid-run promotion.</summary>
public class BotSessionsTests
{
    private readonly InMemoryBotSessionStore _store = new() { ChampionId = 1, ChampionVersion = "v1" };
    private readonly FakeTimeProvider _time = new();

    private BotSessions Sessions(BotOptions? options = null) =>
        new(_store, Options.Create(options ?? new BotOptions()), [new BotRegimePart("horizon", "15m")], NullBotLogSink.Instance, _time);

    [Fact]
    public async Task The_first_start_opens_a_session()
    {
        BotSession session = await Sessions(new BotOptions { SessionLabel = "first" }).StartAsync();

        Assert.Equal(BotSessionDecision.New, session.Decision);
        Assert.Equal("v1", session.ModelVersion);
        Assert.Equal("first", Assert.Single(_store.Sessions).Label);
    }

    [Fact]
    public async Task A_restart_with_the_same_regime_continues_the_session()
    {
        BotSession first = await Sessions().StartAsync();
        BotSession second = await Sessions(new BotOptions { SessionLabel = "ignored" }).StartAsync();

        Assert.Equal(BotSessionDecision.Continue, second.Decision);
        Assert.Equal(first.Id, second.Id);
        Assert.Single(_store.Sessions);
    }

    [Fact]
    public async Task A_restart_with_another_configuration_opens_a_new_session()
    {
        BotSession first = await Sessions().StartAsync();
        BotSession second = await Sessions(new BotOptions { MinPumpProb = 0.3 }).StartAsync();

        Assert.Equal(BotSessionDecision.New, second.Decision);
        Assert.NotEqual(first.Id, second.Id);
        Assert.True(_store.Sessions[0].Ended);
    }

    [Fact]
    public async Task A_restart_under_a_new_champion_opens_a_new_session()
    {
        BotSession first = await Sessions().StartAsync();
        _store.ChampionId = 2;
        _store.ChampionVersion = "v2";

        BotSession second = await Sessions().StartAsync();

        Assert.NotEqual(first.Id, second.Id);
        Assert.Equal("v2", second.ModelVersion);
    }

    [Fact]
    public async Task A_promotion_while_running_rolls_over_once_the_refresh_interval_passed()
    {
        BotSessions sessions = Sessions();
        long first = (await sessions.StartAsync()).Id;
        _store.ChampionId = 2;
        _store.ChampionVersion = "v2";

        _time.Advance(BotSessions.RefreshInterval - TimeSpan.FromSeconds(1));
        Assert.Equal(first, await sessions.CurrentIdAsync()); // not re-checked yet

        _time.Advance(TimeSpan.FromSeconds(1));
        long second = await sessions.CurrentIdAsync();

        Assert.NotEqual(first, second);
        Assert.Equal(second, sessions.Current!.Id);
    }

    [Fact]
    public async Task No_trade_can_be_recorded_before_the_session_started() =>
        await Assert.ThrowsAsync<InvalidOperationException>(async () => await Sessions().CurrentIdAsync());
}
