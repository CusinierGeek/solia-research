using Solia.Bot.Sessions;

namespace Solia.Bot.Tests.Sessions;

public class BotSessionRulesTests
{
    [Fact]
    public void Without_a_current_session_a_new_one_starts() =>
        Assert.Equal(BotSessionDecision.New, BotSessionRules.Decide(null, "h", championId: 1));

    [Fact]
    public void The_same_fingerprint_continues_the_current_session() =>
        Assert.Equal(BotSessionDecision.Continue, BotSessionRules.Decide(new CurrentBotSession(3, "h", 1), "h", championId: 1));

    [Fact]
    public void A_different_fingerprint_starts_a_new_session() =>
        Assert.Equal(BotSessionDecision.New, BotSessionRules.Decide(new CurrentBotSession(3, "h", 1), "other", championId: 1));

    [Theory]
    [InlineData(1L, 1L)]
    [InlineData(null, null)]
    public void A_backfilled_session_served_by_the_same_champion_is_adopted(long? sessionModel, long? champion) =>
        Assert.Equal(BotSessionDecision.Adopt, BotSessionRules.Decide(new CurrentBotSession(2, null, sessionModel), "h", champion));

    [Theory]
    [InlineData(1L, 2L)]  // promoted since the backfill
    [InlineData(1L, null)]
    [InlineData(null, 2L)]
    public void A_backfilled_session_under_another_champion_is_replaced(long? sessionModel, long? champion) =>
        Assert.Equal(BotSessionDecision.New, BotSessionRules.Decide(new CurrentBotSession(2, null, sessionModel), "h", champion));

    [Fact]
    public void An_empty_fingerprint_is_rejected() =>
        Assert.Throws<ArgumentException>(() => BotSessionRules.Decide(null, "", championId: null));
}
