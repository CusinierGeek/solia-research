using Solia.Api.Live;

namespace Solia.Api.Tests.Live;

public class LiveNotificationTests
{
    [Theory]
    [InlineData("prediction_created:1", LiveEvents.PredictionCreated, 1L)]
    [InlineData("trade_opened:42", LiveEvents.TradeOpened, 42L)]
    [InlineData("trade_closed:9223372036854775807", LiveEvents.TradeClosed, long.MaxValue)]
    [InlineData("log_written:7", LiveEvents.LogWritten, 7L)]
    public void A_trigger_payload_parses_into_its_event_and_row_id(string payload, string expectedEvent, long expectedId)
    {
        Assert.True(LiveNotification.TryParse(payload, out LiveNotification notification));
        Assert.Equal(new LiveNotification(expectedEvent, expectedId), notification);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("log_written")]
    [InlineData(":5")]
    [InlineData("log_written:")]
    [InlineData("log_written:0")]
    [InlineData("log_written:-3")]
    [InlineData("log_written: 3")]
    [InlineData("log_written:3.5")]
    [InlineData("log_written:99999999999999999999")]
    [InlineData("stats_changed:1")]
    [InlineData("LOG_WRITTEN:1")]
    [InlineData("tokens_updated:1")]
    public void Anything_but_a_known_row_event_and_a_positive_id_is_rejected(string? payload)
    {
        Assert.False(LiveNotification.TryParse(payload, out LiveNotification notification));
        Assert.Equal(default, notification);
    }

    [Fact]
    public void Every_row_event_is_raised_by_the_migration_triggers()
    {
        string sql = string.Concat(Directory.GetFiles(PostgresFixture.FindMigrationsDir(), "*.sql").Select(File.ReadAllText));

        Assert.Contains($"pg_notify('{LiveEvents.Channel}'", sql);
        Assert.All(LiveEvents.RowEvents, e => Assert.Contains($"solia_notify_live('{e}')", sql));
    }
}
