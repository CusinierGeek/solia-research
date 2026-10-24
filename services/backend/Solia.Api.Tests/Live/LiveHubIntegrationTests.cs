using System.Text.Json;
using Solia.Api.Bot;
using Solia.Api.Live;
using Solia.Bot.Configuration;
using Solia.Bot.Domain;
using Solia.Bot.Trading;
using static Solia.Api.Tests.Live.LiveHubFixture;

namespace Solia.Api.Tests.Live;

/// <summary>
/// The live hub end to end — real Postgres triggers, the real host's LISTEN feed, a SignalR client: rows written
/// through the bot's own stores and by another process (raw SQL) each reach the client, in their REST shape.
/// Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class LiveHubIntegrationTests(LiveHubFixture live) : IClassFixture<LiveHubFixture>
{
    private static readonly DateTimeOffset T0 = new(2026, 2, 1, 0, 0, 0, TimeSpan.Zero);

    private static ModelPrediction Prediction(DateTimeOffset ts) => new()
    {
        Mint = Mint,
        Ts = ts,
        Horizon = "1m",
        PumpProb = 0.8,
        RugProb = 0.1,
        ExpectedReturn = 0.5,
        Confidence = 80,
        ModelVersion = ModelVersion,
    };

    [Fact]
    public async Task A_prediction_the_bot_records_is_pushed_as_its_history_item()
    {
        long id = await new PostgresPredictionStore(live.DataSource).InsertAsync(Prediction(T0), featureId: null);

        JsonElement pushed = await live.NextAsync(LiveEvents.PredictionCreated, id);

        // Byte for byte the item /predictions/history returns: the two shapes can't drift.
        using JsonDocument history = JsonDocument.Parse(
            await live.Client.GetStringAsync($"/api/v1/predictions/history?mint={Mint}&pageSize=100"));
        JsonElement item = history.RootElement.GetProperty("items").EnumerateArray().Single(e => e.GetProperty("id").GetInt64() == id);
        Assert.Equal(item.GetRawText(), pushed.GetRawText());
        Assert.Equal(ModelVersion, pushed.GetProperty("model_version").GetString());
        Assert.Equal("LIVE", pushed.GetProperty("symbol").GetString());
    }

    [Fact]
    public async Task A_trade_opened_then_closed_by_the_bot_is_pushed_at_each_step_with_a_stats_change()
    {
        ModelPrediction prediction = Prediction(T0.AddMinutes(1)) with
        {
            PredictionId = await new PostgresPredictionStore(live.DataSource).InsertAsync(Prediction(T0.AddMinutes(1)), null),
        };
        var store = new PostgresTradeStore(live.DataSource);
        var opening = new TradeOpening(Mint, T0.AddMinutes(1), 0.0001m, 2m,
            ExitLevels.Compute(prediction, 0.0001m, new BotOptions()), prediction);

        long id = await store.OpenAsync(opening);
        JsonElement opened = await live.NextAsync(LiveEvents.TradeOpened, id);
        Assert.Equal("open", opened.GetProperty("status").GetString());
        Assert.Equal(prediction.PredictionId, opened.GetProperty("prediction_id").GetInt64());
        Assert.Equal(JsonValueKind.Null, opened.GetProperty("closed_at").ValueKind);

        var position = new OpenPosition(id, opening, T0.AddMinutes(5));
        await store.CloseAsync(ClosedTrade.Settle(position, T0.AddMinutes(2), 0.0002m, ExitReason.TakeProfit));

        JsonElement closed = await live.NextAsync(LiveEvents.TradeClosed, id);
        Assert.Equal("closed", closed.GetProperty("status").GetString());
        Assert.Equal(2m, closed.GetProperty("pnl").GetDecimal());
        Assert.Equal("tp", closed.GetProperty("exit_reason").GetString());
        Assert.Equal(T0.AddMinutes(2), closed.GetProperty("closed_at").GetDateTimeOffset());

        JsonElement stats = await live.NextAsync(LiveEvents.StatsChanged, id, idProperty: "trade_id");
        Assert.Equal(["closed_at", "trade_id"], stats.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal));
        Assert.Equal(T0.AddMinutes(2), stats.GetProperty("closed_at").GetDateTimeOffset());
    }

    [Fact]
    public async Task A_log_row_written_by_another_process_is_pushed_as_its_logs_item()
    {
        long id = await live.InsertAsync(
            "INSERT INTO logs (service, level, message, context) " +
            "VALUES ('data-pipeline', 'warn', 'replay lagging', '{\"lag_sec\": 12}') RETURNING id");

        JsonElement pushed = await live.NextAsync(LiveEvents.LogWritten, id);

        Assert.Equal(["context", "id", "level", "message", "service", "ts"],
            pushed.EnumerateObject().Select(p => p.Name).Order(StringComparer.Ordinal));
        Assert.Equal("data-pipeline", pushed.GetProperty("service").GetString());
        Assert.Equal("warn", pushed.GetProperty("level").GetString());
        Assert.Equal(12, pushed.GetProperty("context").GetProperty("lag_sec").GetInt32());
    }

    [Fact]
    public async Task A_failing_broadcast_neither_fails_the_bots_write_nor_stops_the_feed()
    {
        live.Broadcaster.FailNext = 1;

        // The bot's write succeeds whatever happens to the push of its row ...
        long lost = await new PostgresPredictionStore(live.DataSource).InsertAsync(Prediction(T0.AddMinutes(3)), null);
        Assert.Equal(1L, await live.ScalarAsync<long>("SELECT count(*) FROM predictions WHERE id = @id", ("id", lost)));

        // ... and the feed carries on with the next row.
        long next = await live.InsertAsync(
            "INSERT INTO logs (service, level, message) VALUES ('ai-service', 'info', 'after the failure') RETURNING id");
        await live.NextAsync(LiveEvents.LogWritten, next);
        Assert.Equal(0, live.CountOf(LiveEvents.PredictionCreated, lost));
    }

    [Fact]
    public async Task A_trade_inserted_already_closed_is_announced_closed_only()
    {
        long id = await live.InsertAsync(
            $"INSERT INTO trades (mint, opened_at, closed_at, pnl, strategy, status) " +
            $"VALUES ('{Mint}', now() - interval '1 minute', now(), -0.5, 'buy', 'closed') RETURNING id");

        await live.NextAsync(LiveEvents.TradeClosed, id);
        await live.NextAsync(LiveEvents.StatsChanged, id, idProperty: "trade_id");
        Assert.Equal(0, live.CountOf(LiveEvents.TradeOpened, id));
    }

    [Fact]
    public async Task A_failing_trade_closed_broadcast_still_sends_its_stats_change()
    {
        live.Broadcaster.FailNext = 1;
        long id = await live.InsertAsync(
            $"INSERT INTO trades (mint, opened_at, closed_at, pnl, strategy, status) " +
            $"VALUES ('{Mint}', now() - interval '1 minute', now(), 0.5, 'buy', 'closed') RETURNING id");

        await live.NextAsync(LiveEvents.StatsChanged, id, idProperty: "trade_id");
        Assert.Equal(0, live.CountOf(LiveEvents.TradeClosed, id));
    }

    [Fact]
    public async Task Debug_log_rows_are_not_pushed()
    {
        long debug = await live.InsertAsync(
            "INSERT INTO logs (service, level, message) VALUES ('data-pipeline', 'debug', 'noise') RETURNING id");
        long info = await live.InsertAsync(
            "INSERT INTO logs (service, level, message) VALUES ('data-pipeline', 'info', 'signal') RETURNING id");

        // Logs go out in id order, so once the later info row is here the debug row would have been too.
        await live.NextAsync(LiveEvents.LogWritten, info);
        Assert.Equal(0, live.CountOf(LiveEvents.LogWritten, debug));
    }
}
