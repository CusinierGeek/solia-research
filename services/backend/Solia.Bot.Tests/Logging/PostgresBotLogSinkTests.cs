using Npgsql;
using Solia.Bot.Logging;
using Solia.Bot.Tests.Trading;
using Solia.Shared.Logging;

namespace Solia.Bot.Tests.Logging;

/// <summary>Integration tests against a real PostgreSQL (Testcontainers) — run in CI.</summary>
public class PostgresBotLogSinkTests(PostgresFixture db) : IClassFixture<PostgresFixture>
{
    private PostgresBotLogSink Sink => new(db.DataSource);

    [Fact]
    public async Task Writes_a_bot_row_with_level_message_and_jsonb_context()
    {
        await Sink.WriteAsync(LogSeverity.Info, "position opened", new { tradeId = 7, mint = "mintX" });

        await using NpgsqlCommand cmd = db.DataSource.CreateCommand(
            "SELECT service, level, message, context ->> 'mint' FROM logs WHERE message = 'position opened'");
        await using NpgsqlDataReader row = await cmd.ExecuteReaderAsync();
        Assert.True(await row.ReadAsync());

        Assert.Equal("bot", row.GetString(0));
        Assert.Equal("info", row.GetString(1));
        Assert.Equal("position opened", row.GetString(2));
        Assert.Equal("mintX", row.GetString(3));
    }

    [Fact]
    public async Task A_null_context_is_stored_as_sql_null()
    {
        await Sink.WriteAsync(LogSeverity.Warn, "bot stopping");

        Assert.Equal(1, await db.ScalarAsync<int>(
            "SELECT count(*) FROM logs WHERE message = 'bot stopping' AND level = 'warn' AND context IS NULL"));
    }

    [Fact]
    public async Task Context_keys_are_stored_camel_case_so_rows_join_on_one_spelling()
    {
        await Sink.WriteAsync(LogSeverity.Info, "camel check", new { TradeId = 42L });

        Assert.Equal(1, await db.ScalarAsync<int>(
            "SELECT count(*) FROM logs WHERE message = 'camel check' AND (context ->> 'tradeId') = '42'"));
    }
}
