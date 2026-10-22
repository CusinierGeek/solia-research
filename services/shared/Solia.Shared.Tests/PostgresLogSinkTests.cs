using Npgsql;
using Solia.Shared.Logging;

namespace Solia.Shared.Tests;

/// <summary>
/// The shared `logs` sink's contract without a database: the service name is required, and context
/// keys are camelCase whatever the C# member spelling. The INSERT itself is covered against a real
/// PostgreSQL by the bot's PostgresBotLogSinkTests (CI).
/// </summary>
public class PostgresLogSinkTests
{
    private static NpgsqlDataSource DataSource() => NpgsqlDataSource.Create("Host=localhost;Database=solia");

    [Fact]
    public void Carries_the_service_name_it_was_built_with()
    {
        using NpgsqlDataSource ds = DataSource();

        Assert.Equal("bot", new PostgresLogSink(ds, "bot").Service);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void A_blank_service_name_is_rejected(string service)
    {
        using NpgsqlDataSource ds = DataSource();

        Assert.Throws<ArgumentException>(() => new PostgresLogSink(ds, service));
    }

    [Fact]
    public void A_null_data_source_is_rejected() =>
        Assert.Throws<ArgumentNullException>(() => new PostgresLogSink(null!, "bot"));

    [Fact]
    public void Context_keys_are_camel_case_whatever_the_member_spelling()
    {
        var context = new { TradeId = 7L, tradeMint = "m", EntryPrice = 0.5m };

        Assert.Equal("{\"tradeId\":7,\"tradeMint\":\"m\",\"entryPrice\":0.5}", PostgresLogSink.SerializeContext(context));
    }

    [Fact]
    public void A_null_context_serializes_to_null() => Assert.Null(PostgresLogSink.SerializeContext(null));

    [Fact]
    public void Every_severity_maps_to_a_level_the_logs_table_accepts()
    {
        string[] allowed = ["debug", "info", "warn", "error"];

        Assert.All(Enum.GetValues<LogSeverity>(), s => Assert.Contains(s.ToDbValue(), allowed));
    }
}
