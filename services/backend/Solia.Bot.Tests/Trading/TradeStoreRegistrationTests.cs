using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Bot.Configuration;
using Solia.Bot.Policy;
using Solia.Bot.Trading;

namespace Solia.Bot.Tests.Trading;

public class TradeStoreRegistrationTests
{
    private static ServiceProvider Build(Dictionary<string, string?> values)
    {
        IConfiguration config = new ConfigurationBuilder().AddInMemoryCollection(values).Build();
        return new ServiceCollection().AddPostgresTradeStore(config).BuildServiceProvider();
    }

    [Fact]
    public void Store_and_counter_resolve_to_one_postgres_store()
    {
        using ServiceProvider provider = Build(new() { ["SOLIA_DB_CONNECTION"] = "Host=db;Database=solia;Username=u;Password=p" });

        var store = Assert.IsType<PostgresTradeStore>(provider.GetRequiredService<ITradeStore>());
        Assert.Same(store, provider.GetRequiredService<IClosedTradeCounter>());
    }

    [Fact]
    public void The_connection_string_comes_from_the_shared_db_settings()
    {
        using ServiceProvider provider = Build(new() { ["POSTGRES_HOST"] = "pg", ["POSTGRES_PORT"] = "6543" });

        var builder = new NpgsqlConnectionStringBuilder(provider.GetRequiredService<NpgsqlDataSource>().ConnectionString);
        Assert.Equal("pg", builder.Host);
        Assert.Equal(6543, builder.Port);
    }

    [Fact]
    public void An_invalid_db_setting_fails_when_the_store_is_resolved()
    {
        using ServiceProvider provider = Build(new() { ["POSTGRES_PORT"] = "not-a-port" });

        Assert.Throws<InvalidOperationException>(() => provider.GetRequiredService<ITradeStore>());
    }

    [Theory]
    [InlineData(TradeSignal.Buy, "buy")]
    [InlineData(TradeSignal.Sell, "sell")]
    [InlineData(TradeSignal.Hold, "hold")]
    public void Strategy_codes_match_the_check_constraint(TradeSignal signal, string code) =>
        Assert.Equal(code, signal.ToCode());

    [Fact]
    public void An_undefined_signal_has_no_code() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => ((TradeSignal)42).ToCode());
}
