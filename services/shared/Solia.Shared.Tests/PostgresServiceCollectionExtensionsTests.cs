using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Shared.Data;
using Solia.Shared.Logging;

namespace Solia.Shared.Tests;

public class PostgresServiceCollectionExtensionsTests
{
    private static IConfiguration Config(string connection) =>
        new ConfigurationBuilder()
            .AddInMemoryCollection([new KeyValuePair<string, string?>("SOLIA_DB_CONNECTION", connection)])
            .Build();

    [Fact]
    public void Registers_one_shared_data_source_and_a_postgres_sink_for_the_service()
    {
        var services = new ServiceCollection();
        IConfiguration config = Config("Host=localhost;Database=solia");

        services.AddSoliaDataSource(config);
        services.AddPostgresLogSink(config, "backend");
        using ServiceProvider provider = services.BuildServiceProvider();

        var sink = Assert.IsType<PostgresLogSink>(provider.GetRequiredService<ILogSink>());
        Assert.Equal("backend", sink.Service);
        Assert.Single(services, d => d.ServiceType == typeof(NpgsqlDataSource));
    }

    [Fact]
    public void A_data_source_registered_first_wins()
    {
        using NpgsqlDataSource own = NpgsqlDataSource.Create("Host=own;Database=solia");
        var services = new ServiceCollection();
        services.AddSingleton(own);

        services.AddSoliaDataSource(Config("Host=localhost;Database=solia"));
        using ServiceProvider provider = services.BuildServiceProvider();

        Assert.Same(own, provider.GetRequiredService<NpgsqlDataSource>());
    }

    [Fact]
    public void Connection_string_is_read_when_resolved_not_when_registered()
    {
        var config = new ConfigurationManager();
        var services = new ServiceCollection();
        services.AddSoliaDataSource(config);

        config.AddInMemoryCollection([new KeyValuePair<string, string?>("SOLIA_DB_CONNECTION", "Host=late;Database=solia")]);
        using ServiceProvider provider = services.BuildServiceProvider();

        Assert.Contains("Host=late", provider.GetRequiredService<NpgsqlDataSource>().ConnectionString);
    }

    [Fact]
    public void Malformed_connection_string_falls_back_to_a_null_sink_instead_of_failing()
    {
        var services = new ServiceCollection();

        services.AddPostgresLogSink(Config("NotAKeyword=oops"), "backend");
        using ServiceProvider provider = services.BuildServiceProvider();

        Assert.Same(NullLogSink.Instance, provider.GetRequiredService<ILogSink>());
    }

    [Fact]
    public void The_fallback_message_names_the_setting_but_never_echoes_the_connection_string()
    {
        string password = "fake-" + Guid.NewGuid().ToString("N");
        string connection = $"Host=db;Password={password};Port=not-a-number";
        ArgumentException rejected = Assert.ThrowsAny<ArgumentException>(() => NpgsqlDataSource.Create(connection));

        string message = PostgresServiceCollectionExtensions.DisabledMessage("backend", rejected);

        Assert.StartsWith("[backend] `logs` table sink disabled", message, StringComparison.Ordinal);
        Assert.DoesNotContain(password, message, StringComparison.Ordinal);
        Assert.DoesNotContain(rejected.Message, message, StringComparison.Ordinal);
    }

    [Fact]
    public void A_non_configuration_failure_building_the_data_source_propagates()
    {
        var services = new ServiceCollection();
        services.AddSingleton<NpgsqlDataSource>(_ => throw new InvalidOperationException("secrets could not be decrypted"));

        services.AddPostgresLogSink(Config("Host=x"), "backend");
        using ServiceProvider provider = services.BuildServiceProvider();

        var ex = Assert.Throws<InvalidOperationException>(() => provider.GetRequiredService<ILogSink>());
        Assert.Equal("secrets could not be decrypted", ex.Message);
    }

    [Fact]
    public void A_blank_service_name_is_rejected() =>
        Assert.Throws<ArgumentException>(() => new ServiceCollection().AddPostgresLogSink(Config("Host=x"), " "));

    [Fact]
    public async Task Null_sink_accepts_and_drops_writes() =>
        await NullLogSink.Instance.WriteAsync(LogSeverity.Error, "dropped", new { Any = 1 });
}
