using System.Diagnostics;
using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Options;

namespace Solia.DataPipeline.Tests;

/// <summary>Boots the real host. No database is needed: these point it at unreachable ones.</summary>
public class HostTests
{
    private static WebApplicationFactory<Program> Factory(params (string Key, string Value)[] settings) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(b => b.ConfigureAppConfiguration((_, config) =>
            config.AddInMemoryCollection(
                settings.Select(s => new KeyValuePair<string, string?>(s.Key, s.Value)))));

    [Fact]
    public void Invalid_options_fail_at_startup()
    {
        using WebApplicationFactory<Program> factory = Factory(
            ("SOLIA_DB_CONNECTION", "Host=127.0.0.1;Port=1"),
            ("Pipeline:SnapshotCadenceSeconds", "0"));

        var ex = Assert.Throws<OptionsValidationException>(() => factory.CreateClient());
        Assert.Contains("SnapshotCadenceSeconds", ex.Message);
    }

    [Fact]
    public void Ingest_and_stream_probe_together_fail_at_startup()
    {
        using WebApplicationFactory<Program> factory = Factory(
            ("SOLIA_DB_CONNECTION", "Host=127.0.0.1;Port=1"),
            ("Pipeline:Ingest", "true"),
            ("Pipeline:StreamProbe", "true"));

        var ex = Assert.Throws<OptionsValidationException>(() => factory.CreateClient());
        Assert.Contains("can't both be enabled", ex.Message);
    }

    [Fact]
    public async Task Health_is_unhealthy_when_database_refuses_connections()
    {
        using WebApplicationFactory<Program> factory = Factory(
            ("SOLIA_DB_CONNECTION", "Host=127.0.0.1;Port=1;Timeout=2"));
        HttpClient client = factory.CreateClient();

        HttpResponseMessage response = await client.GetAsync("/health");

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.Equal("Unhealthy", await response.Content.ReadAsStringAsync());
    }

    [Fact]
    public async Task Health_check_is_bounded_when_database_does_not_answer()
    {
        // Non-routable address: the connect would otherwise wait for Npgsql's 15s timeout.
        using WebApplicationFactory<Program> factory = Factory(
            ("SOLIA_DB_CONNECTION", "Host=10.255.255.1;Port=5432;Timeout=15"));
        HttpClient client = factory.CreateClient();

        var sw = Stopwatch.StartNew();
        HttpResponseMessage response = await client.GetAsync("/health");
        sw.Stop();

        Assert.Equal(HttpStatusCode.ServiceUnavailable, response.StatusCode);
        Assert.True(sw.Elapsed < TimeSpan.FromSeconds(10), $"health took {sw.Elapsed}");
    }
}
