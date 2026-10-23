using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;

namespace Solia.Api.Tests;

/// <summary>
/// Boots the real host with in-memory settings layered on top. The DB is unreachable by default (the log
/// sink is best-effort, so writes fail fast and are swallowed); a test's own SOLIA_DB_CONNECTION wins.
/// </summary>
internal static class TestApiFactory
{
    public const string UnreachableDb = "Host=127.0.0.1;Port=1;Timeout=1";

    public static WebApplicationFactory<Program> Create(
        string environment, IEnumerable<(string Key, string Value)> settings, Action<IWebHostBuilder>? configure = null) =>
        new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
        {
            builder.UseEnvironment(environment);
            var values = new Dictionary<string, string?> { ["SOLIA_DB_CONNECTION"] = UnreachableDb };
            foreach ((string key, string value) in settings)
            {
                values[key] = value;
            }

            builder.ConfigureAppConfiguration((_, config) => config.AddInMemoryCollection(values));
            configure?.Invoke(builder);
        });
}
