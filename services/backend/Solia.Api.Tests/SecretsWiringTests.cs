using System.Net;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Npgsql;
using Solia.Api.Hosting;
using Solia.Api.Logging;
using Solia.Bot.Logging;
using Solia.Secrets;

namespace Solia.Api.Tests;

/// <summary>The backend host decrypts secrets.enc at startup and sources the DB connection from it.</summary>
public sealed class SecretsWiringTests : IDisposable
{
    private static readonly string Configured = new NpgsqlConnectionStringBuilder
    {
        Host = "db",
        Database = "solia",
        Username = "solia",
        Password = FakeSecret.New(),
    }.ConnectionString;

    private readonly string _dir = Directory.CreateTempSubdirectory("solia-api-secrets-").FullName;
    private readonly string _key = MasterKey.Generate();
    private readonly string _rpcValue = FakeSecret.New();

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private string WriteSecrets(Dictionary<string, string> secrets)
    {
        string path = Path.Combine(_dir, "secrets.enc");
        SecretsFile.Write(path, secrets, MasterKey.Parse(_key));
        return path;
    }

    [Fact]
    public void Db_password_secret_overrides_the_plaintext_password()
    {
        string fromSecrets = FakeSecret.New();
        var secrets = new SecretsProvider(new Dictionary<string, string> { [SecretNames.DbPassword] = fromSecrets });

        var resolved = new NpgsqlConnectionStringBuilder(
            SecretsServiceCollectionExtensions.ResolveConnectionString(Configured, secrets));

        Assert.Equal(fromSecrets, resolved.Password);
        Assert.Equal("db", resolved.Host);
    }

    [Fact]
    public void Db_connection_secret_replaces_the_configured_string()
    {
        string connection = "Host=secret-host;Password=" + FakeSecret.New();
        var secrets = new SecretsProvider(new Dictionary<string, string>
        {
            [SecretNames.DbConnection] = connection,
            [SecretNames.DbPassword] = FakeSecret.New(),
        });

        Assert.Equal(connection, SecretsServiceCollectionExtensions.ResolveConnectionString(Configured, secrets));
    }

    [Fact]
    public void Without_db_secrets_the_configured_string_is_kept()
    {
        Assert.Equal(Configured, SecretsServiceCollectionExtensions.ResolveConnectionString(Configured, SecretsProvider.Empty));
    }

    [Fact]
    public void The_password_inside_a_db_connection_secret_is_redacted_on_its_own()
    {
        string password = FakeSecret.New();
        var secrets = new SecretsProvider(new Dictionary<string, string>
        {
            [SecretNames.DbConnection] = $"Host=db;Username=solia;Password={password}",
        });
        var redactor = SecretRedactor.For(secrets, SecretsServiceCollectionExtensions.DerivedValues(secrets));
        string reserialised = new NpgsqlConnectionStringBuilder(secrets.Get(SecretNames.DbConnection)).ConnectionString;

        Assert.Equal("auth failed for ***", redactor.Redact($"auth failed for {password}"));
        Assert.DoesNotContain(password, redactor.Redact(reserialised.ToLowerInvariant()), StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void A_malformed_db_connection_secret_derives_nothing()
    {
        var secrets = new SecretsProvider(new Dictionary<string, string> { [SecretNames.DbConnection] = "NotAKeyword=x" });

        Assert.Empty(SecretsServiceCollectionExtensions.DerivedValues(secrets));
        Assert.Empty(SecretsServiceCollectionExtensions.DerivedValues(SecretsProvider.Empty));
    }

    [Fact]
    public async Task Host_decrypts_the_file_feeds_the_data_source_and_redacts_logs()
    {
        string path = WriteSecrets(new Dictionary<string, string>
        {
            [SecretNames.DbConnection] = "Host=secret-host;Port=1;Timeout=1",
            ["rpc_key"] = _rpcValue,
        });
        using WebApplicationFactory<Program> factory = TestApiFactory.Create(
            "Production", [(SecretsProvider.FileKey, path), (MasterKey.EnvironmentKey, _key)]);

        HttpResponseMessage response = await factory.CreateClient().GetAsync("/health/live");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        IServiceProvider services = factory.Services;
        Assert.Equal(_rpcValue, services.GetRequiredService<ISecretsProvider>().Get("rpc_key"));
        Assert.Contains("secret-host", services.GetRequiredService<NpgsqlDataSource>().ConnectionString, StringComparison.Ordinal);
        Assert.IsType<RedactingLoggerFactory>(services.GetRequiredService<ILoggerFactory>());
        Assert.IsType<RedactingBotLogSink>(services.GetRequiredService<IBotLogSink>());
        Assert.Equal("*** / ***", services.GetRequiredService<SecretRedactor>().Redact($"{_rpcValue} / {_key}"));
    }

    [Fact]
    public void Host_fails_to_start_on_a_wrong_master_key()
    {
        string path = WriteSecrets(new Dictionary<string, string> { ["rpc_key"] = _rpcValue });

        AssertStartupFailsWithSecretsException(path, MasterKey.Generate());
    }

    [Fact]
    public void Host_fails_to_start_when_the_configured_secrets_file_is_missing()
    {
        AssertStartupFailsWithSecretsException(Path.Combine(_dir, "typo.enc"), _key);
    }

    private static void AssertStartupFailsWithSecretsException(string path, string key)
    {
        using WebApplicationFactory<Program> factory = TestApiFactory.Create(
            "Production", [(SecretsProvider.FileKey, path), (MasterKey.EnvironmentKey, key)]);

        Exception ex = Assert.ThrowsAny<Exception>(() => factory.CreateClient());

        var chain = new List<Exception>();
        for (Exception? e = ex; e is not null; e = e.InnerException)
        {
            chain.Add(e);
        }
        Assert.Contains(chain, e => e is SecretsException);
    }
}
