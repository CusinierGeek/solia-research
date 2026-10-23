using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Npgsql;
using Solia.Api.Tests.Endpoints;
using Solia.Core.Ai;
using Solia.Secrets;

namespace Solia.Api.Tests.EndToEnd;

/// <summary>
/// The Sprint-4 backend as it runs in production, end to end: a real Postgres (migrations applied, a known
/// seed), the real host with the paper-trading bot enabled, the ai-service stubbed (<see cref="Ai"/>) and a
/// fixed "now". The database password lives only in an AES-256-GCM <c>secrets.enc</c> decrypted with
/// <c>SOLIA_MASTER_KEY</c> — the plaintext connection string has none, so every query the host runs proves the
/// secret was decrypted. A dedicated login role carries that password, so it is a value the logs can be
/// searched for. Every log entry the host emits is captured (<see cref="Logs"/>).
/// </summary>
public sealed class EndToEndFixture : PostgresFixture, IAsyncLifetime
{
    public static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    /// <summary>The host's "now": minutes after the seed, so the bot's price path stays fresh.</summary>
    public static readonly DateTimeOffset Now = T0.AddMinutes(10);

    /// <summary>A token with three feature snapshots, untracked (so only <c>/predict</c> scores it, never the bot).</summary>
    public const string PredictMint = "e2e-predict";

    /// <summary>The token the test feeds to the bot.</summary>
    public const string BotMint = "e2e-bot";

    private const string Role = "solia_e2e";

    private readonly string _dir = Directory.CreateTempSubdirectory("solia-e2e-").FullName;
    private WebApplicationFactory<Program> _factory = null!;

    public StubAiClient Ai { get; } = new();

    public CapturingLoggerProvider Logs { get; } = new();

    public string MasterKeyValue { get; } = MasterKey.Generate();

    public string DbPassword { get; } = FakeSecret.New();

    /// <summary>A non-DB secret stored alongside, which the host must decrypt and never log.</summary>
    public string ApiKey { get; } = FakeSecret.New();

    public const string ApiKeyName = "pumpapi_key";

    /// <summary>The connection string the host is configured with: the role, but no password.</summary>
    public string PasswordlessConnectionString { get; private set; } = null!;

    public IServiceProvider Services => _factory.Services;

    public HttpClient Client { get; private set; } = null!;

    async Task IAsyncLifetime.InitializeAsync()
    {
        await InitializeAsync();
        await ExecAsync($"CREATE ROLE {Role} LOGIN SUPERUSER PASSWORD '{DbPassword}'");
        await SeedAsync();

        PasswordlessConnectionString = new NpgsqlConnectionStringBuilder(ConnectionString)
        {
            Username = Role,
            Password = null,
        }.ConnectionString;
        string secretsPath = Path.Combine(_dir, "secrets.enc");
        SecretsFile.Write(secretsPath,
            new Dictionary<string, string> { [SecretNames.DbPassword] = DbPassword, [ApiKeyName] = ApiKey },
            MasterKey.Parse(MasterKeyValue));

        Ai.Predict = ScoreAsTheChampion;
        _factory = TestApiFactory.Create("Production",
            [
                ("SOLIA_DB_CONNECTION", PasswordlessConnectionString),
                (SecretsProvider.FileKey, secretsPath),
                (MasterKey.EnvironmentKey, MasterKeyValue),
                ("Bot:Enabled", "true"),
                ("Bot:StartingBalance", "100"),
                ("Bot:PollInterval", "00:00:00.1"),
                ("Bot:TimeoutSweepSeconds", "0"),
                ("Bot:MaxMarkAgeSeconds", "3600"),
                ("Logging:LogLevel:Default", "Debug"),
            ],
            builder => builder.ConfigureTestServices(services =>
            {
                services.AddSingleton<ISoliaAiClient>(Ai);
                services.AddSingleton<TimeProvider>(new FixedNow(Now));
                services.AddSingleton<ILoggerProvider>(Logs);
            }));
        Client = _factory.CreateClient(); // starts the host, hence the bot
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        Client.Dispose();
        await _factory.DisposeAsync();
        await DisposeAsync();
        Directory.Delete(_dir, recursive: true);
    }

    private int _botScorings;

    /// <summary>
    /// The bot's token: a strong buy on its first scoring, no edge afterwards. Anything else gets the stub's
    /// default answer.
    /// </summary>
    private AiPredictResponse ScoreAsTheChampion(AiPredictRequest request)
    {
        if (request.Mint != BotMint)
        {
            return new AiPredictResponse(request.Mint, "v3", [new AiPrediction("1m", 0.72, 0.08, 0.35, 81.5)]);
        }

        AiPrediction prediction = Interlocked.Increment(ref _botScorings) == 1
            ? new AiPrediction("1m", 0.8, 0.05, 0.5, 90)
            : new AiPrediction("1m", 0.1, 0.1, 0.0, 50);
        return new AiPredictResponse(request.Mint, "v3", [prediction]);
    }

    private async Task SeedAsync()
    {
        await ExecAsync(
            "INSERT INTO tokens_history (mint, symbol, name, first_seen_at, final_outcome) VALUES " +
            "('e2e-live', 'LIVE', 'Live token', @t, 'pump'), (@p, 'PRED', 'Predict token', @t, NULL), (@b, 'BOT', 'Bot token', @t, NULL)",
            ("t", T0.AddDays(-1)), ("p", PredictMint), ("b", BotMint));
        await ExecAsync(
            "INSERT INTO tokens_live (mint, symbol, last_price, last_market_cap, holders, venue, is_migrated, is_active, updated_at) VALUES " +
            "('e2e-live', 'LIVE', 0.5, 1000, 10, 'pumpfun', FALSE, TRUE, @t), (@p, 'PRED', 1, 1, 1, 'pumpfun', FALSE, FALSE, @t), " +
            "(@b, 'BOT', 1, 1, 1, 'pumpfun', FALSE, TRUE, @t)",
            ("t", T0), ("p", PredictMint), ("b", BotMint));
        await ExecAsync("INSERT INTO models (version, path, is_champion) VALUES ('v3', '/models/v3.onnx', TRUE)");
        for (int i = 0; i < 3; i++)
        {
            await ExecAsync("INSERT INTO features (mint, ts, price, holders) VALUES (@m, @ts, @p, 10)",
                ("m", PredictMint), ("ts", T0.AddSeconds(i)), ("p", 1m + i));
        }

        await ExecAsync(
            "INSERT INTO trades (mint, opened_at, closed_at, entry_price, exit_price, size, pnl, strategy, exit_reason, status) " +
            "VALUES ('e2e-live', @o, @c, 1, 1.5, 2, 1, 'buy', 'tp', 'closed')",
            ("o", T0.AddHours(-2)), ("c", T0.AddHours(-1)));
        await ExecAsync("INSERT INTO logs (ts, service, level, message) VALUES (@t, 'bot', 'info', 'seeded entry')", ("t", T0));
    }

    public async Task ExecAsync(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        await cmd.ExecuteNonQueryAsync();
    }

    /// <summary>Polls until <paramref name="countSql"/> reaches <paramref name="expected"/> (or fails after 20 s).</summary>
    public async Task WaitForAsync(string countSql, long expected, params (string Name, object Value)[] parameters)
    {
        DateTime deadline = DateTime.UtcNow + TimeSpan.FromSeconds(20);
        long actual;
        while ((actual = await ScalarAsync<long>(countSql, parameters)) != expected && DateTime.UtcNow < deadline)
        {
            await Task.Delay(50);
        }

        Assert.True(actual == expected, $"timed out waiting for `{countSql}` = {expected} (last: {actual})");
    }

    /// <summary>A fixed wall clock; timers (the feed's poll delay) still run in real time.</summary>
    private sealed class FixedNow(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;
    }
}
