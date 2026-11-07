using System.Collections.Concurrent;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Api.Tests.Endpoints;
using Solia.Bot.Domain;
using Solia.Bot.Intake;
using Solia.Core.Ai;

namespace Solia.Api.Tests.Bot;

/// <summary>The live feed with a scoring floor on, read against a real Postgres, the bot itself off. Run in CI.</summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class ScoringFloorIntegrationTests : PostgresFixture, IAsyncLifetime
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    private static readonly TimeSpan WaitLimit = TimeSpan.FromSeconds(20);

    private readonly ConcurrentQueue<AiPredictRequest> _requests = new();
    private readonly ConcurrentQueue<BotEvent> _events = new();
    private WebApplicationFactory<Program> _factory = null!;
    private CancellationTokenSource _reading = null!;
    private Task _reader = Task.CompletedTask;

    async Task IAsyncLifetime.InitializeAsync()
    {
        await InitializeAsync();
        await ExecAsync("INSERT INTO models (version, path, is_champion) VALUES ('v9', '/models/v9.onnx', TRUE)");
        var ai = new StubAiClient
        {
            Predict = request =>
            {
                _requests.Enqueue(request);
                return new AiPredictResponse(request.Mint, "v9", [new AiPrediction("1m", 0.8, 0.05, 0.5, 90)]);
            },
        };
        _factory = TestApiFactory.Create("Production",
            [
                ("SOLIA_DB_CONNECTION", ConnectionString),
                ("Bot:PollInterval", "00:00:00.1"),
                ("Bot:ReorderWindow", "00:00:00"),
                ("Bot:MaxMarkAgeSeconds", "3600"),
                ("Bot:Horizon", "1m"),
                ("Bot:MinScoreTraders", "20"),
                ("Bot:MinScoreLiquiditySol", "5"),
            ],
            builder => builder.ConfigureTestServices(services =>
            {
                services.AddSingleton<ISoliaAiClient>(ai);
                services.AddSingleton<TimeProvider>(new FixedNow(T0.AddMinutes(10)));
            }));
    }

    async Task IAsyncLifetime.DisposeAsync()
    {
        if (_reading is not null)
        {
            await _reading.CancelAsync();
            await _reader.ContinueWith(_ => { }, TaskScheduler.Default);
            _reading.Dispose();
        }

        await _factory.DisposeAsync();
        await DisposeAsync();
    }

    [Fact]
    public async Task Rows_under_the_floor_are_not_scored_but_their_ticks_flow()
    {
        await TrackAsync("quiet", "thin", "unknown", "live");
        await InsertAsync(("quiet", T0, 19, 50m), ("thin", T0.AddSeconds(1), 50, 4.9m), ("unknown", T0.AddSeconds(2), 50, null),
            ("live", T0.AddSeconds(3), 20, 5m));
        StartReading();

        await WaitForAsync(() => Predictions("live").Count == 1 && Ticks("unknown").Count == 1);

        Assert.DoesNotContain(_requests, r => r.Mint is "quiet" or "thin" or "unknown");
        Assert.Single(Ticks("quiet")); // an open position on it is still managed
        Assert.Single(Ticks("thin"));
        Assert.Empty(Predictions("quiet"));
    }

    [Fact]
    public async Task A_token_is_scored_from_its_first_row_above_the_floor_and_skipped_again_below_it()
    {
        await TrackAsync("waking", "marker");
        await InsertAsync(("waking", T0, 1, 50m));
        StartReading();
        await WaitForAsync(() => Ticks("waking").Count == 1);
        Assert.Empty(_requests);

        await InsertAsync(("waking", T0.AddSeconds(10), 40, 50m));
        await WaitForAsync(() => Predictions("waking").Count == 1);

        // The window sent is up to the row that cleared the floor, earlier rows included.
        Assert.Equal(2, ((IReadOnlyList<FeatureRow>)_requests.Single(r => r.Mint == "waking").Features).Count);

        await InsertAsync(("waking", T0.AddSeconds(20), 1, 50m));
        await WaitForAsync(() => Ticks("waking").Count == 3);
        await InsertAsync(("marker", T0.AddSeconds(21), 40, 50m));
        await WaitForAsync(() => Predictions("marker").Count == 1);

        Assert.Single(Predictions("waking"));
    }

    private void StartReading()
    {
        IBotEventSource source = _factory.Services.GetRequiredService<IBotEventSource>();
        _reading = new CancellationTokenSource();
        _reader = Task.Run(async () =>
        {
            await foreach (BotEvent botEvent in source.ReadAsync(_reading.Token))
            {
                _events.Enqueue(botEvent);
            }
        });
    }

    private List<PriceTick> Ticks(string mint) =>
        [.. _events.OfType<BotEvent.Price>().Select(p => p.Value).Where(t => t.Mint == mint)];

    private List<ModelPrediction> Predictions(string mint) =>
        [.. _events.OfType<BotEvent.Prediction>().Select(p => p.Value).Where(p => p.Mint == mint)];

    private async Task TrackAsync(params string[] mints)
    {
        foreach (string mint in mints)
        {
            await ExecAsync("INSERT INTO tokens_history (mint, first_seen_at) VALUES (@m, @t)", ("m", mint), ("t", T0));
            await ExecAsync("INSERT INTO tokens_live (mint, is_active) VALUES (@m, TRUE)", ("m", mint));
        }
    }

    private async Task InsertAsync(params (string Mint, DateTimeOffset Ts, int Traders, decimal? Liquidity)[] rows)
    {
        foreach ((string mint, DateTimeOffset ts, int traders, decimal? liquidity) in rows)
        {
            await ExecAsync("INSERT INTO features (mint, ts, price, unique_traders, liquidity) VALUES (@m, @ts, 1.0, @u, CAST(@l AS NUMERIC))",
                ("m", mint), ("ts", ts), ("u", traders), ("l", (object?)liquidity ?? DBNull.Value));
        }
    }

    private async Task WaitForAsync(Func<bool> condition)
    {
        DateTime deadline = DateTime.UtcNow + WaitLimit;
        while (!condition())
        {
            if (_reader.IsFaulted)
            {
                await _reader; // surface the feed's failure
            }

            Assert.True(DateTime.UtcNow < deadline, "timed out waiting for the feed");
            await Task.Delay(50);
        }
    }

    private async Task ExecAsync(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        await cmd.ExecuteNonQueryAsync();
    }

    /// <summary>A fixed wall clock; timers (the feed's poll delay) still run in real time.</summary>
    private sealed class FixedNow(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;
    }
}
