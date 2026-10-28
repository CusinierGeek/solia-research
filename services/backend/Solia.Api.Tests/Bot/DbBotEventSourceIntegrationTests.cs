using System.Collections.Concurrent;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using Solia.Api.Endpoints;
using Solia.Api.Tests.Endpoints;
using Solia.Bot.Domain;
using Solia.Bot.Intake;
using Solia.Core.Ai;
using Solia.Core.Data;

namespace Solia.Api.Tests.Bot;

/// <summary>
/// The live feed (<c>DbBotEventSource</c>) read directly against a real Postgres, the bot itself off: what it
/// emits when inserts commit out of order, when the ai-service is slow or answers garbage, when many tokens are
/// active, and that a prediction is computed only from rows up to its own snapshot. Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class DbBotEventSourceIntegrationTests : PostgresFixture, IAsyncLifetime
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    private static readonly TimeSpan WaitLimit = TimeSpan.FromSeconds(20);

    private readonly ConcurrentQueue<AiPredictRequest> _requests = new();
    private readonly ConcurrentQueue<BotEvent> _events = new();
    private Func<AiPredictRequest, AiPredictResponse> _predict = Strong;
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
                return _predict(request);
            },
        };
        _factory = TestApiFactory.Create("Production",
            [
                ("SOLIA_DB_CONNECTION", ConnectionString),
                ("Bot:PollInterval", "00:00:00.1"),
                ("Bot:ReorderWindow", "00:00:00"),
                ("Bot:MaxMarkAgeSeconds", "3600"),
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

    private static AiPredictResponse Strong(AiPredictRequest request) =>
        new(request.Mint, "v9", [new AiPrediction("1m", 0.8, 0.05, 0.5, 90)]);

    /// <summary>Starts reading the feed in the background, collecting what it emits.</summary>
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

    [Fact]
    public async Task An_insert_that_commits_after_a_higher_id_is_still_emitted()
    {
        await TrackAsync("w", "a", "b");
        await InsertFeaturesAsync(("w", T0, 1.0m));
        StartReading();
        await WaitForAsync(() => Ticks("w").Count == 1); // the cursor is past the existing rows

        // "b" takes the lower id but commits last, after the feed has already read "a"'s higher id.
        await using NpgsqlConnection slow = await DataSource.OpenConnectionAsync();
        await using NpgsqlTransaction tx = await slow.BeginTransactionAsync();
        await using (var insert = new NpgsqlCommand("INSERT INTO features (mint, ts, price) VALUES ('b', @ts, 2.0)", slow, tx))
        {
            insert.Parameters.AddWithValue("ts", T0);
            await insert.ExecuteNonQueryAsync();
        }

        await InsertFeaturesAsync(("a", T0.AddSeconds(1), 1.0m));
        await WaitForAsync(() => Ticks("a").Count == 1);
        await tx.CommitAsync();

        await WaitForAsync(() => Ticks("b").Count == 1);
        Assert.Equal(2.0m, Ticks("b")[0].Price);
        await WaitForAsync(() => Predictions("b").Count == 1);
        Assert.Single(Ticks("a")); // nothing is emitted twice
    }

    [Fact]
    public async Task A_slow_ai_service_does_not_hold_the_price_ticks_back()
    {
        await TrackAsync("slow", "x");
        using var release = new ManualResetEventSlim();
        _predict = request =>
        {
            if (request.Mint == "slow")
            {
                release.Wait(WaitLimit);
            }

            return Strong(request);
        };
        StartReading();

        await InsertFeaturesAsync(("slow", T0, 1.0m));
        await WaitForAsync(() => _requests.Any(r => r.Mint == "slow"));

        // Scoring "slow" hangs; "x"'s tick still flows.
        await InsertFeaturesAsync(("x", T0.AddSeconds(1), 3.0m));
        await WaitForAsync(() => Ticks("x").Count == 1);
        Assert.Empty(Predictions("slow"));

        release.Set();
        await WaitForAsync(() => Predictions("slow").Count == 1 && Predictions("x").Count == 1);
    }

    [Fact]
    public async Task Every_active_token_gets_scored_however_many_update_at_once()
    {
        string[] mints = [.. Enumerable.Range(0, 60).Select(i => $"m{i:00}")];
        await TrackAsync(mints);
        await InsertFeaturesAsync([.. mints.Select((m, i) => (m, T0.AddMilliseconds(i), 1.0m))]);

        StartReading();

        await WaitForAsync(() => _events.OfType<BotEvent.Prediction>().Select(p => p.Mint).Distinct().Count() == mints.Length);
        Assert.Equal(mints.Length, await ScalarAsync<long>("SELECT count(DISTINCT mint) FROM predictions"));
    }

    [Fact]
    public async Task A_malformed_or_failing_scoring_skips_the_token_and_the_feed_keeps_going()
    {
        await TrackAsync("nulls", "boom", "ok");
        _predict = request => request.Mint switch
        {
            "nulls" => new AiPredictResponse(request.Mint, "v9", null!), // 200 with "predictions": null
            "boom" => throw new InvalidOperationException("unexpected"),
            _ => Strong(request),
        };
        StartReading();

        await InsertFeaturesAsync(("nulls", T0, 1.0m), ("boom", T0.AddSeconds(1), 1.0m));
        await WaitForAsync(() => _requests.Count(r => r.Mint is "nulls" or "boom") == 2);
        await InsertFeaturesAsync(("ok", T0.AddSeconds(2), 1.0m));

        await WaitForAsync(() => Predictions("ok").Count == 1);
        Assert.Empty(Predictions("nulls"));
        Assert.Empty(Predictions("boom"));
        Assert.False(_reader.IsCompleted);
    }

    [Fact]
    public async Task A_prediction_is_computed_only_from_rows_up_to_its_own_snapshot()
    {
        await TrackAsync("a");
        await InsertFeaturesAsync(("a", T0, 1.0m), ("a", T0.AddSeconds(5), 2.0m), ("a", T0.AddSeconds(10), 3.0m));

        // The scoring window ends at the queued row, whatever was written after it.
        await using (AsyncServiceScope scope = _factory.Services.CreateAsyncScope())
        {
            var window = await PredictEndpoints.LoadWindowAsync(scope.ServiceProvider.GetRequiredService<SoliaDbContext>(),
                "a", 100, CancellationToken.None, upTo: T0.AddSeconds(5));
            Assert.Equal([1.0, 2.0], window.Select(r => r.Numeric["price"]));
        }

        // End to end: each prediction's window ends at its own row, the one feature_id points at.
        StartReading();
        await WaitForAsync(() => Predictions("a").Count >= 1);
        foreach (ModelPrediction prediction in Predictions("a"))
        {
            decimal price = await ScalarAsync<decimal>("SELECT f.price FROM predictions p JOIN features f ON f.id = p.feature_id WHERE p.id = @id",
                ("id", prediction.PredictionId!.Value));
            Assert.Equal((double)price, prediction.Features["price"]);
            Assert.DoesNotContain("venue", prediction.Features.Keys); // the raw venue string is the ai-service's to encode
            Assert.True(await ScalarAsync<bool>(
                "SELECT f.ts = @ts AND p.ts = @ts FROM predictions p JOIN features f ON f.id = p.feature_id WHERE p.id = @id",
                ("id", prediction.PredictionId!.Value), ("ts", prediction.Ts)));
        }

        // Queued on its newest row.
        Assert.All(_requests, r => Assert.Equal(3.0, ((IReadOnlyList<FeatureRow>)r.Features)[^1].Numeric["price"]));
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

    private async Task InsertFeaturesAsync(params (string Mint, DateTimeOffset Ts, decimal Price)[] rows)
    {
        foreach ((string mint, DateTimeOffset ts, decimal price) in rows)
        {
            await ExecAsync("INSERT INTO features (mint, ts, price) VALUES (@m, @ts, @p)", ("m", mint), ("ts", ts), ("p", price));
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
