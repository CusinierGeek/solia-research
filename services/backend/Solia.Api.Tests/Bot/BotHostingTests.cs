using System.Net;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using Solia.Api.Bot;
using Solia.Bot;
using Solia.Bot.Configuration;
using Solia.Bot.Intake;
using Solia.Bot.Logging;
using Solia.Bot.Sessions;
using Solia.Core.Data;

namespace Solia.Api.Tests.Bot;

/// <summary>
/// How the API host hosts the bot: the on/off switch, startup validation of the host options. No database is
/// needed (see <see cref="TestApiFactory"/>); the trading itself is covered by <see cref="HostedBotIntegrationTests"/>.
/// </summary>
public class BotHostingTests
{
    private static WebApplicationFactory<Program> Factory(IBotEventSource source, params (string Key, string Value)[] settings) =>
        TestApiFactory.Create("Production", settings, builder => builder.ConfigureTestServices(services =>
        {
            services.AddSingleton(source);
            services.AddSingleton<IBotLogSink>(NullBotLogSink.Instance);
            services.AddSingleton<IBotSessionStore>(new InMemoryBotSessionStore());
        }));

    private static GatedPaperTradingBot Gate(WebApplicationFactory<Program> factory) =>
        factory.Services.GetServices<IHostedService>().OfType<GatedPaperTradingBot>().Single();

    [Fact]
    public async Task The_bot_is_off_by_default_and_its_feed_is_never_built()
    {
        var source = new ThrowingSource();
        await using WebApplicationFactory<Program> factory = Factory(source);
        HttpClient client = factory.CreateClient();

        Assert.Equal(HttpStatusCode.OK, (await client.GetAsync("/health/live")).StatusCode);
        await Gate(factory).ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(Gate(factory).ExecuteTask!.IsCompletedSuccessfully);
        Assert.Equal(0, source.Reads);
    }

    [Fact]
    public async Task When_enabled_the_bot_runs_on_the_registered_feed_with_the_host()
    {
        var source = new CountingSource();
        await using WebApplicationFactory<Program> factory = Factory(source, ("Bot:Enabled", "true"));
        _ = factory.Services;

        await source.Started.Task.WaitAsync(TimeSpan.FromSeconds(10));
        PaperTradingBot bot = factory.Services.GetRequiredService<PaperTradingBot>();
        Assert.NotNull(bot.ExecuteTask);
        Assert.False(bot.ExecuteTask.IsCompleted);

        // Stopping the host stops the bot (its live feed only ends on cancellation).
        await factory.Services.GetRequiredService<IHost>().StopAsync();
        Assert.True(bot.ExecuteTask.IsCompletedSuccessfully);
        Assert.True(source.Cancelled);
    }

    [Theory]
    [InlineData("Bot:Horizon", "2m")]          // not a horizon the predictions table accepts
    [InlineData("Bot:PollInterval", "00:00:00")] // would spin on the database
    [InlineData("Bot:ReorderWindow", "00:02:00")]
    [InlineData("Bot:MaxCommitDelay", "00:00:00")] // would give up on every in-flight insert
    [InlineData("Bot:MaxTicksPerPoll", "0")]
    public void Invalid_host_options_fail_at_startup(string key, string value)
    {
        using WebApplicationFactory<Program> factory = Factory(new ThrowingSource(), (key, value));

        var error = Assert.Throws<OptionsValidationException>(() => factory.CreateClient());
        Assert.Contains(key.Split(':')[1], error.Message);
    }

    [Fact]
    public void The_accepted_horizons_are_exactly_the_predictions_table_check()
    {
        // The migrations are the source of truth: the last one constraining predictions.horizon wins.
        string[] horizons = MigrationChecks.InList("horizon");

        Assert.Equal(horizons, PredictionHorizons.All);
        Assert.All(horizons, h => Assert.True(new BotHostOptions { Horizon = h }.HasKnownHorizon(), h));
        Assert.False(new BotHostOptions { Horizon = "1M" }.HasKnownHorizon());
    }

    [Fact]
    public void The_default_horizon_is_the_swept_entry_rule_horizon()
    {
        // The entry-rule sweep chose 15m (docs/ML-MODEL.md "Choosing the entry rule"), the same as the
        // ai-service's ReferencePolicy default horizon, and the span of the default position timeout.
        var options = new BotHostOptions();

        Assert.Equal("15m", options.Horizon);
        Assert.True(options.HasKnownHorizon());
        Assert.Equal(15 * 60, new BotOptions().PositionTimeoutSeconds);
    }

    [Fact]
    public async Task Stopping_the_host_while_the_bot_is_still_being_built_waits_for_its_run_to_end()
    {
        var source = new CountingSource();
        using var building = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        PaperTradingBot? built = null;
        await using WebApplicationFactory<Program> factory = TestApiFactory.Create("Production", [("Bot:Enabled", "true")],
            builder => builder.ConfigureTestServices(services =>
            {
                services.AddSingleton<IBotEventSource>(source);
                services.AddSingleton<IBotLogSink>(NullBotLogSink.Instance);
                services.AddSingleton<IBotSessionStore>(new InMemoryBotSessionStore());
                services.AddSingleton(sp =>
                {
                    building.Set();
                    release.Wait(TimeSpan.FromSeconds(10)); // a slow construction (stores, feed)
                    return built = ActivatorUtilities.CreateInstance<PaperTradingBot>(sp);
                });
            }));
        _ = factory.Services; // the host starts without waiting for the bot to be built
        Assert.True(building.Wait(TimeSpan.FromSeconds(10)));

        GatedPaperTradingBot gate = Gate(factory);
        Task stop = factory.Services.GetRequiredService<IHost>().StopAsync();
        await Task.Delay(100);
        Assert.False(stop.IsCompleted); // the stop waits for the bot under construction
        release.Set();
        await stop.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.NotNull(built);
        Assert.True(built.ExecuteTask is null || built.ExecuteTask.IsCompleted);
        Assert.True(gate.ExecuteTask!.IsCompletedSuccessfully, gate.ExecuteTask.Status.ToString());
    }

    private sealed class ThrowingSource : IBotEventSource
    {
        public int Reads { get; private set; }

        public IAsyncEnumerable<Solia.Bot.Domain.BotEvent> ReadAsync(CancellationToken cancellationToken = default)
        {
            Reads++;
            throw new InvalidOperationException("the disabled bot must not read its feed");
        }
    }

    /// <summary>A live-like feed: no events, open until cancelled.</summary>
    private sealed class CountingSource : IBotEventSource
    {
        public TaskCompletionSource Started { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public bool Cancelled { get; private set; }

        public async IAsyncEnumerable<Solia.Bot.Domain.BotEvent> ReadAsync(
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            Started.TrySetResult();
            try
            {
                await Task.Delay(Timeout.Infinite, cancellationToken);
            }
            finally
            {
                Cancelled = cancellationToken.IsCancellationRequested;
            }

            yield break;
        }
    }
}
