using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using Solia.DataPipeline.Hosting;
using Solia.DataPipeline.Ingestion.PumpApi;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Tests.Ingestion;
using static Solia.DataPipeline.Tests.Ingestion.StreamFakes;

namespace Solia.DataPipeline.Tests;

public class StreamProbeServiceTests
{
    [Fact]
    public async Task Logs_each_event_at_debug_and_a_periodic_count_at_information()
    {
        var logger = new CapturingLogger<StreamProbeService>();
        PumpApiStreamSource source = Source(new StreamConnectionState(), [new FakeWebSocket(Step.Frame(BuyA), Step.Frame(BuyB))]);
        var probe = new StreamProbeService(source, Probe(enabled: true), logger) { SummaryInterval = TimeSpan.Zero };

        await probe.StartAsync(CancellationToken.None);
        await WaitUntil(() => logger.Entries.Count(e => e.Message.StartsWith("event #")) == 2);
        await probe.StopAsync(CancellationToken.None);

        Assert.All(logger.Entries.Where(e => e.Message.StartsWith("event #")), e => Assert.Equal(LogLevel.Debug, e.Level));
        Assert.Contains(logger.Entries, e => e.Level == LogLevel.Information && e.Message == "stream probe: 2 events so far");
        Assert.Contains(logger.Entries, e => e.Message == "stream probe stopped after 2 events");
    }

    [Fact]
    public async Task Does_nothing_when_disabled()
    {
        var logger = new CapturingLogger<StreamProbeService>();
        bool connected = false;
        var source = new PumpApiStreamSource(
            () => { connected = true; return new FakeWebSocket(); },
            new PumpApiOptions(),
            new StreamConnectionState(),
            Microsoft.Extensions.Logging.Abstractions.NullLogger<PumpApiStreamSource>.Instance,
            static (_, _) => Task.CompletedTask);
        var probe = new StreamProbeService(source, Probe(enabled: false), logger);

        await probe.StartAsync(CancellationToken.None);
        await probe.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(5));
        await probe.StopAsync(CancellationToken.None);

        Assert.False(connected);
        Assert.Empty(logger.Entries);
    }

    private static Microsoft.Extensions.Options.IOptions<PipelineOptions> Probe(bool enabled) =>
        Microsoft.Extensions.Options.Options.Create(new PipelineOptions { StreamProbe = enabled });

    private static async Task WaitUntil(Func<bool> condition)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        while (!condition())
        {
            await Task.Delay(10, cts.Token);
        }
    }

    private sealed class CapturingLogger<T> : ILogger<T>
    {
        public ConcurrentQueue<(LogLevel Level, string Message)> Entries { get; } = new();

        public IDisposable? BeginScope<TState>(TState state)
            where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter) =>
            Entries.Enqueue((logLevel, formatter(state, exception)));
    }
}
