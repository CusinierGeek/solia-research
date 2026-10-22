using Microsoft.Extensions.Logging.Abstractions;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Logging;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using static Solia.DataPipeline.Tests.Pipeline.PipelineEngineTests;

namespace Solia.DataPipeline.Tests;

public class WorkerTests
{
    [Fact]
    public async Task Ingest_mode_resets_active_tokens_before_ingesting()
    {
        var sink = new RecordingSink();
        Worker worker = Build(sink, ingest: true, Enumerable.Range(0, 12).Select(i => Buy("M", i)));

        await worker.StartAsync(default);
        await worker.ExecuteTask!.WaitAsync(TimeSpan.FromSeconds(10));

        Assert.Equal("reset", sink.Operations.First());
        Assert.Contains("write", sink.Operations);
        Assert.Equal(1, sink.ResetCalls);
    }

    [Fact]
    public async Task Without_ingest_nothing_is_reset()
    {
        var sink = new RecordingSink();
        Worker worker = Build(sink, ingest: false, []);

        await worker.StartAsync(default);
        await worker.StopAsync(default);

        Assert.Equal(0, sink.ResetCalls);
        Assert.Empty(sink.Operations);
    }

    private static Worker Build(RecordingSink sink, bool ingest, IEnumerable<RawEvent> events)
    {
        var options = new PipelineOptions { Ingest = ingest, SnapshotCadenceSeconds = 10 };
        PipelineEngine engine = Engine(new ListSource(events), sink);
        return new Worker(NullLogger<Worker>.Instance, new NoOpLogSink(), engine, Microsoft.Extensions.Options.Options.Create(options));
    }

    private sealed class NoOpLogSink : ILogSink
    {
        public Task WriteAsync(LogSeverity level, string message, object? context = null, CancellationToken ct = default) =>
            Task.CompletedTask;
    }
}
