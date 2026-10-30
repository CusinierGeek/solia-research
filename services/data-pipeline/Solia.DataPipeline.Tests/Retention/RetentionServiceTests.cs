using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Time.Testing;
using Solia.DataPipeline.Retention;
using MsOptions = Microsoft.Extensions.Options.Options;

namespace Solia.DataPipeline.Tests.Retention;

public class RetentionServiceTests
{
    private readonly FakeRetentionStore _store = new();
    private readonly FakeTimeProvider _time = new(new DateTimeOffset(2026, 10, 30, 12, 0, 0, TimeSpan.Zero));

    private RetentionService Service(RetentionOptions options)
    {
        var job = new RetentionJob(_store, new RecordingLogSink(), MsOptions.Create(options), _time,
            NullLogger<RetentionJob>.Instance);
        return new RetentionService(job, MsOptions.Create(options), _time, NullLogger<RetentionService>.Instance);
    }

    private static async Task Until(Func<bool> condition)
    {
        for (int i = 0; i < 200 && !condition(); i++)
        {
            await Task.Delay(10);
        }

        Assert.True(condition());
    }

    [Fact]
    public async Task Disabled_never_purges()
    {
        using RetentionService service = Service(new RetentionOptions { Enabled = false });

        await service.StartAsync(default);
        await service.ExecuteTask!;
        _time.Advance(TimeSpan.FromDays(1));

        Assert.Empty(_store.Cutoffs);
    }

    [Fact]
    public async Task Purges_at_startup_then_every_interval()
    {
        using RetentionService service = Service(new RetentionOptions { Enabled = true, IntervalMinutes = 30 });

        await service.StartAsync(default);
        await Until(() => _store.Deletes.Count == 3); // one (empty) batch per table

        _time.Advance(TimeSpan.FromMinutes(29));
        await Task.Delay(50);
        Assert.Equal(3, _store.Deletes.Count);

        _time.Advance(TimeSpan.FromMinutes(1));
        await Until(() => _store.Deletes.Count == 6);

        await service.StopAsync(default);
        Assert.True(service.ExecuteTask!.IsCompletedSuccessfully);
    }

    [Fact]
    public async Task A_failed_run_is_retried_on_the_next_tick()
    {
        _store.Unreachable = true;
        using RetentionService service = Service(new RetentionOptions { Enabled = true, IntervalMinutes = 10 });

        await service.StartAsync(default);
        await Task.Delay(50);
        Assert.False(service.ExecuteTask!.IsCompleted); // still running

        _store.Unreachable = false;
        _store.With(RetainedTable.Logs, 5);
        _time.Advance(TimeSpan.FromMinutes(10));
        await Until(() => _store.Remaining(RetainedTable.Logs) == 0);

        await service.StopAsync(default);
    }
}
