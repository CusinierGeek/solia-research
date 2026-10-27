using System.Collections.Concurrent;
using Solia.Dashboard.Live;

namespace Solia.Dashboard.Tests.Live;

/// <summary>An <see cref="ILiveConnection"/> the test drives: raises hub events and connection transitions by hand.</summary>
internal sealed class FakeLiveConnection : ILiveConnection
{
    private readonly ConcurrentDictionary<string, Func<object?, Task>> _handlers = new();
    private readonly ConcurrentQueue<Func<CancellationToken, Task>> _starts = new();

    public event Func<Exception?, Task>? Reconnecting;

    public event Func<string?, Task>? Reconnected;

    public event Func<Exception?, Task>? Closed;

    public int StartCalls;

    public int StopCalls;

    public bool Disposed { get; private set; }

    /// <summary>What <see cref="StopAsync"/> waits for (completes at once by default).</summary>
    public Func<CancellationToken, Task> OnStop { get; set; } = _ => Task.CompletedTask;

    /// <summary>Scripts the next <see cref="StartAsync"/> outcomes in order; once used up, a start succeeds.</summary>
    public void EnqueueStart(Func<CancellationToken, Task> start) => _starts.Enqueue(start);

    public void FailNextStart(int times = 1)
    {
        for (int i = 0; i < times; i++)
        {
            EnqueueStart(_ => Task.FromException(new HttpRequestException("backend unreachable")));
        }
    }

    public IDisposable On<T>(string eventName, Func<T, Task> handler)
    {
        _handlers[eventName] = payload => handler((T)payload!);
        return new Unregister(() => _handlers.TryRemove(eventName, out _));
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref StartCalls);
        return _starts.TryDequeue(out Func<CancellationToken, Task>? start) ? start(cancellationToken) : Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        Interlocked.Increment(ref StopCalls);
        return OnStop(cancellationToken);
    }

    public ValueTask DisposeAsync()
    {
        Disposed = true;
        return ValueTask.CompletedTask;
    }

    public bool Handles(string eventName) => _handlers.ContainsKey(eventName);

    public Task RaiseAsync(string eventName, object? payload) => _handlers[eventName](payload);

    public Task RaiseReconnectingAsync() => Reconnecting?.Invoke(new IOException("connection reset")) ?? Task.CompletedTask;

    public Task RaiseReconnectedAsync() => Reconnected?.Invoke("conn-2") ?? Task.CompletedTask;

    public Task RaiseClosedAsync() => Closed?.Invoke(null) ?? Task.CompletedTask;

    private sealed class Unregister(Action action) : IDisposable
    {
        public void Dispose() => action();
    }
}
