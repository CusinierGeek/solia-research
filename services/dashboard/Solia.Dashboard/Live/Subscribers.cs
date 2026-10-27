namespace Solia.Dashboard.Live;

/// <summary>
/// The handlers subscribed to one kind of notification. Copy-on-write, so dispatching never holds a lock; a
/// disposed subscription is never called again, even by a dispatch that took its snapshot just before. A handler
/// that throws is logged and doesn't affect the others.
/// </summary>
internal sealed class Subscribers<T>(string name, ILogger logger)
{
    private readonly Lock _gate = new();

    private Subscription[] _items = [];

    internal int Count => Volatile.Read(ref _items).Length;

    public IDisposable Add(Func<T, Task> handler)
    {
        ArgumentNullException.ThrowIfNull(handler);
        Subscription subscription = new(this, handler);
        lock (_gate)
        {
            _items = [.. _items, subscription];
        }

        return subscription;
    }

    /// <summary>Calls every live handler with <paramref name="value"/>, concurrently; completes when all have.</summary>
    public Task InvokeAsync(T value)
    {
        Subscription[] items = Volatile.Read(ref _items);
        return items.Length switch
        {
            0 => Task.CompletedTask,
            1 => items[0].InvokeAsync(value),
            _ => Task.WhenAll(Array.ConvertAll(items, s => s.InvokeAsync(value))),
        };
    }

    /// <summary>
    /// Starts every live handler with <paramref name="value"/> in subscription order without waiting for them. Never
    /// faults: handler failures are logged by <see cref="InvokeAsync"/>.
    /// </summary>
    public void Publish(T value) => _ = InvokeAsync(value);

    private void Remove(Subscription subscription)
    {
        lock (_gate)
        {
            _items = Array.FindAll(_items, s => s != subscription);
        }
    }

    private sealed class Subscription(Subscribers<T> owner, Func<T, Task> handler) : IDisposable
    {
        private volatile Func<T, Task>? _handler = handler;

        public async Task InvokeAsync(T value)
        {
            Func<T, Task>? current = _handler;
            if (current is null)
            {
                return;
            }

            try
            {
                await current(value);
            }
            catch (Exception ex)
            {
                owner.LogFailure(ex);
            }
        }

        public void Dispose()
        {
            if (_handler is not null)
            {
                _handler = null;
                owner.Remove(this);
            }
        }
    }

    private void LogFailure(Exception ex) => logger.LogWarning(ex, "live: a {Name} handler failed", name);
}
