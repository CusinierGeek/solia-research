using MudBlazor.Services;

namespace Solia.Dashboard.Tests;

/// <summary>
/// A bUnit context wired for MudBlazor components: its services, and loose JS interop (MudBlazor's
/// popover/scroll interop calls return defaults instead of failing the render). Disposed asynchronously,
/// since some MudBlazor services are only <see cref="IAsyncDisposable"/>.
/// </summary>
public abstract class MudTestContext : BunitContext, IAsyncLifetime
{
    protected MudTestContext()
    {
        JSInterop.Mode = JSRuntimeMode.Loose;
        Services.AddMudServices();
    }

    Task IAsyncLifetime.InitializeAsync() => Task.CompletedTask;

    async Task IAsyncLifetime.DisposeAsync() => await DisposeAsync();
}
