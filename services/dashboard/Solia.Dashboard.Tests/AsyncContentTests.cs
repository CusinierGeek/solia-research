using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using MudBlazor.Services;
using Solia.Dashboard.Components.State;

namespace Solia.Dashboard.Tests;

public class AsyncContentTests : MudTestContext
{
    private IRenderedComponent<AsyncContent<string[]>> RenderWith(Func<CancellationToken, Task<string[]>> load) =>
        Render<AsyncContent<string[]>>(p => p
            .Add(c => c.Load, load)
            .Add(c => c.IsEmpty, items => items.Length == 0)
            .Add(c => c.EmptyMessage, "Nothing here.")
            .Add(c => c.ErrorMessage, "Load failed.")
            .Add(c => c.ChildContent, (RenderFragment<string[]>)(items => b =>
                b.AddMarkupContent(0, $"<ul id=\"items\">{string.Concat(items.Select(i => $"<li>{i}</li>"))}</ul>"))));

    [Fact]
    public void Shows_loading_until_the_data_arrives_then_the_content()
    {
        var pending = new TaskCompletionSource<string[]>();
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => pending.Task);

        Assert.Equal(AsyncLoadState.Loading, cut.Instance.State);
        Assert.NotNull(cut.Find(".solia-loading"));

        pending.SetResult(["a", "b"]);

        cut.WaitForAssertion(() => Assert.Equal(2, cut.FindAll("#items li").Count));
        Assert.Equal(AsyncLoadState.Content, cut.Instance.State);
        Assert.Empty(cut.FindAll(".solia-loading"));
    }

    [Fact]
    public void Empty_result_shows_the_empty_state()
    {
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => Task.FromResult(Array.Empty<string>()));

        cut.WaitForAssertion(() => Assert.Contains("Nothing here.", cut.Find(".solia-empty").TextContent));
        Assert.Equal(AsyncLoadState.Empty, cut.Instance.State);
    }

    [Fact]
    public void Emptiness_is_rechecked_on_every_render()
    {
        bool overlaid = false;
        IRenderedComponent<AsyncContent<string[]>> cut = Render<AsyncContent<string[]>>(p => p
            .Add(c => c.Load, _ => Task.FromResult(Array.Empty<string>()))
            .Add(c => c.IsEmpty, items => items.Length == 0 && !overlaid)
            .Add(c => c.EmptyMessage, "Nothing here.")
            .Add(c => c.ChildContent, (RenderFragment<string[]>)(_ => b => b.AddMarkupContent(0, "<p id=\"shown\">shown</p>"))));
        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Empty, cut.Instance.State));

        overlaid = true; // e.g. a row pushed live on top of an empty page
        cut.Render();

        Assert.Equal(AsyncLoadState.Content, cut.Instance.State);
        Assert.NotNull(cut.Find("#shown"));
    }

    [Fact]
    public void Null_result_shows_the_empty_state()
    {
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => Task.FromResult<string[]>(null!));

        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Empty, cut.Instance.State));
    }

    [Fact]
    public void Failure_shows_the_error_state_without_the_exception_text()
    {
        IRenderedComponent<AsyncContent<string[]>> cut =
            RenderWith(_ => Task.FromException<string[]>(new HttpRequestException("connection refused: secret-host")));

        cut.WaitForAssertion(() => Assert.Contains("Load failed.", cut.Find(".solia-error").TextContent));
        Assert.Equal(AsyncLoadState.Error, cut.Instance.State);
        Assert.DoesNotContain("secret-host", cut.Markup);
    }

    [Fact]
    public void Synchronous_throw_in_load_is_also_an_error_state()
    {
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => throw new InvalidOperationException("boom"));

        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Error, cut.Instance.State));
    }

    [Fact]
    public void Retry_after_a_failure_reloads_and_shows_the_content()
    {
        int calls = 0;
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => ++calls == 1
            ? Task.FromException<string[]>(new HttpRequestException("down"))
            : Task.FromResult(new[] { "recovered" }));
        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Error, cut.Instance.State));

        cut.Find("button.solia-error__retry").Click();

        cut.WaitForAssertion(() => Assert.Equal("recovered", cut.Find("#items li").TextContent));
        Assert.Equal(2, calls);
    }

    [Fact]
    public async Task Reload_discards_the_stale_attempt_and_cancels_its_token()
    {
        var first = new TaskCompletionSource<string[]>();
        var tokens = new List<CancellationToken>();
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(ct =>
        {
            tokens.Add(ct);
            return tokens.Count == 1 ? first.Task : Task.FromResult(new[] { "fresh" });
        });

        await cut.InvokeAsync(cut.Instance.ReloadAsync);
        first.SetResult(["stale"]);

        cut.WaitForAssertion(() => Assert.Equal("fresh", cut.Find("#items li").TextContent));
        Assert.True(tokens[0].IsCancellationRequested);
        Assert.False(tokens[1].IsCancellationRequested);
    }

    [Fact]
    public async Task Prerendering_shows_the_loading_state_without_calling_the_backend()
    {
        // HtmlRenderer renders like the server prerender pass: OnAfterRender never runs.
        await using ServiceProvider services = new ServiceCollection().AddLogging().AddMudServices().BuildServiceProvider();
        await using var renderer = new HtmlRenderer(services, services.GetRequiredService<ILoggerFactory>());
        int calls = 0;
        Func<CancellationToken, Task<string[]>> load = _ =>
        {
            calls++;
            return Task.FromResult(new[] { "a" });
        };

        string html = await renderer.Dispatcher.InvokeAsync(async () =>
        {
            var parameters = new Dictionary<string, object?>
            {
                [nameof(AsyncContent<string[]>.Load)] = load,
                [nameof(AsyncContent<string[]>.ChildContent)] = (RenderFragment<string[]>)(_ => _ => { }),
            };
            return (await renderer.RenderComponentAsync<AsyncContent<string[]>>(ParameterView.FromDictionary(parameters)))
                .ToHtmlString();
        });

        Assert.Contains("solia-loading", html);
        Assert.Equal(0, calls);
    }

    [Fact]
    public async Task Refresh_keeps_the_loaded_content_until_the_new_data_arrives()
    {
        var second = new TaskCompletionSource<string[]>();
        int calls = 0;
        IRenderedComponent<AsyncContent<string[]>> cut =
            RenderWith(_ => ++calls == 1 ? Task.FromResult(new[] { "old" }) : second.Task);
        cut.WaitForAssertion(() => Assert.Equal("old", cut.Find("#items li").TextContent));

        Task reload = cut.InvokeAsync(cut.Instance.ReloadAsync);

        Assert.Equal(AsyncLoadState.Content, cut.Instance.State);
        Assert.Equal("old", cut.Find("#items li").TextContent);
        Assert.Empty(cut.FindAll(".solia-loading"));

        second.SetResult(["new"]);
        await reload;
        cut.WaitForAssertion(() => Assert.Equal("new", cut.Find("#items li").TextContent));
    }

    [Fact]
    public void Retry_after_a_failure_shows_the_loading_state_meanwhile()
    {
        var second = new TaskCompletionSource<string[]>();
        int calls = 0;
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => ++calls == 1
            ? Task.FromException<string[]>(new HttpRequestException("down"))
            : second.Task);
        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Error, cut.Instance.State));

        cut.Find("button.solia-error__retry").Click();

        cut.WaitForAssertion(() => Assert.NotNull(cut.Find(".solia-loading")));
        Assert.Equal(AsyncLoadState.Loading, cut.Instance.State);
    }

    [Fact]
    public void Changing_the_reload_key_reloads()
    {
        var keys = new List<int>();
        int key = 1;
        IRenderedComponent<AsyncContent<string[]>> cut = Render<AsyncContent<string[]>>(p => p
            .Add(c => c.Load, _ =>
            {
                keys.Add(key);
                return Task.FromResult(new[] { $"range {key}" });
            })
            .Add(c => c.ReloadKey, key)
            .Add(c => c.ChildContent, (RenderFragment<string[]>)(items => b => b.AddMarkupContent(0, $"<p id=\"v\">{items[0]}</p>"))));
        cut.WaitForAssertion(() => Assert.Equal("range 1", cut.Find("#v").TextContent));

        key = 2;
        cut.Render(p => p.Add(c => c.ReloadKey, key));

        cut.WaitForAssertion(() => Assert.Equal("range 2", cut.Find("#v").TextContent));
        Assert.Equal([1, 2], keys);
    }

    [Fact]
    public void Re_rendering_with_the_same_key_and_load_does_not_reload()
    {
        int calls = 0;
        Func<CancellationToken, Task<string[]>> load = _ =>
        {
            calls++;
            return Task.FromResult(new[] { "a" });
        };
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(load);
        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Content, cut.Instance.State));

        cut.Render(p => p.Add(c => c.Load, load).Add(c => c.EmptyMessage, "changed"));

        Assert.Equal(1, calls);
    }

    [Fact]
    public void Swapping_the_load_method_reloads()
    {
        IRenderedComponent<AsyncContent<string[]>> cut = RenderWith(_ => Task.FromResult(new[] { "first" }));
        cut.WaitForAssertion(() => Assert.Equal("first", cut.Find("#items li").TextContent));

        cut.Render(p => p.Add(c => c.Load, _ => Task.FromResult(new[] { "second" })));

        cut.WaitForAssertion(() => Assert.Equal("second", cut.Find("#items li").TextContent));
    }

    [Fact]
    public void A_timeout_not_caused_by_the_component_is_an_error_state()
    {
        IRenderedComponent<AsyncContent<string[]>> cut =
            RenderWith(_ => Task.FromException<string[]>(new TaskCanceledException("HttpClient.Timeout elapsed")));

        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Error, cut.Instance.State));
    }

    [Fact]
    public async Task Disposing_cancels_the_in_flight_load()
    {
        CancellationToken token = default;
        RenderWith(ct =>
        {
            token = ct;
            return new TaskCompletionSource<string[]>().Task;
        });

        await DisposeComponentsAsync();

        Assert.True(token.IsCancellationRequested);
    }

    [Fact]
    public void A_failure_recognised_as_invalid_shows_its_message_as_a_warning_without_retry()
    {
        IRenderedComponent<AsyncContent<string[]>> cut = Render<AsyncContent<string[]>>(p => p
            .Add(c => c.Load, _ => Task.FromException<string[]>(new ArgumentException("bad")))
            .Add(c => c.InvalidMessage, ex => ex is ArgumentException ? $"Fix it: {ex.Message}" : null)
            .Add(c => c.ChildContent, (RenderFragment<string[]>)(_ => _ => { })));

        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Invalid, cut.Instance.State));
        Assert.Contains("Fix it: bad", cut.Find(".solia-invalid").TextContent);
        Assert.Empty(cut.FindAll(".solia-error"));
        Assert.Empty(cut.FindAll("button"));
    }

    [Fact]
    public void A_failure_not_recognised_as_invalid_is_still_an_error()
    {
        IRenderedComponent<AsyncContent<string[]>> cut = Render<AsyncContent<string[]>>(p => p
            .Add(c => c.Load, _ => Task.FromException<string[]>(new HttpRequestException("down")))
            .Add(c => c.InvalidMessage, _ => null)
            .Add(c => c.ChildContent, (RenderFragment<string[]>)(_ => _ => { })));

        cut.WaitForAssertion(() => Assert.Equal(AsyncLoadState.Error, cut.Instance.State));
        Assert.Empty(cut.FindAll(".solia-invalid"));
    }
}
