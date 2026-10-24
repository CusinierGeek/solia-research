using Solia.Dashboard.Components.State;

namespace Solia.Dashboard.Tests;

public class StateComponentTests : MudTestContext
{
    [Fact]
    public void Loading_defaults_to_skeleton_rows()
    {
        IRenderedComponent<LoadingState> cut = Render<LoadingState>(p => p.Add(c => c.Lines, 4));

        Assert.NotNull(cut.Find(".solia-loading--skeleton"));
        Assert.Equal(4, cut.FindAll(".mud-skeleton").Count);
        Assert.Empty(cut.FindAll(".mud-progress-circular"));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Loading_skeleton_falls_back_to_a_default_accessible_label(string? message)
    {
        IRenderedComponent<LoadingState> cut = Render<LoadingState>(p => p.Add(c => c.Message, message));

        Assert.Equal("Loading", cut.Find(".solia-loading--skeleton").GetAttribute("aria-label"));
    }

    [Fact]
    public void Loading_skeleton_draws_at_least_one_row()
    {
        IRenderedComponent<LoadingState> cut = Render<LoadingState>(p => p.Add(c => c.Lines, 0));

        Assert.Single(cut.FindAll(".mud-skeleton"));
    }

    [Fact]
    public void Loading_spinner_variant_shows_a_spinner_and_its_message()
    {
        IRenderedComponent<LoadingState> cut = Render<LoadingState>(p => p
            .Add(c => c.Variant, LoadingVariant.Spinner)
            .Add(c => c.Message, "Fetching stats"));

        Assert.NotNull(cut.Find(".solia-loading--spinner .mud-progress-circular"));
        Assert.Contains("Fetching stats", cut.Markup);
        Assert.Empty(cut.FindAll(".mud-skeleton"));
    }

    [Fact]
    public void Empty_shows_its_message()
    {
        IRenderedComponent<EmptyState> cut = Render<EmptyState>(p => p.Add(c => c.Message, "No trades yet."));

        Assert.Contains("No trades yet.", cut.Find(".solia-empty").TextContent);
    }

    [Fact]
    public void Error_shows_its_message_and_retry_invokes_the_callback()
    {
        int retries = 0;
        IRenderedComponent<ErrorState> cut = Render<ErrorState>(p => p
            .Add(c => c.Message, "Backend unreachable.")
            .Add(c => c.OnRetry, () => retries++));

        Assert.Contains("Backend unreachable.", cut.Find(".solia-error").TextContent);

        cut.Find("button.solia-error__retry").Click();

        Assert.Equal(1, retries);
    }

    [Fact]
    public void Error_without_a_retry_callback_has_no_retry_button()
    {
        IRenderedComponent<ErrorState> cut = Render<ErrorState>(p => p.Add(c => c.Message, "Nope."));

        Assert.Empty(cut.FindAll("button"));
    }
}
