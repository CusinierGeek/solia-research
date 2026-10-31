using System.Globalization;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Routing;
using Solia.Dashboard.Api;
using Solia.Dashboard.Formatting;

namespace Solia.Dashboard.Components.Tables;

/// <summary>
/// A page whose view (filters, page) lives in the URL. The URL is the source of truth: the filter bar and the pager
/// only <see cref="Navigate"/>, and every navigation (theirs, the browser's back/forward, a shared link) is read back
/// from it through <see cref="ReadUrl"/>. Navigating away to another page's URL is not: its query string means
/// nothing here, and the page is about to go away. Also holds the filter inputs' value conversions.
/// </summary>
public abstract class UrlStatePage : ComponentBase, IDisposable
{
    private const string InputFormat = "yyyy-MM-dd'T'HH:mm:ss";

    [Inject] protected NavigationManager Navigation { get; set; } = default!;

    /// <summary>The viewer's zone, cascaded by the layout (UTC outside it): every time shown or typed is in it.</summary>
    [CascadingParameter] protected TimeZoneInfo DisplayZone { get; set; } = TimeZoneInfo.Utc;

    /// <summary>The page's base-relative path, without a query string (e.g. <c>trades</c>).</summary>
    protected abstract string Path { get; }

    /// <summary>Reads the view from <see cref="NavigationManager.Uri"/>.</summary>
    protected abstract void ReadUrl();

    protected override void OnInitialized()
    {
        ReadUrl();
        Navigation.LocationChanged += OnLocationChanged;
    }

    /// <summary>Goes to this page with these query-string parameters (null removes one).</summary>
    protected void Navigate(IReadOnlyDictionary<string, object?> parameters) =>
        Navigation.NavigateTo(Navigation.GetUriWithQueryParameters(parameters));

    private void OnLocationChanged(object? sender, LocationChangedEventArgs e)
    {
        if (!IsThisPage(e.Location))
        {
            return;
        }
        _ = InvokeAsync(() =>
        {
            ReadUrl();
            StateHasChanged();
        });
    }

    private bool IsThisPage(string uri)
    {
        string relative = Navigation.ToBaseRelativePath(uri);
        int end = relative.IndexOfAny(['?', '#']);
        return string.Equals((end < 0 ? relative : relative[..end]).TrimEnd('/'), Path, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>An enum filter's label: its wire spelling, or "All" when not filtered.</summary>
    protected static string Wire<T>(T? value) where T : struct, Enum => value is { } v ? QueryValues.Format(v) : "All";

    protected string? ToInput(DateTimeOffset? ts) =>
        ts is { } t ? DisplayTimeZone.ToWallClock(t, DisplayZone).ToString(InputFormat, CultureInfo.InvariantCulture) : null;

    // A datetime-local value has no zone: it is read in the display zone, like the column it filters (the URL keeps UTC).
    protected DateTimeOffset? FromInput(string? value) =>
        QueryValues.ParseTimestamp(value) is { } parsed ? DisplayTimeZone.FromWallClock(parsed.UtcDateTime, DisplayZone) : null;

    public virtual void Dispose()
    {
        Navigation.LocationChanged -= OnLocationChanged;
        GC.SuppressFinalize(this);
    }
}
