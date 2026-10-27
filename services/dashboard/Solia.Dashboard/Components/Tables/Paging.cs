using Solia.Dashboard.Api;

namespace Solia.Dashboard.Components.Tables;

/// <summary>
/// The backend's paging rules (docs/API-CONTRACTS.md): 1-based pages of 1–200 rows, and no page reaching past
/// <see cref="MaxDepth"/> rows (<c>page × pageSize ≤ 10000</c>, else a 400) — so the last page a pager offers is
/// the earlier of the data's end and that cap.
/// </summary>
public static class Paging
{
    public const int MaxDepth = 10_000;

    public const int DefaultPageSize = 50;

    /// <summary>The largest page the backend serves.</summary>
    public const int MaxPageSize = 200;

    /// <summary>The page sizes the pager offers.</summary>
    public static readonly IReadOnlyList<int> PageSizes = [25, 50, 100, MaxPageSize];

    /// <summary>How many pages <paramref name="total"/> rows fill (at least one, so an empty result is "page 1 of 1").</summary>
    public static long PageCount(long total, int pageSize) =>
        total <= 0 ? 1 : (total + pageSize - 1) / pageSize;

    /// <summary>The deepest page the backend serves at this page size.</summary>
    public static int MaxReachablePage(int pageSize) => Math.Max(1, MaxDepth / pageSize);

    /// <summary>The last page the pager can go to.</summary>
    public static int LastPage(long total, int pageSize) =>
        (int)Math.Min(PageCount(total, pageSize), MaxReachablePage(pageSize));

    /// <summary>Whether the depth cap hides rows: only narrowing the filters reaches them.</summary>
    public static bool IsCapped(long total, int pageSize) => PageCount(total, pageSize) > MaxReachablePage(pageSize);

    /// <summary>The 1-based rows a page shows: <c>(51, 100)</c>; <c>(0, 0)</c> for a page with no rows.</summary>
    public static (long First, long Last) Range<T>(PagedResult<T> result)
    {
        if (result.Items.Count == 0)
        {
            return (0, 0);
        }
        long first = ((long)result.Page - 1) * result.PageSize + 1;
        return (first, first + result.Items.Count - 1);
    }

    /// <summary>The message for a request the backend rejected as invalid (bad filter, page too deep), else null.</summary>
    public static string? InvalidRequestMessage(Exception exception) =>
        exception is BackendErrorException { Code: "invalid_request" } invalid
            ? $"Check the filters: {invalid.Message}"
            : null;
}
