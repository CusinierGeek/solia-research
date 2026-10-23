using System.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Storage;
using Solia.Core.Http;

namespace Solia.Api.Endpoints;

/// <summary>
/// The shared pagination contract of every paginated read (<c>?page=&amp;pageSize=</c>): 1-based
/// <see cref="Page"/>, <see cref="PageSize"/> in [1, <see cref="MaxPageSize"/>], and no page reaching past
/// the first <see cref="MaxRows"/> rows. Built through <see cref="From"/>, which rejects out-of-range values
/// with a 400 envelope.
/// </summary>
public sealed record PageRequest
{
    public const int DefaultPageSize = 50;

    public const int MaxPageSize = 200;

    /// <summary>
    /// Deepest row a page may reach (<c>page × pageSize</c>). OFFSET makes Postgres read and discard every
    /// skipped row, so an unbounded page would let one public request scan the whole table; past this,
    /// narrow the <c>from</c>/<c>to</c> range instead.
    /// </summary>
    public const int MaxRows = 10_000;

    private PageRequest(int page, int pageSize)
    {
        Page = page;
        PageSize = pageSize;
    }

    public int Page { get; }

    public int PageSize { get; }

    /// <summary>Rows before this page.</summary>
    public int Skip => (Page - 1) * PageSize;

    /// <summary>Validates the raw query values (null = default: page 1, <see cref="DefaultPageSize"/>).</summary>
    /// <exception cref="InvalidRequestException">A value is out of range.</exception>
    public static PageRequest From(int? page, int? pageSize)
    {
        int p = page ?? 1;
        int size = pageSize ?? DefaultPageSize;
        if (size is < 1 or > MaxPageSize)
        {
            throw new InvalidRequestException($"pageSize must be between 1 and {MaxPageSize}.");
        }

        int maxPage = MaxRows / size;
        if (p < 1 || p > maxPage)
        {
            throw new InvalidRequestException(
                $"page must be between 1 and {maxPage} at pageSize {size} (at most {MaxRows} rows deep); narrow from/to instead.");
        }

        return new PageRequest(p, size);
    }
}

/// <summary>One page of <typeparamref name="T"/> plus the total row count matching the filters.</summary>
public sealed record PagedResponse<T>(IReadOnlyList<T> Items, int Page, int PageSize, long Total);

public static class PagingExtensions
{
    /// <summary>
    /// Counts <paramref name="query"/> (already filtered and ordered — the order makes pages stable) and
    /// fetches the requested page, both in one read-only repeatable-read transaction so <c>total</c> and
    /// <c>items</c> see the same snapshot even while the writers insert.
    /// </summary>
    public static async Task<PagedResponse<T>> ToPageAsync<T>(
        this IQueryable<T> query, DbContext db, PageRequest page, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(db);
        await using IDbContextTransaction? snapshot = db.Database.CurrentTransaction is null
            ? await db.Database.BeginTransactionAsync(IsolationLevel.RepeatableRead, cancellationToken)
            : null;

        long total = await query.LongCountAsync(cancellationToken);
        List<T> items = await query.Skip(page.Skip).Take(page.PageSize).ToListAsync(cancellationToken);
        return new PagedResponse<T>(items, page.Page, page.PageSize, total);
    }
}
