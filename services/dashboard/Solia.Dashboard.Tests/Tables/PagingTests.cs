using Solia.Dashboard.Api;
using Solia.Dashboard.Components.Tables;

namespace Solia.Dashboard.Tests.Tables;

public class PagingTests
{
    [Theory]
    [InlineData(0, 50, 1)]
    [InlineData(1, 50, 1)]
    [InlineData(50, 50, 1)]
    [InlineData(51, 50, 2)]
    [InlineData(1234, 50, 25)]
    public void Page_count_rounds_up_and_is_at_least_one(long total, int pageSize, long expected) =>
        Assert.Equal(expected, Paging.PageCount(total, pageSize));

    [Theory]
    [InlineData(25, 400)]
    [InlineData(50, 200)]
    [InlineData(200, 50)]
    [InlineData(30, 333)] // 333 × 30 = 9 990 ≤ 10 000 < 334 × 30
    [InlineData(20_000, 1)]
    public void The_deepest_page_keeps_page_times_size_within_the_cap(int pageSize, int expected)
    {
        Assert.Equal(expected, Paging.MaxReachablePage(pageSize));
        Assert.True(expected == 1 || (long)expected * pageSize <= Paging.MaxDepth);
    }

    [Fact]
    public void The_last_page_is_the_data_end_unless_the_cap_comes_first()
    {
        Assert.Equal(25, Paging.LastPage(1234, 50));
        Assert.False(Paging.IsCapped(1234, 50));

        Assert.Equal(200, Paging.LastPage(10_000, 50));
        Assert.False(Paging.IsCapped(10_000, 50));

        Assert.Equal(200, Paging.LastPage(10_001, 50));
        Assert.True(Paging.IsCapped(10_001, 50));
    }

    [Fact]
    public void Range_gives_the_one_based_rows_of_the_page()
    {
        Assert.Equal((51L, 100L), Paging.Range(new PagedResult<int>(Enumerable.Range(0, 50).ToArray(), 2, 50, 1234)));
        Assert.Equal((1201L, 1234L), Paging.Range(new PagedResult<int>(Enumerable.Range(0, 34).ToArray(), 25, 50, 1234)));
        Assert.Equal((0L, 0L), Paging.Range(new PagedResult<int>([], 30, 50, 1234)));
    }

    [Fact]
    public void Only_an_invalid_request_gets_a_friendly_message()
    {
        Assert.Equal("Check the filters: 'from' must not be after 'to'.",
            Paging.InvalidRequestMessage(new BackendErrorException(400, "invalid_request", "'from' must not be after 'to'.")));

        Assert.Null(Paging.InvalidRequestMessage(new BackendErrorException(500, "internal_error", "boom")));
        Assert.Null(Paging.InvalidRequestMessage(new BackendUnavailableException()));
        Assert.Null(Paging.InvalidRequestMessage(new InvalidOperationException()));
    }
}
