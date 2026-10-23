using Solia.Api.Endpoints;
using Solia.Core.Http;

namespace Solia.Api.Tests.Endpoints;

public class PagingTests
{
    [Fact]
    public void Defaults_to_the_first_page_of_the_default_size()
    {
        PageRequest page = PageRequest.From(null, null);

        Assert.Equal(1, page.Page);
        Assert.Equal(PageRequest.DefaultPageSize, page.PageSize);
        Assert.Equal(0, page.Skip);
    }

    [Theory]
    [InlineData(1, 1, 0)]
    [InlineData(3, 20, 40)]
    [InlineData(PageRequest.MaxRows / PageRequest.MaxPageSize, PageRequest.MaxPageSize, PageRequest.MaxRows - PageRequest.MaxPageSize)]
    [InlineData(PageRequest.MaxRows, 1, PageRequest.MaxRows - 1)]
    public void Skips_the_rows_of_the_previous_pages(int page, int pageSize, int skip) =>
        Assert.Equal(skip, PageRequest.From(page, pageSize).Skip);

    [Theory]
    [InlineData(0, null)]
    [InlineData(-1, null)]
    [InlineData(PageRequest.MaxRows / PageRequest.DefaultPageSize + 1, null)]
    [InlineData(PageRequest.MaxRows / PageRequest.MaxPageSize + 1, PageRequest.MaxPageSize)]
    [InlineData(PageRequest.MaxRows + 1, 1)]
    [InlineData(int.MaxValue, PageRequest.MaxPageSize)]
    public void Rejects_a_page_out_of_range_or_too_deep(int page, int? pageSize)
    {
        var ex = Assert.Throws<InvalidRequestException>(() => PageRequest.From(page, pageSize));
        Assert.StartsWith("page must be", ex.Message);
    }

    [Fact]
    public void Too_deep_page_message_states_the_bound_for_the_page_size()
    {
        var ex = Assert.Throws<InvalidRequestException>(() => PageRequest.From(51, 200));
        Assert.Equal("page must be between 1 and 50 at pageSize 200 (at most 10000 rows deep); narrow from/to instead.", ex.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(PageRequest.MaxPageSize + 1)]
    public void Rejects_a_page_size_out_of_range(int pageSize)
    {
        var ex = Assert.Throws<InvalidRequestException>(() => PageRequest.From(1, pageSize));
        Assert.StartsWith("pageSize must be", ex.Message);
    }
}
