using Solia.Api.Endpoints;
using Solia.Core.Http;

namespace Solia.Api.Tests.Endpoints;

public class QueryValidationTests
{
    private static readonly string[] Allowed = ["open", "closed"];

    [Theory]
    [InlineData("open", "open")]
    [InlineData(" Closed ", "closed")]
    [InlineData(null, null)]
    [InlineData("", null)]
    [InlineData("   ", null)]
    public void OneOf_normalizes_an_allowed_value_and_passes_absence_through(string? value, string? expected) =>
        Assert.Equal(expected, QueryValidation.OneOf(value, "status", Allowed));

    [Theory]
    [InlineData("pending")]
    [InlineData("open,closed")]
    public void OneOf_rejects_anything_else_naming_the_parameter_and_choices(string value)
    {
        var ex = Assert.Throws<InvalidRequestException>(() => QueryValidation.OneOf(value, "status", Allowed));
        Assert.Equal("status must be one of: open, closed.", ex.Message);
    }

    [Fact]
    public void Range_accepts_open_and_ordered_bounds()
    {
        DateTimeOffset t = DateTimeOffset.UnixEpoch;
        QueryValidation.Range(null, null);
        QueryValidation.Range(t, null);
        QueryValidation.Range(null, t);
        QueryValidation.Range(t, t);
    }

    [Fact]
    public void Range_rejects_from_after_to()
    {
        DateTimeOffset t = DateTimeOffset.UnixEpoch;
        Assert.Throws<InvalidRequestException>(() => QueryValidation.Range(t.AddSeconds(1), t));
    }

    [Theory]
    [InlineData(null, 50)]
    [InlineData(1, 1)]
    [InlineData(500, 500)]
    public void Limit_defaults_and_accepts_values_in_range(int? value, int expected) =>
        Assert.Equal(expected, QueryValidation.Limit(value, 50, 500));

    [Theory]
    [InlineData(0)]
    [InlineData(501)]
    public void Limit_rejects_values_out_of_range(int value) =>
        Assert.Throws<InvalidRequestException>(() => QueryValidation.Limit(value, 50, 500));
}
