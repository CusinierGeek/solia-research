using Solia.Core.Http;

namespace Solia.Core.Tests;

public class ApiExceptionTests
{
    [Fact]
    public void Not_found_carries_404_and_not_found_code()
    {
        var ex = new NotFoundException("unknown mint");

        Assert.Equal(404, ex.StatusCode);
        Assert.Equal(ApiErrorCodes.NotFound, ex.Code);
        Assert.Equal("unknown mint", ex.Message);
    }

    [Fact]
    public void Invalid_request_carries_400_and_invalid_request_code()
    {
        var ex = new InvalidRequestException("missing horizons");

        Assert.Equal(400, ex.StatusCode);
        Assert.Equal(ApiErrorCodes.InvalidRequest, ex.Code);
    }

    [Fact]
    public void Base_exception_keeps_status_code_and_inner()
    {
        var inner = new InvalidOperationException("boom");
        var ex = new ApiException(503, ApiErrorCodes.Unavailable, "ai-service down", inner);

        Assert.Equal(503, ex.StatusCode);
        Assert.Equal(ApiErrorCodes.Unavailable, ex.Code);
        Assert.Same(inner, ex.InnerException);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Blank_code_is_rejected(string code)
    {
        Assert.Throws<ArgumentException>(() => new ApiException(500, code, "message"));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(200)]
    [InlineData(399)]
    [InlineData(600)]
    public void Non_error_status_is_rejected(int statusCode)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new ApiException(statusCode, ApiErrorCodes.Conflict, "message"));
    }

    [Theory]
    [InlineData(400)]
    [InlineData(599)]
    public void Error_status_bounds_are_accepted(int statusCode)
    {
        Assert.Equal(statusCode, new ApiException(statusCode, ApiErrorCodes.Conflict, "message").StatusCode);
    }
}
