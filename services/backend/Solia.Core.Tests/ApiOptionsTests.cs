using Solia.Core.Configuration;

namespace Solia.Core.Tests;

public class ApiOptionsTests
{
    [Fact]
    public void Default_base_url_is_valid()
    {
        Assert.True(new ApiOptions().HasValidBaseUrl());
    }

    [Theory]
    [InlineData("http://backend:8080")]
    [InlineData("https://api.solia.example")]
    [InlineData("http://localhost:5000/")]
    public void Accepts_absolute_http_urls(string url)
    {
        Assert.True(new ApiOptions { BaseUrl = url }.HasValidBaseUrl());
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("backend:8080")]
    [InlineData("ftp://backend")]
    [InlineData("ws://backend:8080")]
    [InlineData("/api/v1")]
    public void Rejects_non_absolute_or_non_http_urls(string url)
    {
        Assert.False(new ApiOptions { BaseUrl = url }.HasValidBaseUrl());
    }
}
