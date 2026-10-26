using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using Solia.Dashboard.Configuration;

namespace Solia.Dashboard.Tests;

public class BackendOptionsTests
{
    private static BackendOptions Resolve(params (string Key, string? Value)[] settings)
    {
        IConfiguration configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(settings.Select(s => new KeyValuePair<string, string?>(s.Key, s.Value)))
            .Build();
        ServiceProvider services = new ServiceCollection().AddDashboardOptions(configuration).BuildServiceProvider();
        return services.GetRequiredService<IOptions<BackendOptions>>().Value;
    }

    [Theory]
    [InlineData("http://localhost:8080")]
    [InlineData("https://backend.solia.example/")]
    public void Accepts_an_absolute_http_url(string url) =>
        Assert.Equal(url, Resolve(("Backend:BaseUrl", url)).BaseUrl);

    [Theory]
    [InlineData("not a url")]
    [InlineData("/relative/path")]
    [InlineData("ftp://backend:21")]
    [InlineData("   ")]
    public void Rejects_a_bad_base_url(string url)
    {
        OptionsValidationException ex = Assert.Throws<OptionsValidationException>(() => Resolve(("Backend:BaseUrl", url)));

        Assert.Contains(DashboardServiceCollectionExtensions.InvalidBaseUrlMessage, ex.Failures);
    }

    [Fact]
    public void Rejects_a_missing_base_url_with_a_single_message()
    {
        OptionsValidationException ex = Assert.Throws<OptionsValidationException>(() => Resolve());

        Assert.Equal([DashboardServiceCollectionExtensions.InvalidBaseUrlMessage], ex.Failures);
    }

    [Fact]
    public void Flat_backend_url_wins_over_the_section()
    {
        BackendOptions options = Resolve(("Backend:BaseUrl", "http://localhost:8080"), ("BACKEND_URL", " http://backend:8080 "));

        Assert.Equal("http://backend:8080", options.BaseUrl);
    }

    [Fact]
    public void Flat_backend_url_alone_is_enough() =>
        Assert.Equal("http://backend:8080", Resolve(("BACKEND_URL", "http://backend:8080")).BaseUrl);

    [Fact]
    public void Bad_flat_backend_url_is_rejected_even_with_a_valid_section() =>
        Assert.Throws<OptionsValidationException>(() =>
            Resolve(("Backend:BaseUrl", "http://localhost:8080"), ("BACKEND_URL", "backend")));

    [Fact]
    public void Default_timeouts_are_consistent()
    {
        BackendOptions options = Resolve(("Backend:BaseUrl", "http://localhost:8080"));

        Assert.Equal((TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(12)), (options.AttemptTimeout, options.TotalTimeout));
    }

    [Theory]
    [InlineData("00:00:00", "00:00:10")]
    [InlineData("-00:00:01", "00:00:10")]
    [InlineData("00:00:05", "00:00:09")]
    [InlineData("00:00:05", "00:00:10")] // two attempts fit, the jittered retry delay between them does not
    [InlineData("00:00:00.005", "00:00:10")] // below Polly's 10 ms minimum
    [InlineData("00:00:05", "2.00:00:00")] // above Polly's 24 h maximum
    public void Rejects_inconsistent_timeouts(string attempt, string total)
    {
        OptionsValidationException ex = Assert.Throws<OptionsValidationException>(() => Resolve(
            ("Backend:BaseUrl", "http://localhost:8080"), ("Backend:AttemptTimeout", attempt), ("Backend:TotalTimeout", total)));

        Assert.Contains(DashboardServiceCollectionExtensions.InvalidTimeoutsMessage, ex.Failures);
    }

    [Fact]
    public void Accepts_two_attempts_plus_the_maximum_retry_delay()
    {
        BackendOptions options = Resolve(
            ("Backend:BaseUrl", "http://localhost:8080"), ("Backend:AttemptTimeout", "00:00:05"), ("Backend:TotalTimeout", "00:00:10.250"));

        Assert.Equal(TimeSpan.FromSeconds(10.25), options.TotalTimeout);
    }
}
