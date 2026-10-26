using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Text;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Polly;
using Polly.Timeout;
using Solia.Dashboard.Api;
using Solia.Dashboard.Configuration;

namespace Solia.Dashboard.Tests.Api;

/// <summary>The DI-registered client: base address, resilience pipeline (retry + timeouts).</summary>
public class SoliaApiClientRegistrationTests
{
    private const string WinRateJson = """{ "winrate": 0.5, "wins": 1, "losses": 1, "break_even": 0, "window": "all" }""";

    private static ISoliaApiClient Resolve(StubHttpHandler handler, string baseUrl = "http://backend:8080",
        string attemptTimeout = "00:00:05", string totalTimeout = "00:00:12")
    {
        IConfiguration configuration = new ConfigurationBuilder()
            .AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Backend:BaseUrl"] = baseUrl,
                ["Backend:AttemptTimeout"] = attemptTimeout,
                ["Backend:TotalTimeout"] = totalTimeout,
            })
            .Build();
        var services = new ServiceCollection().AddDashboardOptions(configuration);
        services.AddSoliaApiClient().ConfigurePrimaryHttpMessageHandler(() => handler);
        return services.BuildServiceProvider().GetRequiredService<ISoliaApiClient>();
    }

    [Theory]
    [InlineData("http://backend:8080", "http://backend:8080/api/v1/stats/winrate")]
    [InlineData("http://backend:8080/", "http://backend:8080/api/v1/stats/winrate")]
    [InlineData("https://proxy.example/solia", "https://proxy.example/solia/api/v1/stats/winrate")]
    public async Task Requests_go_under_the_api_prefix_of_the_base_url(string baseUrl, string expected)
    {
        StubHttpHandler handler = StubHttpHandler.Json(WinRateJson);

        await Resolve(handler, baseUrl).GetWinRateAsync();

        Assert.Equal(expected, Assert.Single(handler.Requests).ToString());
    }

    [Fact]
    public async Task A_transient_failure_is_retried_once()
    {
        var handler = new StubHttpHandler(
            StubHttpHandler.Respond("down", HttpStatusCode.ServiceUnavailable, "text/plain"),
            StubHttpHandler.Respond(WinRateJson));

        WinRateResponse winrate = await Resolve(handler).GetWinRateAsync();

        Assert.Equal(1, winrate.Wins);
        Assert.Equal(2, handler.Calls);
    }

    [Fact]
    public async Task A_network_failure_is_retried_only_once_then_unavailable()
    {
        var handler = new StubHttpHandler(StubHttpHandler.Throw(new HttpRequestException("connection refused")));

        await Assert.ThrowsAsync<BackendUnavailableException>(() => Resolve(handler).GetLiveTokensAsync());

        Assert.Equal(2, handler.Calls);
    }

    [Theory]
    [InlineData(HttpStatusCode.BadRequest)]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.InternalServerError)]
    public async Task A_deterministic_error_is_not_retried(HttpStatusCode status)
    {
        var handler = new StubHttpHandler(StubHttpHandler.Respond("""{ "error": { "code": "x", "message": "y" } }""", status));

        await Assert.ThrowsAsync<BackendErrorException>(() => Resolve(handler).GetModelStatusAsync());

        Assert.Equal(1, handler.Calls);
    }

    [Fact]
    public async Task A_hanging_backend_times_out_as_unavailable()
    {
        var handler = new StubHttpHandler(StubHttpHandler.Hang());

        var ex = await Assert.ThrowsAsync<BackendUnavailableException>(() =>
            Resolve(handler, attemptTimeout: "00:00:00.100", totalTimeout: "00:00:01").GetBotLogsAsync());

        Assert.Equal(2, handler.Calls); // the attempt timeout is transient: one retry, then give up
        Assert.Equal(BackendUnavailableException.DefaultMessage, ex.Message);
    }

    [Fact]
    public async Task A_retry_after_header_does_not_delay_the_retry()
    {
        var handler = new StubHttpHandler(
            (_, _) =>
            {
                var response = new HttpResponseMessage(HttpStatusCode.ServiceUnavailable)
                {
                    Content = new StringContent("""{ "error": { "code": "unavailable", "message": "busy" } }""", Encoding.UTF8, "application/json"),
                };
                response.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(30));
                return Task.FromResult(response);
            });
        var watch = Stopwatch.StartNew();

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => Resolve(handler).GetWinRateAsync());

        Assert.True(watch.Elapsed < TimeSpan.FromSeconds(5), $"took {watch.Elapsed}");
        Assert.Equal((2, "unavailable", "busy"), (handler.Calls, ex.Code, ex.Message));
    }

    [Fact]
    public async Task An_oversized_body_is_an_invalid_response_and_not_retried()
    {
        var handler = new StubHttpHandler((_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new ByteArrayContent(new byte[SoliaApiClientServiceCollectionExtensions.MaxResponseBytes + 1]),
        }));

        var ex = await Assert.ThrowsAsync<BackendErrorException>(() => Resolve(handler).GetLiveTokensAsync());

        Assert.Equal(BackendErrorException.InvalidResponseCode, ex.Code);
        Assert.Equal(1, handler.Calls);
    }

    [Theory]
    [InlineData(HttpStatusCode.RequestTimeout, true)]
    [InlineData(HttpStatusCode.TooManyRequests, true)]
    [InlineData(HttpStatusCode.BadGateway, true)]
    [InlineData(HttpStatusCode.ServiceUnavailable, true)]
    [InlineData(HttpStatusCode.GatewayTimeout, true)]
    [InlineData(HttpStatusCode.InternalServerError, false)]
    [InlineData(HttpStatusCode.NotImplemented, false)]
    [InlineData(HttpStatusCode.NotFound, false)]
    [InlineData(HttpStatusCode.OK, false)]
    public void Transient_statuses_are_the_librarys_minus_500_and_501(HttpStatusCode status, bool transient)
    {
        using var response = new HttpResponseMessage(status);

        Assert.Equal(transient, SoliaApiClientServiceCollectionExtensions.IsTransient(Outcome.FromResult(response)));
    }

    [Fact]
    public void Transport_failures_are_transient_but_an_oversized_body_is_not()
    {
        Assert.True(SoliaApiClientServiceCollectionExtensions.IsTransient(
            Outcome.FromException<HttpResponseMessage>(new HttpRequestException("refused"))));
        Assert.True(SoliaApiClientServiceCollectionExtensions.IsTransient(
            Outcome.FromException<HttpResponseMessage>(new TimeoutRejectedException())));
        Assert.False(SoliaApiClientServiceCollectionExtensions.IsTransient(
            Outcome.FromException<HttpResponseMessage>(new HttpRequestException(HttpRequestError.ConfigurationLimitExceeded, "too big"))));
    }

    [Fact]
    public void The_host_registers_the_client()
    {
        using var factory = new Microsoft.AspNetCore.Mvc.Testing.WebApplicationFactory<Program>()
            .WithWebHostBuilder(b => b.UseSetting("Backend:BaseUrl", "http://backend:8080"));

        Assert.NotNull(factory.Services.GetRequiredService<ISoliaApiClient>());
    }
}
