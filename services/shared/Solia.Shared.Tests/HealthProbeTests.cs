using System.Net;
using Solia.Shared.Hosting;

namespace Solia.Shared.Tests;

public class HealthProbeTests
{
    private sealed class StubHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send)
        : HttpMessageHandler
    {
        public Uri? LastUri { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            LastUri = request.RequestUri;
            return send(request, cancellationToken);
        }
    }

    private static StubHandler Returning(HttpStatusCode status) =>
        new((_, _) => Task.FromResult(new HttpResponseMessage(status)));

    [Theory]
    [InlineData(new[] { "--healthcheck" }, true)]
    [InlineData(new[] { "--healthcheck", "http://localhost:9000/health" }, true)]
    [InlineData(new string[0], false)]
    [InlineData(new[] { "--HEALTHCHECK" }, false)]
    [InlineData(new[] { "--urls", "--healthcheck" }, false)]
    public void IsRequested_only_when_the_flag_comes_first(string[] args, bool expected)
    {
        Assert.Equal(expected, HealthProbe.IsRequested(args));
    }

    [Fact]
    public async Task Healthy_or_degraded_200_exits_zero_on_the_default_url()
    {
        var handler = Returning(HttpStatusCode.OK);

        int code = await HealthProbe.RunAsync(["--healthcheck"], handler, error: TextWriter.Null, environment: _ => null);

        Assert.Equal(0, code);
        Assert.Equal(new Uri("http://localhost:8080/health"), handler.LastUri);
    }

    [Fact]
    public async Task Default_url_follows_the_configured_kestrel_port()
    {
        var handler = Returning(HttpStatusCode.OK);

        await HealthProbe.RunAsync(["--healthcheck"], handler, error: TextWriter.Null,
            environment: key => key == "ASPNETCORE_HTTP_PORTS" ? "9090" : null);

        Assert.Equal(new Uri("http://localhost:9090/health"), handler.LastUri);
    }

    [Theory]
    [InlineData(null, null, 8080)]
    [InlineData(null, "5000", 5000)]
    [InlineData(null, " 5001 ; 5002 ", 5001)]
    [InlineData(null, "nope", 8080)]
    [InlineData("http://+:7000", "5000", 7000)]
    [InlineData("https://*:7443;http://*:7001/", null, 7001)]
    [InlineData("http://localhost", null, 80)]
    [InlineData("http://[::]:7002", null, 7002)]
    [InlineData("https://*:7443", "5000", 5000)]
    [InlineData("http://*:99999", "5000", 5000)]
    public void ResolveDefaultUrl_reads_the_kestrel_port(string? urls, string? httpPorts, int expected)
    {
        var env = new Dictionary<string, string?> { ["ASPNETCORE_URLS"] = urls, ["ASPNETCORE_HTTP_PORTS"] = httpPorts };

        Uri url = HealthProbe.ResolveDefaultUrl(key => env.GetValueOrDefault(key));

        Assert.Equal(new Uri($"http://localhost:{expected}/health"), url);
    }

    [Fact]
    public async Task Probes_the_url_given_after_the_flag()
    {
        var handler = Returning(HttpStatusCode.OK);

        await HealthProbe.RunAsync(["--healthcheck", "http://localhost:9000/health"], handler, error: TextWriter.Null);

        Assert.Equal(new Uri("http://localhost:9000/health"), handler.LastUri);
    }

    [Fact]
    public async Task Unhealthy_503_exits_one_and_reports_the_status()
    {
        var error = new StringWriter();

        int code = await HealthProbe.RunAsync(["--healthcheck"], Returning(HttpStatusCode.ServiceUnavailable), error: error);

        Assert.Equal(1, code);
        Assert.Contains("503", error.ToString());
    }

    [Fact]
    public async Task Unreachable_service_exits_one()
    {
        var handler = new StubHandler((_, _) => throw new HttpRequestException("Connection refused"));
        var error = new StringWriter();

        int code = await HealthProbe.RunAsync(["--healthcheck"], handler, error: error);

        Assert.Equal(1, code);
        Assert.Contains("unreachable", error.ToString());
    }

    [Fact]
    public async Task Hung_service_times_out_and_exits_one()
    {
        var handler = new StubHandler(async (_, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct);
            return new HttpResponseMessage(HttpStatusCode.OK);
        });

        int code = await HealthProbe.RunAsync(
            ["--healthcheck"], handler, TimeSpan.FromMilliseconds(50), TextWriter.Null);

        Assert.Equal(1, code);
    }

    [Fact]
    public async Task Unexpected_failure_exits_one_instead_of_throwing()
    {
        var handler = new StubHandler((_, _) => throw new InvalidOperationException("boom"));
        var error = new StringWriter();

        int code = await HealthProbe.RunAsync(["--healthcheck"], handler, error: error);

        Assert.Equal(1, code);
        Assert.Contains("boom", error.ToString());
    }

    [Theory]
    [InlineData("not a url")]
    [InlineData("/health")]
    [InlineData("ftp://localhost/health")]
    [InlineData("file:///health")]
    public async Task Invalid_url_exits_one_without_a_request(string url)
    {
        var handler = Returning(HttpStatusCode.OK);
        var error = new StringWriter();

        int code = await HealthProbe.RunAsync(["--healthcheck", url], handler, error: error);

        Assert.Equal(1, code);
        Assert.Null(handler.LastUri);
        Assert.Contains("invalid url", error.ToString());
    }
}
