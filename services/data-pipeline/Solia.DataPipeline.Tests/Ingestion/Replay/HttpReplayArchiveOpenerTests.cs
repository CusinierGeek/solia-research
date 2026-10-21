using System.Net;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Tests.Ingestion.Replay;

public class HttpReplayArchiveOpenerTests
{
    private static readonly DateTimeOffset Hour = new(2026, 4, 18, 5, 0, 0, TimeSpan.Zero);

    [Theory]
    [InlineData("https://replay.pumpapi.io")]
    [InlineData("https://replay.pumpapi.io/")]
    public void BuildUrl_formats_the_hour_path(string baseUrl)
    {
        Assert.Equal(
            "https://replay.pumpapi.io/2026/04/18/05.jsonl.zst",
            HttpReplayArchiveOpener.BuildUrl(baseUrl, Hour));
    }

    [Fact]
    public async Task Success_returns_the_body_from_the_configured_base_url()
    {
        var handler = new StubHandler(HttpStatusCode.OK, "archive-bytes"u8.ToArray());
        HttpReplayArchiveOpener opener = Opener(handler, "https://mirror.example/replay");

        await using Stream? body = await opener.OpenAsync(Hour, CancellationToken.None);

        Assert.NotNull(body);
        using var copy = new MemoryStream();
        await body!.CopyToAsync(copy);
        Assert.Equal("archive-bytes"u8.ToArray(), copy.ToArray());
        Assert.Equal("https://mirror.example/replay/2026/04/18/05.jsonl.zst", handler.RequestedUri!.ToString());
    }

    [Fact]
    public async Task Not_found_returns_null_and_releases_the_response()
    {
        var handler = new StubHandler(HttpStatusCode.NotFound);

        Stream? body = await Opener(handler).OpenAsync(Hour, CancellationToken.None);

        Assert.Null(body);
        Assert.True(handler.Content.Disposed);
    }

    [Theory]
    [InlineData(HttpStatusCode.InternalServerError)]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    [InlineData(HttpStatusCode.Forbidden)]
    public async Task Error_status_throws_and_releases_the_response(HttpStatusCode status)
    {
        var handler = new StubHandler(status);

        await Assert.ThrowsAsync<HttpRequestException>(() => Opener(handler).OpenAsync(Hour, CancellationToken.None));
        Assert.True(handler.Content.Disposed);
    }

    [Fact]
    public async Task Unreachable_host_throws()
    {
        using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
        var opener = new HttpReplayArchiveOpener(client,
            Microsoft.Extensions.Options.Options.Create(new PumpApiOptions { ReplayBaseUrl = "http://127.0.0.1:1" }));

        await Assert.ThrowsAsync<HttpRequestException>(() => opener.OpenAsync(Hour, CancellationToken.None));
    }

    private static HttpReplayArchiveOpener Opener(StubHandler handler, string baseUrl = "https://replay.pumpapi.io") =>
        new(new HttpClient(handler), Microsoft.Extensions.Options.Options.Create(new PumpApiOptions { ReplayBaseUrl = baseUrl }));

    private sealed class StubHandler(HttpStatusCode status, byte[]? body = null) : HttpMessageHandler
    {
        public TrackingContent Content { get; } = new(body ?? []);

        public Uri? RequestedUri { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            RequestedUri = request.RequestUri;
            return Task.FromResult(new HttpResponseMessage(status) { Content = Content, RequestMessage = request });
        }
    }

    private sealed class TrackingContent(byte[] body) : ByteArrayContent(body)
    {
        public bool Disposed { get; private set; }

        protected override void Dispose(bool disposing)
        {
            Disposed = true;
            base.Dispose(disposing);
        }
    }
}
