using System.Net;
using System.Text;

namespace Solia.Dashboard.Tests.Api;

/// <summary>
/// An <see cref="HttpMessageHandler"/> that records every request URI and answers from a queue of responders
/// (the last one repeats), so a test controls each attempt.
/// </summary>
internal sealed class StubHttpHandler(params Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>>[] responders)
    : HttpMessageHandler
{
    private int _calls;

    public List<Uri> Requests { get; } = [];

    public int Calls => _calls;

    public static StubHttpHandler Json(string json, HttpStatusCode status = HttpStatusCode.OK) =>
        new(Respond(json, status));

    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Respond(
        string body, HttpStatusCode status = HttpStatusCode.OK, string mediaType = "application/json") =>
        (_, _) => Task.FromResult(new HttpResponseMessage(status) { Content = new StringContent(body, Encoding.UTF8, mediaType) });

    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Throw(Exception ex) =>
        (_, _) => Task.FromException<HttpResponseMessage>(ex);

    /// <summary>Never answers: waits until the request is cancelled (an attempt timeout).</summary>
    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Hang() =>
        async (_, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct);
            throw new InvalidOperationException("unreachable");
        };

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        lock (Requests)
        {
            Requests.Add(request.RequestUri!);
        }

        int call = Interlocked.Increment(ref _calls) - 1;
        return responders[Math.Min(call, responders.Length - 1)](request, cancellationToken);
    }
}
