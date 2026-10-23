using System.Net;
using System.Text;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Solia.Core.Ai;

namespace Solia.Core.Tests.Ai;

/// <summary>A request as the stub saw it (the body is read eagerly: the message is disposed after the call).</summary>
internal sealed record RecordedRequest(HttpMethod Method, Uri? Uri, string? Body);

/// <summary>
/// Stubbed transport: answers each attempt with the next responder (the last one repeats) and records
/// every attempt, so tests can assert on retries without a network.
/// </summary>
internal sealed class StubHandler(params Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>>[] responders)
    : HttpMessageHandler
{
    private int _calls;

    public List<RecordedRequest> Requests { get; } = [];

    public int Calls => _calls;

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        string? body = request.Content is null ? null : await request.Content.ReadAsStringAsync(cancellationToken);
        lock (Requests)
        {
            Requests.Add(new RecordedRequest(request.Method, request.RequestUri, body));
        }

        int index = Interlocked.Increment(ref _calls) - 1;
        return await responders[Math.Min(index, responders.Length - 1)](request, cancellationToken);
    }

    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Json(HttpStatusCode status, string json) =>
        (_, _) => Task.FromResult(new HttpResponseMessage(status)
        {
            Content = new StringContent(json, Encoding.UTF8, "application/json"),
        });

    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Error(HttpStatusCode status, string code, string message = "boom") =>
        Json(status, $$$"""{"error":{"code":"{{{code}}}","message":"{{{message}}}"}}""");

    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Throws(Exception exception) =>
        (_, _) => Task.FromException<HttpResponseMessage>(exception);

    /// <summary>A 200 whose content type carries the given (possibly bogus) charset.</summary>
    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> JsonWithCharset(string json, string charset) =>
        (_, _) =>
        {
            var content = new ByteArrayContent(Encoding.UTF8.GetBytes(json));
            content.Headers.TryAddWithoutValidation("Content-Type", $"application/json; charset={charset}");
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = content });
        };

    /// <summary>Headers arrive at once, then the body never does.</summary>
    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> StalledBody() =>
        (_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new StalledContent() });

    public static Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> Hangs() =>
        async (_, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct);
            throw new InvalidOperationException("unreachable");
        };
}

/// <summary>Content whose body stalls until the read is cancelled.</summary>
internal sealed class StalledContent : HttpContent
{
    protected override Task SerializeToStreamAsync(Stream stream, System.Net.TransportContext? context) =>
        SerializeToStreamAsync(stream, context, CancellationToken.None);

    protected override async Task SerializeToStreamAsync(Stream stream, System.Net.TransportContext? context, CancellationToken cancellationToken)
    {
        await stream.WriteAsync("{\"mint\":"u8.ToArray(), cancellationToken);
        await Task.Delay(Timeout.Infinite, cancellationToken);
    }

    protected override bool TryComputeLength(out long length)
    {
        length = 0;
        return false;
    }
}

/// <summary>Builds the real DI registration (options + resilience pipeline) over a <see cref="StubHandler"/>.</summary>
internal static class AiClientTestHost
{
    public const string PredictJson = """
        {"mint":"So1abc","model_version":"v3","predictions":[
          {"horizon":"1m","pump_prob":0.72,"rug_prob":0.08,"expected_return":0.35,"confidence":81.5}]}
        """;

    /// <summary>Fast defaults: tiny backoff, short timeouts, a breaker that never trips unless asked to.</summary>
    public static ServiceProvider Build(StubHandler handler, IDictionary<string, string?>? overrides = null)
    {
        var settings = new Dictionary<string, string?>
        {
            ["AiService:BaseUrl"] = "http://ai-service:8000/root",
            ["AiService:RetryBaseDelay"] = "00:00:00.001",
            ["AiService:AttemptTimeout"] = "00:00:02",
            ["AiService:TotalTimeout"] = "00:00:10",
            ["AiService:CircuitBreakerMinimumThroughput"] = "1000",
        };
        foreach ((string key, string? value) in overrides ?? new Dictionary<string, string?>())
        {
            settings[key] = value;
        }

        IConfiguration configuration = new ConfigurationBuilder().AddInMemoryCollection(settings).Build();
        var services = new ServiceCollection();
        services.AddSoliaAiClient(configuration).ConfigurePrimaryHttpMessageHandler(() => handler);
        return services.BuildServiceProvider(validateScopes: true);
    }

    public static ISoliaAiClient Client(this ServiceProvider provider) => provider.GetRequiredService<ISoliaAiClient>();

    public static AiPredictRequest SampleRequest() =>
        AiPredictRequest.ForVector("So1abc", new Dictionary<string, double?> { ["price"] = 1.5, ["volume_1m"] = null });
}
