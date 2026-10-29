using System.Net;
using System.Text.Json;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http.Resilience;
using Microsoft.Extensions.Options;
using Polly;
using Polly.Timeout;
using Solia.Core.Configuration;
using Solia.Core.Http;

namespace Solia.Core.Ai;

/// <summary>Registers the typed, resilient <see cref="ISoliaAiClient"/>.</summary>
public static class AiClientServiceCollectionExtensions
{
    /// <summary>Name of the client's resilience pipeline (shows up in Polly telemetry).</summary>
    public const string PipelineName = "ai-service";

    /// <summary>
    /// Binds and validates <see cref="AiServiceOptions"/> ("AiService" section, AI_SERVICE_URL overlay) and
    /// registers <see cref="ISoliaAiClient"/> as a typed <see cref="HttpClient"/> whose pipeline is, outermost
    /// first: total timeout (<c>RetrainTimeout</c> for <c>/model/retrain</c>) → retry (exponential backoff + jitter; availability failures of idempotent calls
    /// only — never a 4xx, a 500 or a 503 <c>no_champion</c>, never <c>/model/retrain</c>) → circuit breaker
    /// (same failures) → per-attempt timeout (<c>RetrainTimeout</c> again for <c>/model/retrain</c>) → response buffering. One pipeline — hence one breaker — per
    /// route, so a failing <c>/model/retrain</c> never blocks <c>/predict</c>. A probe (<c>/health</c>) is one
    /// attempt the breaker ignores: a readiness check must report the service as it is now, inside the check's
    /// own timeout, not wait out retries or a break opened by earlier probes. Returns the client builder so
    /// callers (tests) can swap the handler.
    /// </summary>
    public static IHttpClientBuilder AddSoliaAiClient(this IServiceCollection services, IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        services.AddOptions<AiServiceOptions>()
            .Bind(configuration.GetSection(AiServiceOptions.Section))
            .Configure(options => ApiFlatEnvAliases.Apply(configuration, options))
            .ValidateDataAnnotations()
            .Validate(o => o.HasValidBaseUrl(), "AiService:BaseUrl (AI_SERVICE_URL) must be an absolute http(s) URL.")
            .Validate(o => o.HasConsistentTimeouts(),
                "AiService: (MaxRetryAttempts + 1) × AttemptTimeout plus the nominal backoff delays must fit in TotalTimeout.")
            .ValidateOnStart();

        IHttpClientBuilder builder = services.AddHttpClient<ISoliaAiClient, SoliaAiClient>((sp, client) =>
        {
            AiServiceOptions options = sp.GetRequiredService<IOptions<AiServiceOptions>>().Value;
            client.BaseAddress = WithTrailingSlash(options.BaseUrl);
            client.Timeout = Timeout.InfiniteTimeSpan; // the resilience pipeline owns every timeout
        });

        builder
            .AddResilienceHandler(PipelineName, (pipeline, context) =>
            {
                AiServiceOptions options = context.ServiceProvider.GetRequiredService<IOptions<AiServiceOptions>>().Value;
                // /model/retrain waits for the run to be recorded: it gets its own, longer bound (never retried).
                bool retrain = IsRetrainPipeline(context.InstanceName);

                pipeline.AddTimeout(retrain ? options.RetrainTimeout : options.TotalTimeout);

                if (options.MaxRetryAttempts > 0)
                {
                    pipeline.AddRetry(new HttpRetryStrategyOptions
                    {
                        MaxRetryAttempts = options.MaxRetryAttempts,
                        Delay = options.RetryBaseDelay,
                        BackoffType = DelayBackoffType.Exponential,
                        UseJitter = true,
                        ShouldHandle = async args =>
                            IsIdempotent(args.Context) && !IsProbe(args.Context) && await IsAvailabilityFailureAsync(args.Outcome).ConfigureAwait(false),
                    });
                }

                pipeline.AddCircuitBreaker(new HttpCircuitBreakerStrategyOptions
                {
                    FailureRatio = options.CircuitBreakerFailureRatio,
                    MinimumThroughput = options.CircuitBreakerMinimumThroughput,
                    SamplingDuration = options.CircuitBreakerSamplingDuration,
                    BreakDuration = options.CircuitBreakerBreakDuration,
                    ShouldHandle = args => IsProbe(args.Context)
                        ? ValueTask.FromResult(false)
                        : IsAvailabilityFailureAsync(args.Outcome),
                });

                pipeline.AddTimeout(retrain ? options.RetrainTimeout : options.AttemptTimeout);
            })
            .SelectPipelineBy(_ => request => request.RequestUri?.AbsolutePath ?? string.Empty);

        // Inside the per-attempt timeout: HttpClient would otherwise buffer the body after the pipeline
        // returned, where no timeout bounds a server that sends headers and then stalls.
        builder.AddHttpMessageHandler(() => new ResponseBufferingHandler());

        return builder;
    }

    /// <summary>
    /// Whether an attempt failed because the ai-service could not serve it (transport error, timeout, 408,
    /// 429, 502, 503, 504) — the only failures worth retrying or counting toward the breaker. A 503
    /// <c>no_champion</c>/<c>champion_incompatible</c>, a 4xx or a 500 is a deterministic answer: retrying it
    /// only multiplies the load.
    /// </summary>
    internal static async ValueTask<bool> IsAvailabilityFailureAsync(Outcome<HttpResponseMessage> outcome)
    {
        if (outcome.Exception is not null)
        {
            return outcome.Exception is HttpRequestException or TimeoutRejectedException;
        }

        if (outcome.Result is not { } response)
        {
            return false;
        }

        return response.StatusCode switch
        {
            HttpStatusCode.ServiceUnavailable => !SoliaAiClient.IsNoChampionCode(await ReadErrorCodeAsync(response).ConfigureAwait(false)),
            HttpStatusCode.RequestTimeout
                or HttpStatusCode.TooManyRequests
                or HttpStatusCode.BadGateway
                or HttpStatusCode.GatewayTimeout => true,
            _ => false,
        };
    }

    /// <summary>
    /// The envelope's <c>error.code</c>, or null. Reads the buffered bytes, not the content stream: HttpContent
    /// caches that stream, so consuming it here would leave the client an empty body to map.
    /// </summary>
    internal static async Task<string?> ReadErrorCodeAsync(HttpResponseMessage response)
    {
        try
        {
            byte[] body = await response.Content.ReadAsByteArrayAsync().ConfigureAwait(false);
            return JsonSerializer.Deserialize<ApiErrorResponse>(body, AiJson.Options)?.Error?.Code;
        }
        catch (Exception ex) when (SoliaAiClient.IsUnreadableBody(ex))
        {
            return null;
        }
    }

    /// <summary>Whether the route-keyed pipeline (the request's absolute path) is <c>/model/retrain</c>'s.</summary>
    internal static bool IsRetrainPipeline(string pipelineKey) =>
        pipelineKey.EndsWith("/" + SoliaAiClient.RetrainPath, StringComparison.Ordinal);

    private static bool IsIdempotent(ResilienceContext context) =>
        context.GetRequestMessage() is not { } request
        || !request.Options.TryGetValue(SoliaAiClient.NonIdempotent, out bool nonIdempotent)
        || !nonIdempotent;

    private static bool IsProbe(ResilienceContext context) =>
        context.GetRequestMessage() is { } request
        && request.Options.TryGetValue(SoliaAiClient.Probe, out bool probe)
        && probe;

    private static Uri WithTrailingSlash(string baseUrl) =>
        new(baseUrl.EndsWith('/') ? baseUrl : baseUrl + "/", UriKind.Absolute);

    /// <summary>Reads the whole response body within the attempt (and its timeout's token).</summary>
    private sealed class ResponseBufferingHandler : DelegatingHandler
    {
        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            HttpResponseMessage response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
            try
            {
                await response.Content.LoadIntoBufferAsync(cancellationToken).ConfigureAwait(false);
                return response;
            }
            catch
            {
                response.Dispose();
                throw;
            }
        }
    }
}
