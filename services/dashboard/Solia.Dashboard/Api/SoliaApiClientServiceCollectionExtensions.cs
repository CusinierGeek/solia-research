using System.Net;
using Microsoft.Extensions.Http.Resilience;
using Microsoft.Extensions.Options;
using Polly;
using Solia.Dashboard.Configuration;

namespace Solia.Dashboard.Api;

/// <summary>Registers the typed, resilient <see cref="ISoliaApiClient"/>.</summary>
public static class SoliaApiClientServiceCollectionExtensions
{
    /// <summary>Name of the client's resilience pipeline (shows up in Polly telemetry).</summary>
    public const string PipelineName = "backend";

    /// <summary>The REST surface's path prefix under <see cref="BackendOptions.BaseUrl"/>.</summary>
    public const string ApiPrefix = "api/v1/";

    /// <summary>
    /// Largest response body the client buffers. The biggest documented payloads (a 200-row page, the equity curve)
    /// are well under 1 MB; a larger body is a misbehaving backend or proxy, not data.
    /// </summary>
    internal const long MaxResponseBytes = 16 * 1024 * 1024;

    /// <summary>
    /// Registers <see cref="ISoliaApiClient"/> as a typed <see cref="HttpClient"/> on
    /// <c>{Backend:BaseUrl}/api/v1/</c> whose pipeline is, outermost first: total timeout → one retry (transport
    /// failure, timeout, 408/429/5xx except 500/501; never a non-idempotent method; no Retry-After wait) →
    /// per-attempt timeout → bounded response buffering. Needs
    /// <see cref="DashboardServiceCollectionExtensions.AddDashboardOptions"/>. Returns the client builder so callers
    /// (tests) can swap the handler.
    /// </summary>
    public static IHttpClientBuilder AddSoliaApiClient(this IServiceCollection services)
    {
        IHttpClientBuilder builder = services.AddHttpClient<ISoliaApiClient, SoliaApiClient>((sp, client) =>
        {
            BackendOptions options = sp.GetRequiredService<IOptions<BackendOptions>>().Value;
            client.BaseAddress = ApiBaseAddress(options.BaseUrl);
            client.Timeout = Timeout.InfiniteTimeSpan; // the resilience pipeline owns every timeout
        });

        builder.AddResilienceHandler(PipelineName, (pipeline, context) =>
        {
            BackendOptions options = context.ServiceProvider.GetRequiredService<IOptions<BackendOptions>>().Value;

            pipeline.AddTimeout(options.TotalTimeout);
            var retry = new HttpRetryStrategyOptions
            {
                MaxRetryAttempts = 1,
                Delay = BackendOptions.RetryDelay,
                BackoffType = DelayBackoffType.Constant,
                UseJitter = true,
                // A Retry-After (e.g. 30 s on a 503) would outlast TotalTimeout and hide the backend's own error.
                ShouldRetryAfterHeader = false,
                ShouldHandle = args => ValueTask.FromResult(IsTransient(args.Outcome)),
            };
            retry.DisableForUnsafeHttpMethods();
            pipeline.AddRetry(retry);
            pipeline.AddTimeout(options.AttemptTimeout);
        });

        // Inside the per-attempt timeout: HttpClient would otherwise buffer the body after the pipeline returned,
        // where no timeout bounds a server that sends headers and then stalls.
        builder.AddHttpMessageHandler(() => new ResponseBufferingHandler());

        return builder;
    }

    /// <summary>
    /// Whether an attempt failed for a reason a second attempt may not hit — the library's classification
    /// (<see cref="HttpClientResiliencePredicates.IsTransient(Outcome{HttpResponseMessage})"/>)
    /// minus the deterministic answers: a 500/501 and an over-sized body would only repeat.
    /// </summary>
    internal static bool IsTransient(Outcome<HttpResponseMessage> outcome) =>
        HttpClientResiliencePredicates.IsTransient(outcome)
        && outcome.Result?.StatusCode is not (HttpStatusCode.InternalServerError or HttpStatusCode.NotImplemented)
        && !IsTooLarge(outcome.Exception);

    /// <summary>The body exceeded <see cref="MaxResponseBytes"/>.</summary>
    internal static bool IsTooLarge(Exception? ex) =>
        ex is HttpRequestException { HttpRequestError: HttpRequestError.ConfigurationLimitExceeded };

    internal static Uri ApiBaseAddress(string baseUrl) => BackendOptions.Resolve(baseUrl, ApiPrefix);

    /// <summary>
    /// Reads the whole response body within the attempt (and its timeout's token), up to
    /// <see cref="MaxResponseBytes"/>: beyond it, an <see cref="HttpRequestException"/> with
    /// <see cref="HttpRequestError.ConfigurationLimitExceeded"/>.
    /// </summary>
    private sealed class ResponseBufferingHandler : DelegatingHandler
    {
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            HttpResponseMessage response = await base.SendAsync(request, cancellationToken).ConfigureAwait(false);
            try
            {
                await response.Content.LoadIntoBufferAsync(MaxResponseBytes, cancellationToken).ConfigureAwait(false);
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
