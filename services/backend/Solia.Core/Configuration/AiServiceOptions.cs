using System.ComponentModel.DataAnnotations;

namespace Solia.Core.Configuration;

/// <summary>
/// How the backend reaches the internal ai-service, bound from the "AiService" config section and
/// validated at startup. <see cref="BaseUrl"/> is also overridable via the flat AI_SERVICE_URL env key
/// (mirrors .env.example). The remaining knobs tune the client's resilience pipeline: a total timeout
/// around retries with exponential backoff + jitter, a circuit breaker, and a per-attempt timeout.
/// </summary>
public sealed class AiServiceOptions
{
    public const string Section = "AiService";

    /// <summary>Absolute http(s) URL of the ai-service (mirrors AI_SERVICE_URL). Defaults to local dev.</summary>
    [Required]
    public string BaseUrl { get; set; } = "http://localhost:8000";

    /// <summary>Upper bound for one call, retries and backoff delays included.</summary>
    [Range(typeof(TimeSpan), "00:00:00.010", "01:00:00")]
    public TimeSpan TotalTimeout { get; set; } = TimeSpan.FromSeconds(15);

    /// <summary>Upper bound for a single HTTP attempt (body included); an attempt that times out is retried.</summary>
    [Range(typeof(TimeSpan), "00:00:00.010", "01:00:00")]
    public TimeSpan AttemptTimeout { get; set; } = TimeSpan.FromSeconds(4);

    /// <summary>Retries after the first attempt on transient failures of idempotent calls (0 disables).</summary>
    [Range(0, 10)]
    public int MaxRetryAttempts { get; set; } = 2;

    /// <summary>Base delay of the exponential, jittered backoff between retries.</summary>
    [Range(typeof(TimeSpan), "00:00:00", "00:01:00")]
    public TimeSpan RetryBaseDelay { get; set; } = TimeSpan.FromMilliseconds(200);

    /// <summary>Share of failed attempts within <see cref="CircuitBreakerSamplingDuration"/> that opens the circuit.</summary>
    [Range(0.01, 1.0)]
    public double CircuitBreakerFailureRatio { get; set; } = 0.5;

    /// <summary>Minimum attempts within the sampling window before the breaker may open.</summary>
    [Range(2, 1000)]
    public int CircuitBreakerMinimumThroughput { get; set; } = 10;

    /// <summary>Sliding window the failure ratio is computed over.</summary>
    [Range(typeof(TimeSpan), "00:00:00.500", "01:00:00")]
    public TimeSpan CircuitBreakerSamplingDuration { get; set; } = TimeSpan.FromSeconds(30);

    /// <summary>How long the circuit stays open (calls fail fast) before a trial call is let through.</summary>
    [Range(typeof(TimeSpan), "00:00:00.500", "01:00:00")]
    public TimeSpan CircuitBreakerBreakDuration { get; set; } = TimeSpan.FromSeconds(15);

    /// <summary>
    /// True when every attempt fits in <see cref="TotalTimeout"/>: (<see cref="MaxRetryAttempts"/> + 1) ×
    /// <see cref="AttemptTimeout"/> plus the nominal exponential backoff (<see cref="RetryBaseDelay"/> × (2^retries − 1)).
    /// Otherwise later retries would be cut short — or, with AttemptTimeout ≥ TotalTimeout, never happen. The
    /// total timeout remains a backstop for jitter.
    /// </summary>
    public bool HasConsistentTimeouts()
    {
        double attempts = (MaxRetryAttempts + 1) * AttemptTimeout.TotalMilliseconds;
        double backoff = RetryBaseDelay.TotalMilliseconds * (Math.Pow(2, MaxRetryAttempts) - 1);
        return attempts + backoff <= TotalTimeout.TotalMilliseconds;
    }

    /// <summary>True when <see cref="BaseUrl"/> is an absolute http(s) URL.</summary>
    public bool HasValidBaseUrl() =>
        Uri.TryCreate(BaseUrl, UriKind.Absolute, out Uri? uri)
        && (uri.Scheme == Uri.UriSchemeHttp || uri.Scheme == Uri.UriSchemeHttps);
}
