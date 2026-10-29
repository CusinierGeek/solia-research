using Solia.DataPipeline.Ingestion.PumpApi;

namespace Solia.DataPipeline.Pipeline;

/// <summary>
/// The bounded retry shared by sink writes, replay hours and mirror downloads: up to
/// <see cref="MaxAttempts"/> attempts with jittered exponential backoff between them
/// (<see cref="PumpApiStreamSource.ComputeBackoff"/>), then give up. Cancellation of the caller's
/// token is never retried.
/// </summary>
public sealed record RetryPolicy(int MaxAttempts, double BaseDelaySeconds = 1, double MaxDelaySeconds = 30)
{
    /// <summary>The wait before attempt <paramref name="failedAttempt"/> + 1, or null when that was the last attempt.</summary>
    public TimeSpan? DelayAfter(int failedAttempt) =>
        failedAttempt >= MaxAttempts
            ? null
            : PumpApiStreamSource.ComputeBackoff(failedAttempt - 1, BaseDelaySeconds, MaxDelaySeconds, Random.Shared.NextDouble());

    /// <summary>
    /// Runs <paramref name="operation"/> until it succeeds (true) or every attempt failed (false).
    /// <paramref name="onFailure"/> sees each failure with its attempt number and the wait before the
    /// next one (null = giving up). Cancellation of <paramref name="cancellationToken"/> throws.
    /// </summary>
    public async Task<bool> RunAsync(
        Func<CancellationToken, Task> operation,
        Func<TimeSpan, CancellationToken, Task> delay,
        Action<Exception, int, TimeSpan?> onFailure,
        CancellationToken cancellationToken)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(MaxAttempts, 1);
        for (int attempt = 1; ; attempt++)
        {
            try
            {
                await operation(cancellationToken);
                return true;
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                // Includes a TaskCanceledException from an operation's own timeout: a failure, not a shutdown.
                TimeSpan? wait = DelayAfter(attempt);
                onFailure(ex, attempt, wait);
                if (wait is not { } w)
                {
                    return false;
                }

                await delay(w, cancellationToken);
            }
        }
    }
}
