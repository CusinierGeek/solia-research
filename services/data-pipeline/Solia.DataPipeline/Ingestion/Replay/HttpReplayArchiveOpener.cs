using System.Globalization;
using System.Net;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Opens an hour's replay archive over HTTP: <c>{ReplayBaseUrl}/yyyy/MM/dd/HH.jsonl.zst</c>.
/// Returns the compressed body stream, or <c>null</c> when the hour is absent (404); any other
/// failure throws (<see cref="PumpApiReplaySource"/> retries it). The body is wrapped in an
/// <see cref="IdleTimeoutStream"/>, so a download that stalls mid-hour fails instead of hanging.
/// </summary>
public sealed class HttpReplayArchiveOpener(HttpClient httpClient, IOptions<PumpApiOptions> options, TimeSpan readIdleTimeout)
    : IReplayArchive
{
    /// <summary>How long a body read may make no progress before the attempt fails.</summary>
    public static readonly TimeSpan DefaultReadIdleTimeout = TimeSpan.FromSeconds(60);

    public HttpReplayArchiveOpener(HttpClient httpClient, IOptions<PumpApiOptions> options)
        : this(httpClient, options, DefaultReadIdleTimeout)
    {
    }

    public async Task<Stream?> OpenAsync(DateTimeOffset hour, CancellationToken cancellationToken)
    {
        string url = BuildUrl(options.Value.ReplayBaseUrl, hour);
        HttpResponseMessage response =
            await httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);

        try
        {
            if (response.StatusCode == HttpStatusCode.NotFound)
            {
                response.Dispose();
                return null;
            }

            response.EnsureSuccessStatusCode();

            // Disposing the returned content stream releases the connection; the lightweight
            // response wrapper is then collected. The caller (PumpApiReplaySource) disposes it.
            return new IdleTimeoutStream(await response.Content.ReadAsStreamAsync(cancellationToken), readIdleTimeout);
        }
        catch
        {
            // Error status or cancellation: release the response (and its pooled connection).
            response.Dispose();
            throw;
        }
    }

    internal static string BuildUrl(string baseUrl, DateTimeOffset hour)
    {
        DateTimeOffset utc = hour.ToUniversalTime();
        return string.Create(CultureInfo.InvariantCulture, $"{baseUrl.TrimEnd('/')}/{utc:yyyy}/{utc:MM}/{utc:dd}/{utc:HH}.jsonl.zst");
    }
}
