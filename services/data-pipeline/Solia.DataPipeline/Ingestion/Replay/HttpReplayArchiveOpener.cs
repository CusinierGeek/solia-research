using System.Net;
using Microsoft.Extensions.Options;
using Solia.DataPipeline.Options;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// Opens an hour's replay archive over HTTP: <c>{ReplayBaseUrl}/yyyy/MM/dd/HH.jsonl.zst</c>.
/// Returns the compressed body stream, or <c>null</c> when the hour is absent (404); any other
/// failure throws (<see cref="PumpApiReplaySource"/> retries it). Designed to be passed as
/// <see cref="PumpApiReplaySource"/>'s archive opener.
/// </summary>
public sealed class HttpReplayArchiveOpener(HttpClient httpClient, IOptions<PumpApiOptions> options)
{
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
            return await response.Content.ReadAsStreamAsync(cancellationToken);
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
        return $"{baseUrl.TrimEnd('/')}/{utc:yyyy}/{utc:MM}/{utc:dd}/{utc:HH}.jsonl.zst";
    }
}
