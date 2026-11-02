using System.Collections.Concurrent;
using System.Threading.Channels;
using Microsoft.Extensions.Options;

namespace Solia.DataPipeline.Metadata;

/// <summary>
/// Fetches each new token's metadata JSON once (FE-T12, issue #196) and keeps only whether it lists a website / X /
/// Telegram link. Live ingestion only, never on the hot path: the pipeline loop hands tokens over with a
/// non-blocking <see cref="TryEnqueue"/> (a full queue skips the token) and this service fetches them in the
/// background, rate-limited (<see cref="MetadataOptions.MaxRequestsPerSecond"/>, <see cref="MetadataOptions.MaxConcurrency"/>),
/// bounded (<see cref="MetadataOptions.TimeoutSeconds"/>, <see cref="MetadataOptions.MaxResponseBytes"/>) and through
/// <see cref="PublicAddress"/> (the URI is attacker-controlled).
/// <para>
/// Each outcome — success, failure or unsupported URI — is recorded once with its completion time (<c>fetched_at</c>),
/// in the <see cref="MetadataCache"/> and in <c>token_metadata</c>, and never retried: a token is fetched at most
/// once, and a snapshot only sees links fetched by its ts. Replays never fetch: they read the stored results.
/// </para>
/// </summary>
public sealed class MetadataFetcher : BackgroundService, IMetadataFetchQueue
{
    internal const int SummaryEvery = 1_000;

    /// <summary>Redirects followed per fetch (a gateway's path → subdomain redirect is one).</summary>
    internal const int MaxRedirects = 3;

    private readonly Channel<Request> _queue;
    private readonly ConcurrentDictionary<string, byte> _pending = new(StringComparer.Ordinal);
    // Failure reason (HTTP status, timeout, exception type...) → count, for the periodic summary. A small, bounded set.
    private readonly ConcurrentDictionary<string, long> _failureReasons = new(StringComparer.Ordinal);
    private readonly HttpClient _http;
    private readonly MetadataCache _cache;
    private readonly IMetadataStore _store;
    private readonly MetadataOptions _options;
    private readonly ILogger<MetadataFetcher> _logger;
    private readonly TimeProvider _time;
    private readonly Func<TimeSpan, CancellationToken, Task> _delay;

    private long _ok;
    private long _failed;
    private long _unsupported;
    private long _dropped;
    private long _saveFailures;

    public MetadataFetcher(MetadataCache cache, IMetadataStore store, IOptions<MetadataOptions> options, ILogger<MetadataFetcher> logger)
        : this(PublicAddress.CreateHandler(), cache, store, options.Value, logger, TimeProvider.System, static (d, ct) => Task.Delay(d, ct))
    {
    }

    // Test seam: a fake handler (no network), clock and delay.
    internal MetadataFetcher(
        HttpMessageHandler handler,
        MetadataCache cache,
        IMetadataStore store,
        MetadataOptions options,
        ILogger<MetadataFetcher> logger,
        TimeProvider time,
        Func<TimeSpan, CancellationToken, Task> delay)
    {
        // The per-request timeout covers the body read too, so the client's own is off.
        _http = new HttpClient(handler, disposeHandler: true) { Timeout = Timeout.InfiniteTimeSpan };
        _http.DefaultRequestHeaders.Accept.ParseAdd("application/json");
        _cache = cache;
        _store = store;
        _options = options;
        _logger = logger;
        _time = time;
        _delay = delay;
        _queue = Channel.CreateBounded<Request>(new BoundedChannelOptions(options.QueueCapacity)
        {
            SingleReader = true,
            FullMode = BoundedChannelFullMode.Wait, // TryWrite fails when full: the token is skipped, ingestion never waits
        });
    }

    public long Fetched => Interlocked.Read(ref _ok);
    public long Failed => Interlocked.Read(ref _failed);
    public long Unsupported => Interlocked.Read(ref _unsupported);

    /// <summary>Tokens skipped because the queue was full.</summary>
    public long Dropped => Interlocked.Read(ref _dropped);

    public bool TryEnqueue(string mint, string? uri)
    {
        if (!_options.FetchEnabled || _cache.Contains(mint) || !_pending.TryAdd(mint, 0))
        {
            return false;
        }

        if (_queue.Writer.TryWrite(new Request(mint, uri)))
        {
            return true;
        }

        _pending.TryRemove(mint, out _);
        Interlocked.Increment(ref _dropped);
        return false;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        if (!_options.FetchEnabled)
        {
            _logger.LogInformation("metadata fetch disabled (Metadata:FetchEnabled=false): link features stay NULL");
            return;
        }

        using var gate = new SemaphoreSlim(_options.MaxConcurrency);
        TimeSpan interval = TimeSpan.FromSeconds(1 / _options.MaxRequestsPerSecond);
        DateTimeOffset next = _time.GetUtcNow();
        try
        {
            await foreach (Request request in _queue.Reader.ReadAllAsync(stoppingToken))
            {
                DateTimeOffset now = _time.GetUtcNow();
                if (next > now)
                {
                    await _delay(next - now, stoppingToken);
                }

                next = (next > now ? next : now) + interval;
                await gate.WaitAsync(stoppingToken);
                _ = Task.Run(async () =>
                {
                    try
                    {
                        await FetchAsync(request.Mint, request.Uri, stoppingToken);
                    }
                    finally
                    {
                        gate.Release();
                    }
                }, CancellationToken.None);
            }
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Shutdown: queued tokens are not fetched (their links stay NULL).
        }

        // Let in-flight fetches observe the cancellation before the gate is disposed.
        for (int i = 0; i < _options.MaxConcurrency; i++)
        {
            await gate.WaitAsync(CancellationToken.None);
        }
    }

    /// <summary>
    /// Fetches one token's metadata, records the outcome (cache, then <c>token_metadata</c>) and returns it; null when
    /// cancelled (shutdown), in which case nothing is recorded.
    /// </summary>
    internal async Task<TokenMetadataRecord?> FetchAsync(string mint, string? uri, CancellationToken cancellationToken)
    {
        try
        {
            TokenMetadataRecord record = await RequestAsync(mint, uri, cancellationToken);
            if (_cache.TryAdd(record))
            {
                await SaveAsync(record, cancellationToken);
            }

            Count(record.Status);
            return record;
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            return null;
        }
        finally
        {
            _pending.TryRemove(mint, out _);
        }
    }

    private async Task<TokenMetadataRecord> RequestAsync(string mint, string? uri, CancellationToken cancellationToken)
    {
        if (MetadataUri.ResolveFetchUrl(uri, _options) is not { } url)
        {
            return new TokenMetadataRecord(mint, _time.GetUtcNow(), MetadataFetchStatus.Unsupported);
        }

        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(_options.TimeoutSeconds));
        string? failure = null;
        MetadataLinks? links = null;
        try
        {
            // Redirects are followed here, not by the handler, so every hop gets the same URI checks (HTTPS, default
            // port, no user info, no private IP literal); a hostname hop still connects through PublicAddress.
            Uri current = url;
            for (int hop = 0; failure is null && links is null; hop++)
            {
                using HttpResponseMessage response = await _http.GetAsync(current, HttpCompletionOption.ResponseHeadersRead, timeout.Token);
                if (MetadataUri.IsRedirect(response.StatusCode))
                {
                    if (hop == MaxRedirects)
                    {
                        failure = "too many redirects";
                    }
                    else if (MetadataUri.ResolveRedirect(current, response.Headers.Location) is { } next)
                    {
                        current = next;
                    }
                    else
                    {
                        failure = "redirect refused";
                    }
                }
                else if (!response.IsSuccessStatusCode)
                {
                    failure = $"HTTP {(int)response.StatusCode}";
                }
                else if (await ReadBoundedAsync(response.Content, _options.MaxResponseBytes, timeout.Token) is not { } body)
                {
                    failure = "response too large";
                }
                else
                {
                    links = MetadataLinks.Parse(body);
                    failure = links is null ? "malformed JSON" : null;
                }
            }
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            failure = "timeout";
        }
        catch (Exception ex) when (ex is HttpRequestException or IOException or InvalidDataException or NotSupportedException)
        {
            failure = ex.GetType().Name; // never the message: it may echo the URI
        }

        DateTimeOffset fetchedAt = _time.GetUtcNow();
        if (links is { } l)
        {
            return new TokenMetadataRecord(mint, fetchedAt, MetadataFetchStatus.Ok, l.HasWebsite, l.HasTwitter, l.HasTelegram);
        }

        failure ??= "unknown";
        _failureReasons.AddOrUpdate(failure, 1, static (_, n) => n + 1);
        _logger.LogDebug("metadata fetch failed for {Mint}: {Failure}", mint, failure);
        return new TokenMetadataRecord(mint, fetchedAt, MetadataFetchStatus.Failed);
    }

    /// <summary>The body, or null when it is larger than <paramref name="maxBytes"/> (whatever Content-Length claims).</summary>
    private static async Task<byte[]?> ReadBoundedAsync(HttpContent content, int maxBytes, CancellationToken cancellationToken)
    {
        if (content.Headers.ContentLength > maxBytes)
        {
            return null;
        }

        await using Stream stream = await content.ReadAsStreamAsync(cancellationToken);
        var buffer = new byte[maxBytes + 1];
        int total = 0;
        int read;
        while (total < buffer.Length && (read = await stream.ReadAsync(buffer.AsMemory(total), cancellationToken)) > 0)
        {
            total += read;
        }

        return total > maxBytes ? null : buffer[..total];
    }

    private async Task SaveAsync(TokenMetadataRecord record, CancellationToken cancellationToken)
    {
        try
        {
            await _store.SaveAsync(record, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            // The cache still holds it for this run; only a restart (or a replay) loses it.
            Interlocked.Increment(ref _saveFailures);
            _logger.LogWarning(ex, "could not store the metadata fetch result of {Mint}", record.Mint);
        }
    }

    private void Count(MetadataFetchStatus status)
    {
        long done = status switch
        {
            MetadataFetchStatus.Ok => Interlocked.Increment(ref _ok) + Failed + Unsupported,
            MetadataFetchStatus.Failed => Interlocked.Increment(ref _failed) + Fetched + Unsupported,
            _ => Interlocked.Increment(ref _unsupported) + Fetched + Failed,
        };

        if (done % SummaryEvery == 0)
        {
            _logger.LogInformation(
                "metadata fetch: {Ok} ok, {Failed} failed, {Unsupported} unsupported, {Dropped} skipped (queue full), " +
                "{SaveFailures} not stored; {Cached} cached; failures by reason: {FailureReasons}",
                Fetched, Failed, Unsupported, Dropped, Interlocked.Read(ref _saveFailures), _cache.Count, FailureSummary());
        }
    }

    /// <summary>"HTTP 404 ×12, timeout ×3", most frequent first; "none" when nothing failed.</summary>
    internal string FailureSummary()
    {
        KeyValuePair<string, long>[] reasons = _failureReasons.ToArray();
        return reasons.Length == 0
            ? "none"
            : string.Join(", ", reasons.OrderByDescending(r => r.Value).ThenBy(r => r.Key, StringComparer.Ordinal)
                .Select(r => $"{r.Key} ×{r.Value}"));
    }

    public override void Dispose()
    {
        _http.Dispose();
        base.Dispose();
    }

    private readonly record struct Request(string Mint, string? Uri);
}
