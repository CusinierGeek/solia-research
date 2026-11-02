using System.Collections.Concurrent;
using System.Net;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Time.Testing;
using Solia.DataPipeline.Metadata;

namespace Solia.DataPipeline.Tests.Metadata;

/// <summary>FE-T12 (#196): the single, bounded, background metadata fetch per token and its failure paths.</summary>
public class MetadataFetcherTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private const string Uri = "https://meta.example.com/m/1.json";

    private readonly FakeTimeProvider _time = new(T0);
    private readonly MemoryMetadataStore _store = new();
    private readonly MetadataCache _cache = new(TimeSpan.FromHours(48));

    private MetadataFetcher Fetcher(FakeHandler handler, MetadataOptions? options = null,
        Func<TimeSpan, CancellationToken, Task>? delay = null) =>
        new(handler, _cache, _store, options ?? new MetadataOptions(), NullLogger<MetadataFetcher>.Instance, _time,
            delay ?? ((_, _) => Task.CompletedTask));

    private static FakeHandler Respond(HttpStatusCode status, string body, Action<HttpResponseMessage>? configure = null) =>
        new((_, _) =>
        {
            var response = new HttpResponseMessage(status) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
            configure?.Invoke(response);
            return Task.FromResult(response);
        });

    [Fact]
    public async Task A_fetched_token_keeps_only_its_link_flags_with_the_completion_time()
    {
        FakeHandler handler = Respond(HttpStatusCode.OK,
            """{"name":"N","description":"secret plan","twitter":"https://x.com/n","telegram":"https://t.me/n","website":""}""");
        _time.Advance(TimeSpan.FromSeconds(3));

        TokenMetadataRecord record = (await Fetcher(handler).FetchAsync("M", Uri, default))!;

        Assert.Equal(new TokenMetadataRecord("M", T0.AddSeconds(3), MetadataFetchStatus.Ok, false, true, true), record);
        Assert.Equal(record, Assert.Single(_store.Saved));
        Assert.Equal(new MetadataLinks(false, true, true), _cache.Lookup("M", T0.AddSeconds(3)));
        Assert.Null(_cache.Lookup("M", T0.AddSeconds(2))); // point-in-time: not before the fetch completed
        Assert.Equal(Uri, Assert.Single(handler.Requests).AbsoluteUri);
    }

    [Fact]
    public async Task Ipfs_uris_are_fetched_through_the_configured_gateway()
    {
        FakeHandler handler = Respond(HttpStatusCode.OK, "{}");

        await Fetcher(handler).FetchAsync("M", "https://ipfs.io/ipfs/QmCid", default);

        Assert.Equal("https://4everland.io/ipfs/QmCid", Assert.Single(handler.Requests).AbsoluteUri);
    }

    public static TheoryData<string, FakeHandler> Failures => new()
    {
        { "unreachable gateway", new FakeHandler((_, _) => throw new HttpRequestException("connection refused")) },
        { "blocked address", new FakeHandler((_, _) => throw new HttpRequestException("metadata host does not resolve to a public address")) },
        { "HTTP 429", Respond(HttpStatusCode.TooManyRequests, "slow down") },
        { "HTTP 302", Respond(HttpStatusCode.Redirect, "", r => r.Headers.Location = new System.Uri("http://10.0.0.1/")) },
        { "malformed JSON", Respond(HttpStatusCode.OK, """{"twitter":""") },
        { "HTML page", Respond(HttpStatusCode.OK, "<html>gateway error</html>") },
        { "declared too large", Respond(HttpStatusCode.OK, "{}", r => r.Content.Headers.ContentLength = 10_000_000) },
        { "body too large", Respond(HttpStatusCode.OK, "{\"d\":\"" + new string('x', 70_000) + "\"}", r => r.Content.Headers.ContentLength = null) },
        { "broken stream", new FakeHandler((_, _) => throw new IOException("reset")) },
    };

    [Theory]
    [MemberData(nameof(Failures))]
    public async Task A_failed_fetch_is_recorded_once_as_failed_and_its_links_stay_unknown(string reason, FakeHandler handler)
    {
        TokenMetadataRecord record = (await Fetcher(handler).FetchAsync("M", Uri, default))!;

        Assert.True(record.Status == MetadataFetchStatus.Failed, reason);
        Assert.Equal(T0, record.FetchedAt);
        Assert.Single(_store.Saved);
        Assert.Null(_cache.Lookup("M", T0.AddHours(1)));
    }

    [Fact]
    public async Task A_request_that_outlives_the_timeout_fails_instead_of_hanging()
    {
        var handler = new FakeHandler(async (_, ct) =>
        {
            await Task.Delay(Timeout.Infinite, ct); // a gateway that never answers
            return new HttpResponseMessage(HttpStatusCode.OK);
        });

        TokenMetadataRecord? record = await Fetcher(handler, new MetadataOptions { TimeoutSeconds = 1 }).FetchAsync("M", Uri, default);

        Assert.Equal(MetadataFetchStatus.Failed, record!.Status);
    }

    [Fact]
    public async Task Shutdown_mid_request_records_nothing()
    {
        using var stopping = new CancellationTokenSource();
        var handler = new FakeHandler(async (_, ct) =>
        {
            await stopping.CancelAsync();
            await Task.Delay(Timeout.Infinite, ct);
            return new HttpResponseMessage(HttpStatusCode.OK);
        });

        Assert.Null(await Fetcher(handler).FetchAsync("M", Uri, stopping.Token));
        Assert.Empty(_store.Saved);
        Assert.False(_cache.Contains("M"));
    }

    [Theory]
    [InlineData("http://meta.example.com/m/1")]
    [InlineData(null)]
    public async Task An_unfetchable_uri_is_recorded_as_unsupported_without_a_request(string? uri)
    {
        FakeHandler handler = Respond(HttpStatusCode.OK, "{}");

        TokenMetadataRecord record = (await Fetcher(handler).FetchAsync("M", uri, default))!;

        Assert.Equal(MetadataFetchStatus.Unsupported, record.Status);
        Assert.Empty(handler.Requests);
        Assert.Single(_store.Saved);
    }

    [Fact]
    public async Task A_store_outage_keeps_the_result_in_memory()
    {
        _store.Fail = true;

        TokenMetadataRecord record = (await Fetcher(Respond(HttpStatusCode.OK, """{"website":"https://n.io"}""")).FetchAsync("M", Uri, default))!;

        Assert.Equal(MetadataFetchStatus.Ok, record.Status);
        Assert.NotNull(_cache.Lookup("M", T0));
    }

    [Fact]
    public void Enqueue_skips_known_queued_and_overflowing_tokens_and_everything_when_disabled()
    {
        MetadataFetcher fetcher = Fetcher(Respond(HttpStatusCode.OK, "{}"), new MetadataOptions { QueueCapacity = 2 });
        _cache.TryAdd(new TokenMetadataRecord("KNOWN", T0, MetadataFetchStatus.Ok));

        Assert.True(fetcher.TryEnqueue("A", Uri));
        Assert.False(fetcher.TryEnqueue("A", Uri)); // already queued
        Assert.False(fetcher.TryEnqueue("KNOWN", Uri)); // already fetched (e.g. loaded at startup)
        Assert.True(fetcher.TryEnqueue("B", Uri));
        Assert.False(fetcher.TryEnqueue("C", Uri)); // queue full: skipped, never blocks
        Assert.Equal(1, fetcher.Dropped);

        MetadataFetcher disabled = Fetcher(Respond(HttpStatusCode.OK, "{}"), new MetadataOptions { FetchEnabled = false });
        Assert.False(disabled.TryEnqueue("D", Uri));
    }

    [Fact]
    public async Task The_background_loop_fetches_each_queued_token_once_at_the_configured_rate()
    {
        var waits = new ConcurrentQueue<TimeSpan>();
        FakeHandler handler = Respond(HttpStatusCode.OK, """{"twitter":"https://x.com/n"}""");
        MetadataFetcher fetcher = Fetcher(handler, new MetadataOptions { MaxRequestsPerSecond = 2, MaxConcurrency = 1 },
            (d, _) =>
            {
                waits.Enqueue(d);
                _time.Advance(d); // the fake clock moves as the loop waits
                return Task.CompletedTask;
            });
        foreach (string mint in new[] { "A", "B", "C" })
        {
            fetcher.TryEnqueue(mint, Uri);
        }

        await fetcher.StartAsync(default);
        await WaitUntil(() => _store.Saved.Count == 3);
        await fetcher.StopAsync(default);

        Assert.Equal(["A", "B", "C"], _store.Saved.Select(r => r.Mint).Order().ToArray());
        Assert.Equal(3, handler.Requests.Count);
        Assert.Equal([TimeSpan.FromMilliseconds(500), TimeSpan.FromMilliseconds(500)], waits.ToArray()); // 2 / s
        Assert.False(fetcher.TryEnqueue("A", Uri)); // fetched: never again
    }

    private static async Task WaitUntil(Func<bool> condition)
    {
        for (int i = 0; i < 500 && !condition(); i++)
        {
            await Task.Delay(10);
        }

        Assert.True(condition());
    }

    public sealed class FakeHandler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> respond) : HttpMessageHandler
    {
        public ConcurrentQueue<System.Uri> Requests { get; } = new();

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Enqueue(request.RequestUri!);
            return respond(request, cancellationToken);
        }
    }

    internal sealed class NoQueue : IMetadataFetchQueue
    {
        public bool TryEnqueue(string mint, string? uri) => false;
    }

    internal sealed class MemoryMetadataStore : IMetadataStore
    {
        public bool Fail { get; set; }

        public ConcurrentQueue<TokenMetadataRecord> SavedQueue { get; } = new();

        public IReadOnlyList<TokenMetadataRecord> Saved => SavedQueue.ToList();

        public List<TokenMetadataRecord> Stored { get; } = [];

        public List<(DateTimeOffset Since, DateTimeOffset? Until)> Loads { get; } = [];

        public Task<IReadOnlyList<TokenMetadataRecord>> LoadAsync(DateTimeOffset since, DateTimeOffset? until, CancellationToken cancellationToken)
        {
            Loads.Add((since, until));
            return Task.FromResult<IReadOnlyList<TokenMetadataRecord>>(
                Stored.Where(r => r.FetchedAt >= since && (until is null || r.FetchedAt <= until)).ToList());
        }

        public Task SaveAsync(TokenMetadataRecord record, CancellationToken cancellationToken)
        {
            if (Fail)
            {
                throw new InvalidOperationException("database down");
            }

            SavedQueue.Enqueue(record);
            return Task.CompletedTask;
        }
    }
}
