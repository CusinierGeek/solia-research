using System.Net;
using System.Text;
using Solia.DataPipeline.Metadata;

namespace Solia.DataPipeline.Tests.Metadata;

/// <summary>FE-T12 (#196): metadata URI handling, link parsing, the SSRF address guard and the point-in-time cache.</summary>
public class MetadataParsingTests
{
    private static readonly DateTimeOffset T0 = new(2026, 4, 20, 12, 0, 0, TimeSpan.Zero);
    private static readonly MetadataOptions Options = new();

    [Theory]
    [InlineData("https://ipfs.io/ipfs/QmRpvaErvBrkNSX4rVFtmFnQcs8af9PWNrKrvypXhNvtjx", true)]
    [InlineData("https://pump.mypinata.cloud/ipfs/bafkreicqvxenpyxs3nwtzh67zpvddoyhmstyxzxuzzxutblbdlvgriz3tq", true)]
    [InlineData("ipfs://QmRpvaErvBrkNSX4rVFtmFnQcs8af9PWNrKrvypXhNvtjx", true)]
    [InlineData("https://arweave.net/vkpUTtURQ2woUodsx2SpMx5GvtJt3770oFkz1P2az3o", true)]
    [InlineData("https://metadata.j7tracker.io/metadata/f826af1434c641eb.json", false)]
    [InlineData("https://meta.uxento.io/data/1a7356a4-7188-4fe6-91c0-f52c89dee967", false)]
    [InlineData("not a uri", null)]
    [InlineData("", null)]
    [InlineData(null, null)]
    public void Content_addressed_uris_are_told_apart_from_launch_tool_hosts(string? uri, bool? expected)
    {
        Assert.Equal(expected, MetadataUri.IsContentAddressed(uri));
    }

    [Theory]
    [InlineData("https://ipfs.io/ipfs/QmRpvaErvBrkNSX4rVFtmFnQcs8af9PWNrKrvypXhNvtjx",
        "https://4everland.io/ipfs/QmRpvaErvBrkNSX4rVFtmFnQcs8af9PWNrKrvypXhNvtjx")]
    [InlineData("ipfs://QmRpvaErvBrkNSX4rVFtmFnQcs8af9PWNrKrvypXhNvtjx",
        "https://4everland.io/ipfs/QmRpvaErvBrkNSX4rVFtmFnQcs8af9PWNrKrvypXhNvtjx")] // the CID keeps its case
    [InlineData("ipfs://ipfs/QmAbc/meta.json", "https://4everland.io/ipfs/QmAbc/meta.json")]
    [InlineData("ar://vkpUTtURQ2wo", "https://arweave.net/vkpUTtURQ2wo")]
    [InlineData("https://meta.uxento.io/data/1a7356a4", "https://meta.uxento.io/data/1a7356a4")]
    [InlineData("https://meta.example.com:443/m/1", "https://meta.example.com/m/1")]
    public void Fetch_urls_rewrite_content_addressed_uris_to_the_gateways(string uri, string expected)
    {
        Assert.Equal(expected, MetadataUri.ResolveFetchUrl(uri, Options)?.AbsoluteUri);
    }

    [Theory]
    [InlineData("http://meta.example.com/m/1")]          // not HTTPS
    [InlineData("https://meta.example.com:8443/m/1")]    // non-default port
    [InlineData("https://user:pw@meta.example.com/m/1")] // user info
    [InlineData("file:///etc/passwd")]
    [InlineData("data:application/json,{}")]
    [InlineData("ipfs://")]
    [InlineData("garbage")]
    [InlineData(null)]
    public void Uris_that_must_not_be_fetched_resolve_to_nothing(string? uri)
    {
        Assert.Null(MetadataUri.ResolveFetchUrl(uri, Options));
    }

    [Fact]
    public void Links_are_read_from_the_top_level_or_extensions_and_must_be_non_blank()
    {
        static MetadataLinks? Parse(string json) => MetadataLinks.Parse(Encoding.UTF8.GetBytes(json));

        Assert.Equal(new MetadataLinks(false, true, false), Parse(
            """{"name":"Mimicoin","twitter":"https://x.com/a/status/1","website":"","telegram":"  "}"""));
        Assert.Equal(new MetadataLinks(true, false, true), Parse(
            """{"name":"N","extensions":{"website":"https://n.io","telegram":"https://t.me/n"}}"""));
        Assert.Equal(new MetadataLinks(false, false, false), Parse("""{"name":"N","twitter":null,"website":42}"""));
        Assert.Equal(new MetadataLinks(false, true, false), Parse("""{"x":"https://x.com/n"}"""));
    }

    [Theory]
    [InlineData("{\"twitter\":")]
    [InlineData("<html>429 Too Many Requests</html>")]
    [InlineData("[{\"twitter\":\"x\"}]")]
    [InlineData("\"twitter\"")]
    [InlineData("")]
    public void Malformed_metadata_parses_to_nothing(string body)
    {
        Assert.Null(MetadataLinks.Parse(Encoding.UTF8.GetBytes(body)));
    }

    [Theory]
    [InlineData("8.8.8.8", true)]
    [InlineData("104.18.2.1", true)]
    [InlineData("2606:4700::6810:1", true)]
    [InlineData("127.0.0.1", false)]
    [InlineData("10.1.2.3", false)]
    [InlineData("172.18.0.5", false)] // a docker network (postgres, backend)
    [InlineData("192.168.1.10", false)]
    [InlineData("169.254.169.254", false)] // cloud metadata endpoint
    [InlineData("100.64.0.1", false)]
    [InlineData("0.0.0.0", false)]
    [InlineData("224.0.0.1", false)]
    [InlineData("255.255.255.255", false)]
    [InlineData("::1", false)]
    [InlineData("::", false)]
    [InlineData("fe80::1", false)]
    [InlineData("fd00::1", false)]
    [InlineData("::ffff:10.0.0.1", false)] // IPv4-mapped private
    [InlineData("::ffff:8.8.8.8", true)]
    [InlineData("64:ff9b::a00:1", false)]  // NAT64 of 10.0.0.1
    [InlineData("2002:a00:1::1", false)]   // 6to4 of 10.0.0.1
    public void Only_public_addresses_may_be_connected_to(string address, bool expected)
    {
        Assert.Equal(expected, PublicAddress.IsPublic(IPAddress.Parse(address)));
    }

    [Fact]
    public void The_metadata_client_never_follows_redirects_nor_uses_a_proxy()
    {
        using SocketsHttpHandler handler = PublicAddress.CreateHandler();

        Assert.False(handler.AllowAutoRedirect);
        Assert.False(handler.UseProxy);
        Assert.NotNull(handler.ConnectCallback);
    }

    [Fact]
    public void A_snapshot_sees_links_only_from_the_fetch_time_and_within_the_horizon()
    {
        var cache = new MetadataCache(TimeSpan.FromHours(48));
        cache.TryAdd(new TokenMetadataRecord("OK", T0, MetadataFetchStatus.Ok, HasTwitter: true));
        cache.TryAdd(new TokenMetadataRecord("FAILED", T0, MetadataFetchStatus.Failed));

        Assert.Null(cache.Lookup("OK", T0.AddTicks(-1))); // fetched after the snapshot: invisible
        Assert.Equal(new MetadataLinks(false, true, false), cache.Lookup("OK", T0));
        Assert.NotNull(cache.Lookup("OK", T0.AddHours(47)));
        Assert.Null(cache.Lookup("OK", T0.AddHours(48)));
        Assert.Null(cache.Lookup("FAILED", T0.AddHours(1))); // a failure is unknown, never "no link"
        Assert.Null(cache.Lookup("UNKNOWN", T0.AddHours(1)));
    }

    [Fact]
    public void The_first_result_wins_and_pruning_forgets_only_expired_results()
    {
        var cache = new MetadataCache(TimeSpan.FromHours(48));
        Assert.True(cache.TryAdd(new TokenMetadataRecord("A", T0, MetadataFetchStatus.Failed)));
        Assert.False(cache.TryAdd(new TokenMetadataRecord("A", T0.AddMinutes(1), MetadataFetchStatus.Ok, true, true, true)));
        cache.Load([new TokenMetadataRecord("B", T0.AddHours(10), MetadataFetchStatus.Ok)]);

        Assert.Equal(1, cache.Prune(T0.AddHours(48)));

        Assert.False(cache.Contains("A"));
        Assert.True(cache.Contains("B"));
    }
}
