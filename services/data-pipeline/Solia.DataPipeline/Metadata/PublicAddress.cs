using System.Net;
using System.Net.Sockets;

namespace Solia.DataPipeline.Metadata;

/// <summary>
/// Guards the metadata fetch against server-side request forgery: the URI comes from whoever launched the token, so
/// it could point at the pipeline's own network (PostgreSQL, the backend, a cloud metadata endpoint). Every
/// connection goes through <see cref="ConnectAsync"/>, which resolves the host itself and connects only to a public
/// address — the checked one, so a DNS answer can't change between the check and the connect.
/// </summary>
public static class PublicAddress
{
    /// <summary>
    /// False for loopback, private, link-local, carrier-grade NAT, benchmarking, multicast, reserved and unspecified
    /// addresses, and for IPv6 forms that embed an IPv4 one (NAT64, 6to4, Teredo; a mapped one is checked as IPv4).
    /// </summary>
    public static bool IsPublic(IPAddress address)
    {
        if (address.IsIPv4MappedToIPv6)
        {
            address = address.MapToIPv4();
        }

        if (address.AddressFamily == AddressFamily.InterNetwork)
        {
            byte[] b = address.GetAddressBytes();
            return !(b[0] == 0
                     || b[0] == 10
                     || b[0] == 127
                     || (b[0] == 100 && b[1] >= 64 && b[1] <= 127) // 100.64/10 carrier-grade NAT
                     || (b[0] == 169 && b[1] == 254)                // link-local (cloud metadata endpoints)
                     || (b[0] == 172 && b[1] >= 16 && b[1] <= 31)
                     || (b[0] == 192 && b[1] == 0 && b[2] == 0)     // 192.0.0/24 IETF protocol assignments
                     || (b[0] == 192 && b[1] == 168)
                     || (b[0] == 198 && b[1] is 18 or 19)           // benchmarking
                     || b[0] >= 224);                               // multicast, reserved, broadcast
        }

        if (address.AddressFamily != AddressFamily.InterNetworkV6)
        {
            return false;
        }

        byte[] v6 = address.GetAddressBytes();
        bool nat64 = v6[0] == 0x00 && v6[1] == 0x64 && v6[2] == 0xff && v6[3] == 0x9b;
        bool sixToFour = v6[0] == 0x20 && v6[1] == 0x02;
        bool teredo = v6[0] == 0x20 && v6[1] == 0x01 && v6[2] == 0x00 && v6[3] == 0x00;
        return !(IPAddress.IPv6Loopback.Equals(address)
                 || IPAddress.IPv6None.Equals(address)
                 || address.IsIPv6LinkLocal
                 || address.IsIPv6SiteLocal
                 || address.IsIPv6Multicast
                 || (v6[0] & 0xfe) == 0xfc // fc00::/7 unique local
                 || nat64 || sixToFour || teredo);
    }

    /// <summary><see cref="SocketsHttpHandler.ConnectCallback"/>: connects to the first public address of the host, or fails.</summary>
    public static async ValueTask<Stream> ConnectAsync(SocketsHttpConnectionContext context, CancellationToken cancellationToken)
    {
        IPAddress[] addresses = await Dns.GetHostAddressesAsync(context.DnsEndPoint.Host, cancellationToken);
        IPAddress target = addresses.FirstOrDefault(IsPublic)
            ?? throw new HttpRequestException("metadata host does not resolve to a public address");
        var socket = new Socket(target.AddressFamily, SocketType.Stream, ProtocolType.Tcp) { NoDelay = true };
        try
        {
            await socket.ConnectAsync(new IPEndPoint(target, context.DnsEndPoint.Port), cancellationToken);
            return new NetworkStream(socket, ownsSocket: true);
        }
        catch
        {
            socket.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The handler of the metadata client: public addresses only, no automatic redirect (the fetcher follows them
    /// itself, so each hop gets the URI checks — <see cref="MetadataUri.ResolveRedirect"/>), no proxy (it would connect for us, past the address check), no cookies.
    /// </summary>
    public static SocketsHttpHandler CreateHandler() => new()
    {
        ConnectCallback = ConnectAsync,
        AllowAutoRedirect = false,
        UseProxy = false,
        UseCookies = false,
        AutomaticDecompression = DecompressionMethods.All,
        PooledConnectionLifetime = TimeSpan.FromMinutes(5),
    };
}
