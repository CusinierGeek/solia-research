namespace Solia.DataPipeline.Options;

/// <summary>
/// pumpapi.io endpoints. Bound from the "PumpApi" config section; both values are also
/// overridable via the flat PUMPAPI_WS_URL / PUMPAPI_REPLAY_BASE_URL env vars
/// (mirrors .env.example). See docs/DATA-SOURCES.md. Validated at startup.
/// </summary>
public sealed class PumpApiOptions
{
    public const string Section = "PumpApi";

    /// <summary>Live WebSocket stream (server pushes all events; filter client-side).</summary>
    public string WsUrl { get; set; } = "wss://stream.pumpapi.io/";

    /// <summary>Base URL for the historical replay archives (YYYY/MM/DD/HH.jsonl.zst).</summary>
    public string ReplayBaseUrl { get; set; } = "https://replay.pumpapi.io";

    public bool HasValidUrls() =>
        IsAbsolute(WsUrl, "ws", "wss") && IsAbsolute(ReplayBaseUrl, "http", "https");

    private static bool IsAbsolute(string url, params string[] schemes) =>
        Uri.TryCreate(url, UriKind.Absolute, out Uri? uri) && schemes.Contains(uri.Scheme);
}
