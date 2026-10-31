using Solia.Bot.Configuration;
using Solia.Bot.Sessions;

namespace Solia.Bot.Tests.Sessions;

public class BotSessionFingerprintTests
{
    private static string Json(BotOptions options, params BotRegimePart[] parts) => BotSessionFingerprint.OptionsJson(options, parts);

    [Fact]
    public void Equal_options_give_the_same_fingerprint() =>
        Assert.Equal(Json(new BotOptions()), Json(new BotOptions()));

    [Fact]
    public void A_number_compares_by_value_not_by_spelling() =>
        Assert.Equal(Json(new BotOptions { StartingBalance = 10m }), Json(new BotOptions { StartingBalance = 10.000m }));

    [Fact]
    public void Venue_keys_compare_case_insensitively()
    {
        var lower = new BotOptions();
        var upper = new BotOptions();
        upper.FeePct.Remove("pumpfun");
        upper.FeePct["PumpFun"] = 0.0125;

        Assert.Equal(Json(lower), Json(upper));
    }

    [Fact]
    public void The_session_label_is_not_part_of_the_regime() =>
        Assert.Equal(Json(new BotOptions()), Json(new BotOptions { SessionLabel = "tuned" }));

    [Theory]
    [MemberData(nameof(RegimeChanges))]
    public void Changing_the_entry_rule_exits_fills_or_sizing_changes_the_regime(Action<BotOptions> change)
    {
        var changed = new BotOptions();
        change(changed);
        Assert.NotEqual(Json(new BotOptions()), Json(changed));
    }

    public static TheoryData<Action<BotOptions>> RegimeChanges() =>
    [
        o => o.MinPumpProb = 0.3,
        o => o.MaxRugProb = 0.15,
        o => o.TakeProfitPct = 0.6,
        o => o.PositionTimeoutSeconds = 600,
        o => o.RealisticFills = false,
        o => o.LiquidityFloorSol["default"] = 2m,
        o => o.MaxPositionFraction = 0.1,
        o => o.StartingBalance = 20m,
        o => o.EntryLatencyMs = 0,
        o => o.ExitLatencyMs = 500,
        o => o.NetworkFeeSol = 0m,
        o => o.AccountRentSol = 0m,
        o => o.MaxSlippagePct = 0.3,
        o => o.MaxExitRetries = 1,
    ];

    [Fact]
    public void Host_parts_are_part_of_the_regime()
    {
        string m15 = Json(new BotOptions(), new BotRegimePart("horizon", "15m"));
        Assert.Contains("\"horizon\":\"15m\"", m15);
        Assert.NotEqual(m15, Json(new BotOptions(), new BotRegimePart("horizon", "5m")));
    }

    [Fact]
    public void The_hash_covers_the_champion()
    {
        string json = Json(new BotOptions());
        string hash = BotSessionFingerprint.Hash(json, "v1");

        Assert.Equal(64, hash.Length);
        Assert.Equal(hash, BotSessionFingerprint.Hash(json, "v1"));
        Assert.NotEqual(hash, BotSessionFingerprint.Hash(json, "v2"));
        Assert.NotEqual(hash, BotSessionFingerprint.Hash(json, null));
    }
}
