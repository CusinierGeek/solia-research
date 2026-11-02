using System.Text.Json;
using Solia.DataPipeline.Ingestion;
using Solia.DataPipeline.Ingestion.PumpApi;

namespace Solia.DataPipeline.Tests.Ingestion;

public class PumpApiEventParserTests
{
    [Theory]
    [InlineData("create", EventKind.Create)]
    [InlineData("buy", EventKind.Buy)]
    [InlineData("sell", EventKind.Sell)]
    [InlineData("migrate", EventKind.Migrate)]
    [InlineData("createPool", EventKind.CreatePool)]
    [InlineData("add", EventKind.AddLiquidity)]
    [InlineData("remove", EventKind.RemoveLiquidity)]
    [InlineData("distributeFees", EventKind.Other)] // recognized-but-uninteresting, still passes through
    public void Maps_action_to_kind(string action, EventKind expected)
    {
        string json = $$"""{"action":"{{action}}","mint":"MINT1","pool":"pump","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(expected, evt!.Kind);
        Assert.Equal("MINT1", evt.Mint);
    }

    [Fact]
    public void Parses_a_buy_with_all_fields()
    {
        const string json = """
        {"action":"buy","mint":"MINT1","quoteMint":"SOL","pool":"pump","txSigner":"TRADER1",
         "tokenAmount":137837918.03,"quoteAmount":5.06,"tokensInPool":590131414.28,"quoteInPool":6.998,
         "price":0.00000425,"marketCapQuote":42.5,"signature":"SIG1","block":1234,"timestamp":1700000000000}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(EventKind.Buy, evt!.Kind);
        Assert.Equal("MINT1", evt.Mint);
        Assert.Equal("pumpfun", evt.Venue);
        Assert.False(evt.IsMigrated);
        Assert.True(evt.IsBuy);
        Assert.Equal(0.00000425m, evt.Price);
        Assert.Equal(5.06m, evt.QuoteAmount);
        Assert.Equal(42.5m, evt.MarketCap);
        Assert.Equal("TRADER1", evt.Trader);
        Assert.Equal("SIG1", evt.Signature);
        Assert.Equal(1234, evt.Block);
        Assert.Equal(DateTimeOffset.FromUnixTimeMilliseconds(1700000000000), evt.Timestamp);
    }

    [Fact]
    public void Sell_on_amm_is_migrated_and_not_a_buy()
    {
        const string json =
            """{"action":"sell","mint":"MINT1","pool":"pump-amm","price":0.001,"timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(EventKind.Sell, evt!.Kind);
        Assert.True(evt.IsMigrated);
        Assert.False(evt.IsBuy);
    }

    [Fact]
    public void A_creates_dev_buy_and_signer_are_kept()
    {
        const string json = """
        {"txType":"create","mint":"MINT1","pool":"pump","txSigner":"DEV","initialBuy":96449438.144093,
         "solAmount":2.962962962,"supply":1000000000,"timestamp":1700000000000}
        """;

        RawEvent evt = PumpApiEventParser.Parse(json)!;

        Assert.Equal("DEV", evt.Trader);
        Assert.Equal(96449438.144093m, evt.TokenAmount);
        Assert.Equal(2.962962962m, evt.QuoteAmount);
        Assert.Null(PumpApiEventParser.Parse(json.Replace("create", "createPool"))!.TokenAmount); // only a create's
    }

    [Fact]
    public void Parses_a_create_with_metadata_and_scientific_price()
    {
        const string json = """
        {"action":"create","mint":"MINT1","pool":"pump","name":"TokenName","symbol":"SYM",
         "supply":1000000000.0,"vTokensInBondingCurve":919714285.71,"vQuoteInBondingCurve":35.0,
         "price":3.805e-8,"marketCapQuote":38.05,"timestamp":1700000000000}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(EventKind.Create, evt!.Kind);
        Assert.Equal("TokenName", evt.Name);
        Assert.Equal("SYM", evt.Symbol);
        Assert.Equal(1000000000.0m, evt.Supply);
        Assert.Equal(919714285.71m, evt.VirtualTokenReserves);
        Assert.NotNull(evt.Price);
        Assert.Null(evt.IsBuy); // not a trade
    }

    [Fact]
    public void Transfer_takes_mint_from_the_transfers_array()
    {
        const string json = """
        {"action":"transfer","timestamp":1700000000000,
         "transfers":[{"from":"A","to":"B","amount":500.0,"isSolana":false,"mint":"MINTX"}]}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(EventKind.Transfer, evt!.Kind);
        Assert.Equal("MINTX", evt.Mint);
    }

    [Fact]
    public void Derives_price_from_real_reserves_on_amm_when_price_absent()
    {
        const string json = """
        {"action":"buy","mint":"MINT1","pool":"pump-amm","tokensInPool":1000,"quoteInPool":10,"timestamp":1700000000000}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(0.01m, evt!.Price); // 10 / 1000
    }

    [Fact]
    public void Bonding_curve_price_prefers_virtual_reserves_over_real_ones()
    {
        const string json = """
        {"action":"buy","mint":"MINT1","pool":"pump","tokensInPool":793000000,"quoteInPool":0.01,
         "vTokensInBondingCurve":1000000000,"vQuoteInBondingCurve":30,"timestamp":1700000000000}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(0.00000003m, evt!.Price); // 30 / 1e9, not 0.01 / 793e6
    }

    [Fact]
    public void Price_is_null_instead_of_throwing_when_the_division_overflows()
    {
        const string json = """
        {"action":"buy","mint":"MINT1","pool":"pump","vQuoteInBondingCurve":79000000000000000000000000000,
         "vTokensInBondingCurve":0.001,"timestamp":1700000000000}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Null(evt!.Price);
    }

    [Fact]
    public void Parses_utf8_bytes_like_a_string()
    {
        byte[] utf8 = System.Text.Encoding.UTF8.GetBytes(
            """{"action":"buy","mint":"MINT1","pool":"pump","price":0.5,"timestamp":1700000000000}""");

        RawEvent? evt = PumpApiEventParser.ParseUtf8(utf8.AsMemory());

        Assert.NotNull(evt);
        Assert.Equal("MINT1", evt!.Mint);
        Assert.Equal(0.5m, evt.Price);
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json")]
    public void Malformed_utf8_bytes_return_null(string text)
    {
        Assert.Null(PumpApiEventParser.ParseUtf8(System.Text.Encoding.UTF8.GetBytes(text).AsMemory()));
    }

    [Fact]
    public void Price_is_null_when_neither_price_nor_reserves_present()
    {
        const string json = """{"action":"buy","mint":"MINT1","pool":"pump","tokensInPool":0,"quoteInPool":1,"timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Null(evt!.Price);
    }

    [Theory]
    [InlineData("1700000000")]          // seconds
    [InlineData("1700000000000")]       // milliseconds (pumpapi's unit)
    [InlineData("1700000000000000")]    // microseconds
    [InlineData("1700000000000000000")] // nanoseconds
    [InlineData("\"1700000000000\"")]   // numeric string
    public void Timestamp_unit_is_normalized(string rawTimestamp)
    {
        string json = $$"""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":{{rawTimestamp}}}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(DateTimeOffset.FromUnixTimeMilliseconds(1700000000000), evt!.Timestamp);
    }

    [Theory]
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump"}""")]                         // missing
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":"soon"}""")]      // unparseable
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":0}""")]           // zero
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":-5}""")]          // negative
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":1000}""")]        // 1970
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":5000000000}""")]  // year 2128 in seconds
    [InlineData("""{"action":"buy","mint":"MINT1","pool":"pump","timestamp":1e30}""")]        // absurd magnitude
    public void Missing_or_implausible_timestamp_returns_null_instead_of_now(string json)
    {
        Assert.Null(PumpApiEventParser.Parse(json));
    }

    [Fact]
    public void Transfer_skips_sol_and_wsol_legs()
    {
        const string json = """
        {"action":"transfer","timestamp":1700000000000,
         "transfers":[{"isSolana":true,"mint":"So11111111111111111111111111111111111111112"},
                      {"isSolana":false,"mint":"So11111111111111111111111111111111111111112"},
                      {"isSolana":false,"mint":"TOKEN"}]}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal("TOKEN", evt!.Mint);
    }

    [Fact]
    public void Transfer_with_only_sol_legs_returns_null()
    {
        const string json = """
        {"action":"transfer","timestamp":1700000000000,
         "transfers":[{"isSolana":true,"mint":"So11111111111111111111111111111111111111112"},"garbage"]}
        """;

        Assert.Null(PumpApiEventParser.Parse(json));
    }

    [Fact]
    public void Empty_root_mint_falls_back_to_transfers()
    {
        const string json = """
        {"action":"transfer","mint":"  ","timestamp":1700000000000,"transfers":[{"mint":"MINTX"}]}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal("MINTX", evt!.Mint);
    }

    [Fact]
    public void Migrate_on_bonding_venue_is_migrated()
    {
        const string json = """{"action":"migrate","mint":"MINT1","pool":"pump","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(EventKind.Migrate, evt!.Kind);
        Assert.True(evt.IsMigrated);
    }

    [Theory]
    [InlineData("pump", "pumpfun")]
    [InlineData("pump-amm", "pumpswap")]
    [InlineData("raydium-cpmm", "raydium")]
    [InlineData("meteora-dlmm", "meteora")]
    [InlineData("orca-whirlpool", "orca")]
    [InlineData("Some-New-Dex", "some-new-dex")] // unknown passes through, normalized
    public void Venue_is_mapped_to_canonical_vocabulary(string pool, string expected)
    {
        string json = $$"""{"action":"buy","mint":"MINT1","pool":"{{pool}}","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(expected, evt!.Venue);
    }

    [Fact]
    public void Missing_pool_leaves_venue_null_rather_than_a_fake_value()
    {
        const string json = """{"action":"buy","mint":"MINT1","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Null(evt!.Venue);
        Assert.False(evt.IsMigrated);
    }

    [Theory]
    [InlineData("BUY")]
    [InlineData("CreatePool")]
    public void Action_matching_is_case_insensitive(string action)
    {
        string json = $$"""{"action":"{{action}}","mint":"MINT1","pool":"pump","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.NotEqual(EventKind.Other, evt!.Kind);
    }

    [Fact]
    public void Element_from_disposed_document_returns_null_instead_of_throwing()
    {
        JsonElement element;
        using (JsonDocument doc = JsonDocument.Parse("""{"action":"buy","mint":"MINT1","timestamp":1700000000000}"""))
        {
            element = doc.RootElement;
        }

        Assert.Null(PumpApiEventParser.Parse(element));
    }

    [Fact]
    public void IsBuy_is_derived_from_kind()
    {
        Assert.True(new RawEvent { Mint = "M", Kind = EventKind.Buy, Timestamp = DateTimeOffset.UnixEpoch }.IsBuy);
        Assert.False(new RawEvent { Mint = "M", Kind = EventKind.Sell, Timestamp = DateTimeOffset.UnixEpoch }.IsBuy);
        Assert.Null(new RawEvent { Mint = "M", Kind = EventKind.Transfer, Timestamp = DateTimeOffset.UnixEpoch }.IsBuy);
    }

    [Theory]
    [InlineData("100%", 100.0)]
    [InlineData("0%", 0.0)]
    [InlineData("57.5%", 57.5)]
    public void Parses_burned_liquidity_percent(string raw, double expected)
    {
        string json =
            $$"""{"action":"createPool","mint":"MINT1","pool":"raydium-cpmm","burnedLiquidity":"{{raw}}","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(expected, evt!.BurnedLiquidityPercent);
    }

    [Theory]
    [InlineData("NaN%")]
    [InlineData("Infinity%")]
    [InlineData("-1%")]
    [InlineData("150%")]
    [InlineData("lots")]
    public void Invalid_burned_liquidity_percent_is_null(string raw)
    {
        string json =
            $$"""{"action":"createPool","mint":"MINT1","pool":"raydium-cpmm","burnedLiquidity":"{{raw}}","timestamp":1700000000000}""";

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Null(evt!.BurnedLiquidityPercent);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("not json")]
    [InlineData("[]")]                                              // not an object
    [InlineData("{}")]                                             // no action
    [InlineData("""{"action":"buy"}""")]                           // no mint
    [InlineData("""{"action":"createConfig","platformName":"bonk"}""")] // token-less event
    public void Malformed_or_unattributable_returns_null(string json)
    {
        Assert.Null(PumpApiEventParser.Parse(json));
    }

    [Fact]
    public void Real_schema_txType_and_sol_fields_are_read()
    {
        const string json = """
        {"txType":"sell","mint":"MINT1","pool":"pump","tokenAmount":1354150.784229,"solAmount":0.224586568,
         "tokensInPool":161334602.262203,"solInPool":42.954387564,"vTokensInBondingCurve":441234602.262203,
         "vSolInBondingCurve":72.954387564,"marketCapSol":165.34149223556807,"timestamp":1776686399967}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal(EventKind.Sell, evt!.Kind);
        Assert.Equal("So11111111111111111111111111111111111111112", evt.QuoteMint);
        Assert.Equal(0.224586568m, evt.QuoteAmount);
        Assert.Equal(42.954387564m, evt.QuoteInPool);
        Assert.Equal(72.954387564m, evt.VirtualQuoteReserves);
        Assert.Equal(165.34149223556807m, evt.MarketCap);
        Assert.Equal(72.954387564m / 441234602.262203m, evt.Price); // derived from the virtual SOL reserves
    }

    [Fact]
    public void Non_sol_quoted_pool_keeps_its_quote_mint_and_quote_fields()
    {
        const string json = """
        {"txType":"buy","mint":"MINT1","pool":"meteora-launchpad","quoteMint":"EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
         "quoteAmount":1.614924,"quoteInPool":21401.387071,"marketCapQuote":138278.885,"price":0.000138,"timestamp":1776686400811}
        """;

        RawEvent? evt = PumpApiEventParser.Parse(json);

        Assert.NotNull(evt);
        Assert.Equal("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", evt!.QuoteMint);
        Assert.Equal(1.614924m, evt.QuoteAmount);
        Assert.Equal(21401.387071m, evt.QuoteInPool);
        Assert.Equal(138278.885m, evt.MarketCap);
    }

    [Fact]
    public void TxType_wins_over_action_when_both_are_present()
    {
        const string json = """{"txType":"sell","action":"buy","mint":"MINT1","timestamp":1776686400000}""";

        Assert.Equal(EventKind.Sell, PumpApiEventParser.Parse(json)!.Kind);
    }

    [Fact]
    public void Transfer_of_native_sol_only_is_unattributable()
    {
        const string json = """
        {"txType":"transfer","timestamp":1776686399967,
         "transfers":[{"amount":0.0001,"isSolana":true,"mint":"So11111111111111111111111111111111111111111"}]}
        """;

        Assert.Null(PumpApiEventParser.Parse(json));
    }

    [Fact]
    public void Event_without_any_price_fields_has_no_quote_mint()
    {
        const string json = """{"txType":"buy","mint":"MINT1","pool":"pump","timestamp":1776686400000}""";

        Assert.Null(PumpApiEventParser.Parse(json)!.QuoteMint);
    }

    [Fact]
    public void Null_input_returns_null()
    {
        Assert.Null(PumpApiEventParser.Parse(null));
    }
}
