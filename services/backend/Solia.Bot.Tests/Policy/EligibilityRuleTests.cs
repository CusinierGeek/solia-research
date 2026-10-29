using System.Text.Json;
using Solia.Bot.Policy;

namespace Solia.Bot.Tests.Policy;

/// <summary>
/// The bot's eligibility rule. The parity theories run the very fixture the ai-service's
/// <c>tests/test_eligibility.py</c> runs (<c>services/shared/testdata/eligibility-cases.json</c>), so the dataset's
/// training windows and the bot's entries are filtered by the same rule.
/// </summary>
public class EligibilityRuleTests
{
    private static readonly Lazy<JsonDocument> Fixture = new(() =>
        JsonDocument.Parse(File.ReadAllText(Path.Combine(FindTestData(), "eligibility-cases.json"))));

    public static TheoryData<string> CaseNames()
    {
        var names = new TheoryData<string>();
        foreach (JsonElement c in Fixture.Value.RootElement.GetProperty("cases").EnumerateArray())
        {
            names.Add(c.GetProperty("name").GetString()!);
        }

        return names;
    }

    public static TheoryData<string> InvalidRules()
    {
        var rules = new TheoryData<string>();
        foreach (JsonElement rule in Fixture.Value.RootElement.GetProperty("invalid_rules").EnumerateArray())
        {
            rules.Add(rule.GetRawText());
        }

        return rules;
    }

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void The_shared_fixture_cases(string name)
    {
        JsonElement c = Fixture.Value.RootElement.GetProperty("cases").EnumerateArray()
            .Single(e => e.GetProperty("name").GetString() == name);
        EligibilityRule rule = EligibilityRule.FromJson(c.GetProperty("rule"));
        var rows = c.GetProperty("rows").EnumerateArray()
            .Select(r => (Int(r, "unique_traders"), Double(r, "volume"), Double(r, "token_age_sec")))
            .ToList();

        IReadOnlyList<bool> eligible = rule.Evaluate(rows);

        Assert.Equal(c.GetProperty("eligible").EnumerateArray().Select(e => e.GetBoolean()), eligible);
    }

    [Theory]
    [MemberData(nameof(InvalidRules))]
    public void The_shared_fixture_invalid_rules_are_rejected(string json)
    {
        using JsonDocument rule = JsonDocument.Parse(json);

        FormatException ex = Assert.Throws<FormatException>(() => EligibilityRule.FromJson(rule.RootElement));
        Assert.Contains("eligibility", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("{}")]
    [InlineData("{\"model\": {\"seq_len\": 64}}")]
    [InlineData("{\"eligibility\": null}")]
    [InlineData("[1, 2]")]
    public void A_model_recorded_without_a_rule_is_the_disabled_rule(string? hyperparams)
    {
        Assert.Same(EligibilityRule.Disabled, EligibilityRule.FromHyperparams(hyperparams));
    }

    [Fact]
    public void The_rule_the_ai_service_records_is_read_back()
    {
        const string hyperparams =
            "{\"model\": {}, \"eligibility\": {\"min_unique_traders\": 3, \"min_volume_sol\": 0.5, \"min_token_age_sec\": 0.0}}";

        EligibilityRule rule = EligibilityRule.FromHyperparams(hyperparams);

        Assert.Equal(new EligibilityRule(3, 0.5, 0), rule);
        Assert.True(rule.Enabled);
        Assert.Contains("traders>=3", rule.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Zero_thresholds_are_the_disabled_rule()
    {
        Assert.Equal(EligibilityRule.Disabled, EligibilityRule.FromHyperparams("{\"eligibility\": {\"min_unique_traders\": 0}}"));
        Assert.False(EligibilityRule.Disabled.Enabled);
        Assert.True(EligibilityRule.Disabled.IsEligible(default)); // NULL inputs pass a disabled rule
        Assert.Contains("disabled", EligibilityRule.Disabled.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Malformed_hyperparams_json_is_a_format_error()
    {
        Assert.Throws<FormatException>(() => EligibilityRule.FromHyperparams("{not json"));
        Assert.Throws<FormatException>(() => EligibilityRule.FromHyperparams("{\"eligibility\": 3}"));
    }

    [Theory]
    [InlineData(-1, 0, 0)]
    [InlineData(0, -0.1, 0)]
    [InlineData(0, 0, -5)]
    [InlineData(0, double.NaN, 0)]
    [InlineData(0, 0, double.PositiveInfinity)]
    public void Negative_or_non_finite_thresholds_fail_fast(int traders, double volume, double age)
    {
        ArgumentOutOfRangeException ex = Assert.Throws<ArgumentOutOfRangeException>(() => new EligibilityRule(traders, volume, age));
        Assert.Contains(">= 0", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Every_enabled_criterion_must_hold_and_null_inputs_fail_them()
    {
        var rule = new EligibilityRule(2, 1.0, 30);

        Assert.True(rule.IsEligible(new EligibilityInputs(2, 1.0, 30)));
        Assert.False(rule.IsEligible(new EligibilityInputs(1, 1.0, 30)));
        Assert.False(rule.IsEligible(new EligibilityInputs(2, 0.99, 30)));
        Assert.False(rule.IsEligible(new EligibilityInputs(2, 1.0, 29)));
        Assert.False(rule.IsEligible(new EligibilityInputs(null, 1.0, 30)));
        Assert.False(rule.IsEligible(new EligibilityInputs(2, null, 30)));
        Assert.False(rule.IsEligible(new EligibilityInputs(2, 1.0, null)));
    }

    private static int? Int(JsonElement row, string name) =>
        row.GetProperty(name).ValueKind == JsonValueKind.Null ? null : row.GetProperty(name).GetInt32();

    private static double? Double(JsonElement row, string name) =>
        row.GetProperty(name).ValueKind == JsonValueKind.Null ? null : row.GetProperty(name).GetDouble();

    /// <summary>The repo's <c>services/shared/testdata</c>, found by walking up from the test output.</summary>
    private static string FindTestData()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "services", "shared", "testdata");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new DirectoryNotFoundException("Could not locate services/shared/testdata from the test output.");
    }
}
