using System.Text.Json;

namespace Solia.Bot.Policy;

/// <summary>
/// What the eligibility rule reads of a token, as of one of its <c>features</c> rows: the most distinct traders and
/// the highest window SOL volume and the oldest token age any snapshot up to that row reported (null when none
/// reported one).
/// </summary>
public readonly record struct EligibilityInputs(int? MaxUniqueTraders, double? MaxVolumeSol, double? MaxTokenAgeSec);

/// <summary>
/// Which tokens the bot may score and enter: the very rule the model's training windows were filtered by (ai-service
/// <c>solia_ai.eligibility</c>; the shared fixture <c>services/shared/testdata/eligibility-cases.json</c> keeps the two
/// in lockstep). Causal: a row is judged on that row and the token's earlier rows only. A row is eligible when every
/// enabled criterion holds — a threshold of 0 disables its criterion, so <see cref="Disabled"/> accepts every token:
/// <list type="bullet">
///   <item><see cref="MinUniqueTraders"/> ≤ the most distinct traders a snapshot counted so far;</item>
///   <item><see cref="MinVolumeSol"/> ≤ the highest window SOL volume a snapshot reported so far;</item>
///   <item><see cref="MinTokenAgeSec"/> ≤ the oldest token age known so far (the row's, or the last known one).</item>
/// </list>
/// A NULL input carries no information: it doesn't raise a running maximum, so a token with no known value yet fails
/// that (enabled) criterion — an unknown age can't be proven old enough. Every criterion is a running maximum, so once
/// eligible a token stays eligible. The rule is configured once, in the ai-service, and
/// recorded with every model (<c>models.hyperparams.eligibility</c>): the bot reads its champion's.
/// </summary>
public sealed record EligibilityRule
{
    public static readonly EligibilityRule Disabled = new(0, 0, 0);

    public EligibilityRule(int minUniqueTraders, double minVolumeSol, double minTokenAgeSec)
    {
        Check(nameof(MinUniqueTraders), minUniqueTraders);
        Check(nameof(MinVolumeSol), minVolumeSol);
        Check(nameof(MinTokenAgeSec), minTokenAgeSec);
        MinUniqueTraders = minUniqueTraders;
        MinVolumeSol = minVolumeSol;
        MinTokenAgeSec = minTokenAgeSec;
    }

    public int MinUniqueTraders { get; }

    public double MinVolumeSol { get; }

    public double MinTokenAgeSec { get; }

    public bool Enabled => this != Disabled;

    public bool IsEligible(EligibilityInputs inputs) =>
        (MinUniqueTraders == 0 || inputs.MaxUniqueTraders >= MinUniqueTraders)
        && (MinVolumeSol == 0 || inputs.MaxVolumeSol >= MinVolumeSol)
        && (MinTokenAgeSec == 0 || inputs.MaxTokenAgeSec >= MinTokenAgeSec);

    /// <summary>
    /// Each row's eligibility over one token's rows in time order (unique traders, window SOL volume, token age; null
    /// = NULL): the running maxima the database path computes in SQL, here in memory.
    /// </summary>
    public IReadOnlyList<bool> Evaluate(IEnumerable<(int? UniqueTraders, double? VolumeSol, double? TokenAgeSec)> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);
        List<bool> eligible = [];
        int? traders = null;
        double? volume = null, age = null;
        foreach ((int? rowTraders, double? rowVolume, double? rowAge) in rows)
        {
            traders = Max(traders, rowTraders);
            volume = Max(volume, rowVolume);
            age = Max(age, rowAge);
            eligible.Add(IsEligible(new EligibilityInputs(traders, volume, age)));
        }

        return eligible;
    }

    /// <summary>
    /// The rule a model was trained under, from its <c>models.hyperparams</c> JSON: its <c>eligibility</c> object
    /// (an omitted threshold is 0). None recorded — a model from before the rule, trained on every token — is
    /// <see cref="Disabled"/>.
    /// </summary>
    /// <exception cref="FormatException">The JSON is malformed, or the recorded rule is: an unknown key, a
    /// non-numeric, fractional trader count, or a negative / non-finite threshold.</exception>
    public static EligibilityRule FromHyperparams(string? hyperparamsJson)
    {
        if (string.IsNullOrWhiteSpace(hyperparamsJson))
        {
            return Disabled;
        }

        try
        {
            using JsonDocument doc = JsonDocument.Parse(hyperparamsJson);
            if (doc.RootElement.ValueKind != JsonValueKind.Object
                || !doc.RootElement.TryGetProperty("eligibility", out JsonElement rule)
                || rule.ValueKind == JsonValueKind.Null)
            {
                return Disabled;
            }

            return FromJson(rule);
        }
        catch (JsonException ex)
        {
            throw new FormatException($"the model's hyperparams are not valid JSON: {ex.Message}", ex);
        }
    }

    /// <summary>The rule of an <c>eligibility</c> JSON object (see <see cref="FromHyperparams"/>).</summary>
    /// <exception cref="FormatException">An invalid rule.</exception>
    public static EligibilityRule FromJson(JsonElement rule)
    {
        if (rule.ValueKind != JsonValueKind.Object)
        {
            throw new FormatException($"'eligibility' must be an object, got {rule.ValueKind}");
        }

        int traders = 0;
        double volume = 0, age = 0;
        foreach (JsonProperty property in rule.EnumerateObject())
        {
            switch (property.Name)
            {
                case "min_unique_traders":
                    traders = property.Value.ValueKind == JsonValueKind.Number && property.Value.TryGetInt32(out int count)
                        ? count
                        : throw new FormatException($"'eligibility.min_unique_traders' must be an integer, got {property.Value}");
                    break;
                case "min_volume_sol":
                    volume = Number(property);
                    break;
                case "min_token_age_sec":
                    age = Number(property);
                    break;
                default:
                    throw new FormatException($"'eligibility' has an unknown key: {property.Name}");
            }
        }

        try
        {
            return new EligibilityRule(traders, volume, age);
        }
        catch (ArgumentOutOfRangeException ex)
        {
            throw new FormatException($"invalid 'eligibility' rule: {ex.Message}", ex);
        }
    }

    public override string ToString() => Enabled
        ? $"traders>={MinUniqueTraders}, volume>={MinVolumeSol:G} SOL, age>={MinTokenAgeSec:G}s"
        : "disabled (every token)";

    private static double Number(JsonProperty property) =>
        property.Value.ValueKind == JsonValueKind.Number
            ? property.Value.GetDouble()
            : throw new FormatException($"'eligibility.{property.Name}' must be a number, got {property.Value}");

    private static void Check(string name, double value)
    {
        if (!double.IsFinite(value) || value < 0)
        {
            throw new ArgumentOutOfRangeException(name, value, $"{name} must be a finite number >= 0 (0 disables it).");
        }
    }

    private static int? Max(int? a, int? b) => a is null ? b : b is null ? a : Math.Max(a.Value, b.Value);

    private static double? Max(double? a, double? b) => a is null ? b : b is null ? a : Math.Max(a.Value, b.Value);
}
