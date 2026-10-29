using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Time.Testing;
using Npgsql;
using Solia.Api.Bot;
using Solia.Bot.Policy;
using Solia.Core.Data;

namespace Solia.Api.Tests.Bot;

/// <summary>
/// <see cref="ChampionEligibility"/> against a real Postgres: the champion's recorded rule, its refresh, a champion
/// without a rule or with an invalid one, a failed read, and the running-maxima query the rule judges a row on.
/// Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public sealed class ChampionEligibilityIntegrationTests : PostgresFixture
{
    private static readonly DateTimeOffset T0 = new(2026, 1, 10, 0, 0, 0, TimeSpan.Zero);

    private const string Rule3 =
        "{\"eligibility\": {\"min_unique_traders\": 3, \"min_volume_sol\": 0.5, \"min_token_age_sec\": 0.0}}";

    private readonly FakeTimeProvider _time = new(T0);

    private SoliaDbContext Db(string? connectionString = null) =>
        new(new DbContextOptionsBuilder<SoliaDbContext>().UseNpgsql(connectionString ?? ConnectionString).Options);

    private ChampionEligibility Provider() => new(_time, NullLogger<ChampionEligibility>.Instance);

    [Fact]
    public async Task The_champion_s_recorded_rule_is_read()
    {
        await ChampionAsync("v1", Rule3);
        ChampionEligibility provider = Provider();
        await using SoliaDbContext db = Db();

        EligibilityRule? rule = await provider.GetAsync(db, CancellationToken.None);

        Assert.Equal(new EligibilityRule(3, 0.5, 0), rule);
        Assert.Equal("v1", provider.Version);
    }

    [Fact]
    public async Task A_champion_without_a_rule_or_no_champion_scores_every_token()
    {
        await using SoliaDbContext db = Db();
        ChampionEligibility provider = Provider();
        Assert.Equal(EligibilityRule.Disabled, await provider.GetAsync(db, CancellationToken.None));
        Assert.Null(provider.Version);

        await ChampionAsync("legacy", "{\"model\": {\"seq_len\": 64}}");
        provider.Invalidate();

        Assert.Equal(EligibilityRule.Disabled, await provider.GetAsync(db, CancellationToken.None));
        Assert.Equal("legacy", provider.Version);
    }

    [Fact]
    public async Task An_invalid_recorded_rule_fails_closed()
    {
        await ChampionAsync("broken", "{\"eligibility\": {\"min_unique_traders\": -1}}");
        await using SoliaDbContext db = Db();

        Assert.Null(await Provider().GetAsync(db, CancellationToken.None));
    }

    [Fact]
    public async Task A_new_champion_s_rule_applies_after_the_refresh_interval_or_an_invalidation()
    {
        await ChampionAsync("v1", null);
        ChampionEligibility provider = Provider();
        await using SoliaDbContext db = Db();
        Assert.Equal(EligibilityRule.Disabled, await provider.GetAsync(db, CancellationToken.None));

        await ChampionAsync("v2", Rule3);
        Assert.Equal(EligibilityRule.Disabled, await provider.GetAsync(db, CancellationToken.None)); // cached

        _time.Advance(ChampionEligibility.RefreshInterval);
        Assert.Equal(new EligibilityRule(3, 0.5, 0), await provider.GetAsync(db, CancellationToken.None));

        await ChampionAsync("v3", null);
        provider.Invalidate();
        Assert.Equal(EligibilityRule.Disabled, await provider.GetAsync(db, CancellationToken.None));
        Assert.Equal("v3", provider.Version);
    }

    [Fact]
    public async Task A_failed_read_keeps_the_last_rule_and_the_first_one_throws()
    {
        await ChampionAsync("v1", Rule3);
        ChampionEligibility provider = Provider();
        await using (SoliaDbContext db = Db())
        {
            await provider.GetAsync(db, CancellationToken.None);
        }

        const string unreachable = "Host=127.0.0.1;Port=1;Database=solia;Username=solia;Timeout=1";
        await using SoliaDbContext down = Db(unreachable);
        _time.Advance(ChampionEligibility.RefreshInterval);

        Assert.Equal(new EligibilityRule(3, 0.5, 0), await provider.GetAsync(down, CancellationToken.None));
        Exception first = await Assert.ThrowsAnyAsync<Exception>(() => Provider().GetAsync(down, CancellationToken.None));
        Assert.True(first is NpgsqlException || first.InnerException is NpgsqlException, first.ToString());
    }

    [Fact]
    public async Task The_inputs_are_running_maxima_over_the_rows_up_to_the_scored_one()
    {
        await ExecAsync("INSERT INTO tokens_history (mint, first_seen_at) VALUES ('m', @t)", ("t", T0));
        (int? Traders, decimal? Volume, int? Age)[] rows = [(1, 0.2m, 0), (null, null, null), (5, 0.1m, 20), (2, 3.0m, 30)];
        for (int i = 0; i < rows.Length; i++)
        {
            await ExecAsync("INSERT INTO features (mint, ts, unique_traders, volume, token_age_sec) VALUES ('m', @ts, @u, @v, @a)",
                ("ts", T0.AddSeconds(10 * i)), ("u", (object?)rows[i].Traders ?? DBNull.Value),
                ("v", (object?)rows[i].Volume ?? DBNull.Value), ("a", (object?)rows[i].Age ?? DBNull.Value));
        }

        await using SoliaDbContext db = Db();

        Assert.Equal(new EligibilityInputs(1, 0.2, 0), await ChampionEligibility.LoadInputsAsync(db, "m", T0.AddSeconds(10), CancellationToken.None));
        Assert.Equal(new EligibilityInputs(5, 0.2, 20), await ChampionEligibility.LoadInputsAsync(db, "m", T0.AddSeconds(20), CancellationToken.None));
        Assert.Equal(new EligibilityInputs(5, 3.0, 30), await ChampionEligibility.LoadInputsAsync(db, "m", T0.AddSeconds(30), CancellationToken.None));
        Assert.Equal(default, await ChampionEligibility.LoadInputsAsync(db, "unknown", T0.AddSeconds(30), CancellationToken.None));
    }

    private async Task ChampionAsync(string version, string? hyperparams)
    {
        await ExecAsync("UPDATE models SET is_champion = FALSE WHERE is_champion");
        await ExecAsync("INSERT INTO models (version, path, is_champion, hyperparams) VALUES (@v, '/m.pt', TRUE, CAST(@h AS JSONB))",
            ("v", version), ("h", (object?)hyperparams ?? DBNull.Value));
    }

    private async Task ExecAsync(string sql, params (string Name, object Value)[] parameters)
    {
        await using NpgsqlCommand cmd = DataSource.CreateCommand(sql);
        foreach ((string name, object value) in parameters)
        {
            cmd.Parameters.AddWithValue(name, value);
        }

        await cmd.ExecuteNonQueryAsync();
    }
}
