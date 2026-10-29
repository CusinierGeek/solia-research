using System.Data.Common;
using Microsoft.EntityFrameworkCore;
using Solia.Bot.Policy;
using Solia.Core.Data;

namespace Solia.Api.Bot;

/// <summary>
/// Which tokens the feed may score: the <see cref="EligibilityRule"/> the champion was trained under, read from its
/// <c>models.hyperparams.eligibility</c> — the bot never configures its own, so it can't trade a population its
/// model never saw. Re-read every <see cref="RefreshInterval"/>, or sooner after <see cref="Invalidate"/> (the
/// ai-service answered with another model version). A champion recorded without a rule predates it and was trained
/// on every token: <see cref="EligibilityRule.Disabled"/>, with a warning. A champion whose recorded rule is invalid
/// is a contract drift: nothing is scored (fail closed) until it changes. Without any champion every token is
/// eligible (the ai-service refuses to score then anyway). A failed read keeps the last rule; with none yet it
/// throws, and that token's scoring is skipped. Not thread-safe: the feed's scoring loop is its only user.
/// </summary>
public sealed class ChampionEligibility(TimeProvider time, ILogger<ChampionEligibility> logger)
{
    internal static readonly TimeSpan RefreshInterval = TimeSpan.FromSeconds(30);

    private bool _loaded;
    private string? _version;
    private EligibilityRule? _rule;
    private long _checkedAt;

    /// <summary>The model version the current rule was read from; null without a champion.</summary>
    public string? Version => _version;

    /// <summary>Re-reads the champion's rule on the next <see cref="GetAsync"/>.</summary>
    public void Invalidate() => _loaded = false;

    /// <summary>The champion's rule; null when its recorded rule is invalid (score nothing).</summary>
    /// <exception cref="Exception">The first read failed (a <see cref="DbException"/>, which EF may wrap).</exception>
    public async Task<EligibilityRule?> GetAsync(SoliaDbContext db, CancellationToken cancellationToken)
    {
        if (_loaded && time.GetElapsedTime(_checkedAt) < RefreshInterval)
        {
            return _rule;
        }

        try
        {
            var champion = await db.Models.AsNoTracking()
                .Where(m => m.IsChampion)
                .Select(m => new { m.Version, m.Hyperparams })
                .FirstOrDefaultAsync(cancellationToken);
            Apply(champion?.Version, champion?.Hyperparams);
        }
        catch (Exception ex) when (_loaded && IsDbFailure(ex) && !cancellationToken.IsCancellationRequested)
        {
            logger.LogWarning(ex, "bot feed: reading the champion's eligibility rule failed; keeping {Rule}", _rule);
        }

        _loaded = true;
        _checkedAt = time.GetTimestamp();
        return _rule;
    }

    // EF wraps a transient failure (e.g. a refused connection) in an InvalidOperationException.
    private static bool IsDbFailure(Exception ex) => ex is DbException || ex.InnerException is DbException;

    private void Apply(string? version, string? hyperparams)
    {
        bool changed = !_loaded || version != _version;
        _version = version;
        if (version is null)
        {
            _rule = EligibilityRule.Disabled;
            if (changed)
            {
                logger.LogInformation("bot feed: no champion registered; every token is eligible");
            }

            return;
        }

        EligibilityRule? previous = _rule;
        try
        {
            _rule = EligibilityRule.FromHyperparams(hyperparams);
        }
        catch (FormatException ex)
        {
            _rule = null;
            if (changed || previous is not null)
            {
                logger.LogError("bot feed: champion {Version} records an invalid eligibility rule ({Message}); no token is scored until it changes",
                    version, ex.Message);
            }

            return;
        }

        if (!changed && _rule == previous)
        {
            return;
        }

        if (_rule.Enabled)
        {
            logger.LogInformation("bot feed: champion {Version} was trained on eligible tokens only ({Rule}); only those are scored",
                version, _rule);
        }
        else
        {
            logger.LogWarning("bot feed: champion {Version} records no eligibility rule (trained on every token); every token is scored",
                version);
        }
    }

    /// <summary>
    /// What the rule reads of <paramref name="mint"/> as of its row at <paramref name="upTo"/>: the running maxima of
    /// its <c>unique_traders</c>, <c>volume</c> and <c>token_age_sec</c> over its rows up to that one (one snapshot
    /// per token per instant, so <c>ts &lt;= upTo</c> is exactly "this row and the earlier ones").
    /// </summary>
    public static async Task<EligibilityInputs> LoadInputsAsync(SoliaDbContext db, string mint, DateTimeOffset upTo,
        CancellationToken cancellationToken)
    {
        var stats = await db.Features.AsNoTracking()
            .Where(f => f.Mint == mint && f.Ts <= upTo)
            .GroupBy(f => f.Mint)
            .Select(g => new
            {
                Traders = g.Max(f => f.UniqueTraders),
                Volume = g.Max(f => f.Volume),
                Age = g.Max(f => f.TokenAgeSec),
            })
            .FirstOrDefaultAsync(cancellationToken);
        return stats is null
            ? default
            : new EligibilityInputs(stats.Traders, (double?)stats.Volume, stats.Age);
    }
}
