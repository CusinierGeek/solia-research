using Solia.Core.Data.Entities;

namespace Solia.Api.Tests.Endpoints;

/// <summary>The trades-since-last-train reset rule and the manual retrain trigger, as backend constants.</summary>
public class TrainingEventValuesTests
{
    [Fact]
    public void Only_a_run_that_trained_a_model_resets_the_trade_count()
    {
        Assert.Equal(
            [TrainingDecisions.Promoted, TrainingDecisions.Rejected],
            TrainingDecisions.ResetTradeCount.Order(StringComparer.Ordinal));
        Assert.DoesNotContain(TrainingDecisions.Failed, TrainingDecisions.ResetTradeCount);
        Assert.DoesNotContain(TrainingDecisions.Rollback, TrainingDecisions.ResetTradeCount);
    }

    [Fact]
    public void The_manual_retrain_request_uses_the_manual_trigger() =>
        Assert.Contains(Solia.Core.Ai.AiRetrainRequest.Manual, TrainingTriggers.All);
}

/// <summary>
/// The backend's <c>training_events</c> values mirror the CHECKs the migrated database actually enforces (read from
/// <c>pg_constraint</c>, so a later migration that drops or re-adds them is followed). Run in CI.
/// </summary>
[Trait(TestCategories.Key, TestCategories.Integration)]
public class TrainingEventSchemaParityTests(PostgresFixture db) : IClassFixture<PostgresFixture>
{
    /// <summary>The quoted values of a CHECK constraint on <c>training_events</c>, as PostgreSQL renders it.</summary>
    private async Task<string[]> CheckValuesAsync(string constraint)
    {
        string definition = await db.ScalarAsync<string>(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint " +
            "WHERE conrelid = 'training_events'::regclass AND conname = @name",
            ("name", constraint));
        return [.. System.Text.RegularExpressions.Regex.Matches(definition, "'([^']*)'")
            .Select(m => m.Groups[1].Value).Order(StringComparer.Ordinal)];
    }

    [Fact]
    public async Task Triggers_are_the_training_events_check() =>
        Assert.Equal(await CheckValuesAsync("training_events_trigger_check"),
            TrainingTriggers.All.Order(StringComparer.Ordinal));

    [Fact]
    public async Task Decisions_are_the_training_events_check() =>
        Assert.Equal(await CheckValuesAsync("training_events_decision_check"),
            TrainingDecisions.All.Order(StringComparer.Ordinal));

    [Fact]
    public async Task Decision_and_finished_at_are_set_together() =>
        Assert.Equal("CHECK (((decision IS NULL) = (finished_at IS NULL)))", await db.ScalarAsync<string>(
            "SELECT pg_get_constraintdef(oid) FROM pg_constraint " +
            "WHERE conrelid = 'training_events'::regclass AND conname = 'training_events_finished_decision_check'"));
}
