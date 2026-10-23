using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Solia.Api.Health;
using Solia.Api.Logging;
using Solia.Api.Tests.Endpoints;
using Solia.Api.Tests.EndToEnd;
using Solia.Core.Ai;
using Solia.Core.Data;

namespace Solia.Api.Tests;

/// <summary>The checks, their once-per-transition logging and the response sanitizing, without a host.</summary>
public class HealthChecksTests
{
    private readonly CapturingLoggerProvider _logs = new();

    private HealthTransitionLog TransitionLog() =>
        new(LoggerFactory.Create(b => b.AddProvider(_logs).SetMinimumLevel(LogLevel.Trace)).CreateLogger<HealthTransitionLog>());

    private static HealthCheckContext Context(IHealthCheck check) => new()
    {
        Registration = new HealthCheckRegistration("check", check, HealthStatus.Unhealthy, ["ready"]),
    };

    [Fact]
    public async Task A_database_check_cut_short_by_its_timeout_reports_unreachable_instead_of_throwing()
    {
        var services = new ServiceCollection()
            .AddDbContext<SoliaDbContext>(o => o.UseNpgsql("Host=127.0.0.1;Port=1;Database=solia;Username=solia"))
            .BuildServiceProvider();
        var check = new DatabaseHealthCheck(services, TransitionLog());

        HealthCheckResult result = await check.CheckHealthAsync(Context(check), new CancellationToken(canceled: true));

        Assert.Equal(HealthStatus.Unhealthy, result.Status);
        Assert.Equal("database unreachable", result.Description);
        Assert.Null(result.Exception); // the cause is logged, never carried to the response
    }

    [Theory]
    [MemberData(nameof(AiFailures))]
    public async Task Any_ai_service_failure_reports_unreachable(Exception failure)
    {
        var check = new AiServiceHealthCheck(new StubAiClient { Health = () => throw failure }, TransitionLog());

        HealthCheckResult result = await check.CheckHealthAsync(Context(check));

        Assert.Equal(HealthStatus.Unhealthy, result.Status);
        Assert.Equal("ai-service unreachable", result.Description);
        Assert.Null(result.Exception);
    }

    public static TheoryData<Exception> AiFailures() =>
    [
        StubAiClient.Unreachable(),
        new OperationCanceledException("the check's timeout fired"),
        new InvalidOperationException("Host=10.0.0.7 unexpected"),
    ];

    [Fact]
    public async Task A_failing_check_logs_once_when_it_starts_failing_and_once_when_it_recovers()
    {
        var ai = new StubAiClient { Health = () => throw StubAiClient.Unreachable() };
        var check = new AiServiceHealthCheck(ai, TransitionLog());

        for (int i = 0; i < 5; i++)
        {
            await check.CheckHealthAsync(Context(check));
        }

        string failing = Assert.Single(_logs.Entries);
        Assert.StartsWith("Warning", failing, StringComparison.Ordinal);
        Assert.Contains("ai_service is failing: ai-service unreachable", failing, StringComparison.Ordinal);
        Assert.Contains(nameof(AiServiceException), failing, StringComparison.Ordinal); // with the cause

        ai.Health = () => new AiHealthResponse(AiHealthResponse.Ok);
        await check.CheckHealthAsync(Context(check));
        await check.CheckHealthAsync(Context(check));

        Assert.Equal(2, _logs.Entries.Count);
        Assert.Contains(_logs.Entries, e => e.StartsWith("Information", StringComparison.Ordinal) && e.Contains("ai_service recovered"));
    }

    [Fact]
    public void A_check_that_starts_healthy_logs_nothing()
    {
        HealthTransitionLog log = TransitionLog();

        log.Healthy("database");
        log.Healthy("database");

        Assert.Empty(_logs.Entries);
    }

    [Fact]
    public void Each_check_has_its_own_state()
    {
        HealthTransitionLog log = TransitionLog();

        log.Unhealthy("database", "database unreachable", cause: null);
        log.Unhealthy("ai_service", "ai-service unreachable", cause: null);
        log.Unhealthy("database", "database unreachable", cause: null);

        Assert.Equal(2, _logs.Entries.Count);
    }

    [Theory]
    [InlineData("database unreachable", "database unreachable")]
    [InlineData("ai-service reports it is not healthy", "ai-service reports it is not healthy")]
    [InlineData("A timeout occurred while running check.", "check failed")]
    [InlineData("Failed to connect to 10.0.0.7:5432 (Password=hunter2)", "check failed")]
    [InlineData(null, null)]
    public void The_response_carries_only_fixed_descriptions(string? description, string? expected)
    {
        var report = new HealthReport(
            new Dictionary<string, HealthReportEntry>
            {
                ["database"] = new(HealthStatus.Unhealthy, description, TimeSpan.Zero, new InvalidOperationException(description), null),
            },
            TimeSpan.Zero);

        HealthResponse response = HealthEndpoints.ToResponse(report);

        Assert.Equal(expected, response.Entries["database"].Description);
    }

    [Fact]
    public void The_health_check_service_s_per_probe_logs_stay_out_of_the_logs_table()
    {
        var services = new ServiceCollection().AddSoliaHealthChecks().BuildServiceProvider();

        LoggerFilterRule rule = Assert.Single(services.GetRequiredService<IOptions<LoggerFilterOptions>>().Value.Rules,
            r => r.CategoryName == HealthChecksServiceCollectionExtensions.HealthCheckServiceCategory);

        Assert.Equal(typeof(PostgresLoggerProvider).FullName, rule.ProviderName);
        Assert.Equal(LogLevel.None, rule.LogLevel);
    }
}
