using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Npgsql;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Options;
using Solia.DataPipeline.Pipeline;
using Solia.Shared.Data;
using Solia.Shared.Logging;

namespace Solia.DataPipeline.Ingestion.Replay;

/// <summary>
/// The one-shot <c>replay</c> mode of the data-pipeline (see <see cref="ReplayCommandLine"/> for the
/// arguments): replays a historical window into PostgreSQL through the live feature path, then exits.
/// Runs in its own process — never alongside live ingestion's token state — with the same
/// configuration (Pipeline / PumpApi sections, flat .env keys, DB connection) as the live host:
/// same environment resolution (<c>ASPNETCORE_ENVIRONMENT</c> or <c>DOTNET_ENVIRONMENT</c>, so the same
/// <c>appsettings.{Environment}.json</c>) and command-line configuration overrides. The run summary is
/// also written to the <c>logs</c> table (best-effort), like the live host's startup entry.
/// <para>Exit codes: <see cref="ExitOk"/>, <see cref="ExitFatal"/>, <see cref="ExitUsage"/>, <see cref="ExitIncomplete"/>.</para>
/// </summary>
public static class ReplayCommand
{
    public const string Verb = "replay";

    /// <summary>Every hour read (or absent at the source) and every batch written.</summary>
    public const int ExitOk = 0;

    /// <summary>Fatal error: unreachable database, source failure, interruption.</summary>
    public const int ExitFatal = 1;

    /// <summary>Invalid arguments or configuration; nothing was run.</summary>
    public const int ExitUsage = 2;

    /// <summary>Ran to the end, but some hours or write batches were given up on: re-run the window to retry.</summary>
    public const int ExitIncomplete = 3;

    public static bool IsRequested(string[] args) =>
        args.Length > 0 && string.Equals(args[0], Verb, StringComparison.OrdinalIgnoreCase);

    public static int ExitCodeFor(ReplaySummary summary) => summary.IsComplete ? ExitOk : ExitIncomplete;

    /// <summary>Runs the command; <paramref name="args"/> are the arguments after the verb.</summary>
    public static async Task<int> RunAsync(string[] args)
    {
        ReplayCommandLine? command = ReplayCommandLine.TryParse(args, DateTimeOffset.UtcNow, out string? error);
        if (command is null)
        {
            Console.Error.WriteLine($"replay: {error}");
            Console.Error.WriteLine(ReplayCommandLine.Usage);
            return ExitUsage;
        }

        if (command.MirrorDirectory is { } dir && ArchiveMirror.CheckDirectory(dir) is { } mirrorError)
        {
            Console.Error.WriteLine($"replay: {mirrorError}");
            return ExitUsage;
        }

        HostApplicationBuilder builder = CreateHostBuilder(command.ConfigArgs);
        builder.Configuration.AddFlatEnvAliases();
        builder.Services.AddPipelineOptions(builder.Configuration);
        builder.Services.AddSoliaDataSource(builder.Configuration);
        builder.Services.AddPostgresLogSink(builder.Configuration, LogService);

        using IHost host = builder.Build();
        try
        {
            await host.StartAsync(); // validates options (ValidateOnStart); no hosted services to run
        }
        catch (OptionsValidationException ex)
        {
            Console.Error.WriteLine($"replay: invalid configuration: {string.Join("; ", ex.Failures)}");
            return ExitUsage;
        }

        ILoggerFactory loggerFactory = host.Services.GetRequiredService<ILoggerFactory>();
        ILogger logger = loggerFactory.CreateLogger(typeof(ReplayCommand));
        CancellationToken stopping = host.Services.GetRequiredService<IHostApplicationLifetime>().ApplicationStopping;

        try
        {
            ReplaySummary summary = await ExecuteAsync(command, host.Services, loggerFactory, stopping);
            Log(logger, command, summary);
            if (!command.MirrorOnly) // mirror-only needs no database at all
            {
                await WriteSummaryAsync(host.Services, logger, command, summary);
            }

            return ExitCodeFor(summary);
        }
        catch (OperationCanceledException) when (stopping.IsCancellationRequested)
        {
            logger.LogError("replay interrupted; re-run the same window to complete it (re-runs insert no duplicates)");
            return ExitFatal;
        }
        catch (Exception ex)
        {
            logger.LogCritical(ex, "replay failed");
            return ExitFatal;
        }
        finally
        {
            await host.StopAsync(CancellationToken.None);
        }
    }

    /// <summary>The <c>logs.service</c> value: the replay is a mode of the data-pipeline service.</summary>
    internal const string LogService = "data-pipeline";

    /// <summary>
    /// A host builder resolving its environment like the live (web) host — <c>ASPNETCORE_ENVIRONMENT</c>
    /// as well as <c>DOTNET_ENVIRONMENT</c> — with <paramref name="configArgs"/> as command-line configuration.
    /// </summary>
    internal static HostApplicationBuilder CreateHostBuilder(IReadOnlyList<string> configArgs)
    {
        var configuration = new ConfigurationManager();
        configuration.AddEnvironmentVariables(prefix: "ASPNETCORE_");
        return Host.CreateApplicationBuilder(new HostApplicationBuilderSettings
        {
            Args = configArgs.ToArray(),
            Configuration = configuration,
        });
    }

    private static async Task<ReplaySummary> ExecuteAsync(
        ReplayCommandLine command, IServiceProvider services, ILoggerFactory loggerFactory, CancellationToken cancellationToken)
    {
        IOptions<PumpApiOptions> pumpApi = services.GetRequiredService<IOptions<PumpApiOptions>>();
        PipelineOptions pipeline = services.GetRequiredService<IOptions<PipelineOptions>>().Value;
        ILogger logger = loggerFactory.CreateLogger(typeof(ReplayCommand));

        // Default timeout bounds the wait for response headers; body reads are bounded by the
        // opener's idle timeout (a stalled download fails the attempt instead of hanging).
        using var http = new HttpClient();
        var download = new HttpReplayArchiveOpener(http, pumpApi);
        var runner = new ReplayRunner(loggerFactory);

        ArchiveMirror? mirror = null;
        if (command.MirrorDirectory is { } dir)
        {
            mirror = new ArchiveMirror(dir, download.OpenAsync, loggerFactory.CreateLogger<ArchiveMirror>());
            int absent = mirror.CountAbsent(command.Window.Hours());
            if (absent > 0)
            {
                logger.LogWarning("replay: {Warning}", ArchiveMirror.DiskUseWarning(dir, absent));
            }
            else
            {
                logger.LogInformation("replay: every hour of the window is already in the mirror; nothing to download");
            }
        }

        logger.LogInformation(
            "replay {Mode} {From:O} .. {To:O} ({Hours} h) from {Source}{Rebuild}",
            command.MirrorOnly ? "(mirror only)" : "", command.Window.StartUtc, command.Window.EndUtc,
            command.Window.Hours().Count(), mirror?.Root ?? pumpApi.Value.ReplayBaseUrl,
            command.Rebuild ? " — rebuild: existing features rows are refreshed" : "");

        if (command.MirrorOnly)
        {
            return await runner.MirrorOnlyAsync(command.Window, mirror!, cancellationToken);
        }

        var sink = new PostgresTokenSink(services.GetRequiredService<NpgsqlDataSource>(), command.Rebuild);
        return await runner.ReplayAsync(command.Window, mirror ?? (IReplayArchive)download,
            sink, () => sink.FeatureRowsWritten, pipeline, cancellationToken);
    }

    /// <summary>Mirrors the summary to the <c>logs</c> table; best-effort, like the live host's startup entry.</summary>
    private static async Task WriteSummaryAsync(IServiceProvider services, ILogger logger, ReplayCommandLine command, ReplaySummary s)
    {
        try
        {
            await services.GetRequiredService<ILogSink>().WriteAsync(s.IsComplete ? LogSeverity.Info : LogSeverity.Warn,
                s.IsComplete ? "replay completed" : "replay incomplete: re-run the same window to retry", new
                {
                    From = command.Window.StartUtc,
                    To = command.Window.EndUtc,
                    command.MirrorOnly,
                    command.Rebuild,
                    s.Hours,
                    s.ProcessedHours,
                    MissingHours = s.MissingHours.Count,
                    FailedHours = s.FailedHours.Count,
                    s.FeatureRowsWritten,
                    s.FailedBatches,
                });
        }
        catch (Exception ex)
        {
            logger.LogWarning(ex, "could not write the replay summary to the database");
        }
    }

    private static void Log(ILogger logger, ReplayCommandLine command, ReplaySummary s)
    {
        logger.Log(s.IsComplete ? LogLevel.Information : LogLevel.Warning,
            "replay summary: hours={Hours} processed={Processed} missing={Missing} failed={Failed} " +
            "featureRowsWritten={Rows} failedBatches={FailedBatches} mirrorHits={Hits} mirrorDownloads={Downloads}{Hint}",
            s.Hours, s.ProcessedHours, s.MissingHours.Count, s.FailedHours.Count,
            s.FeatureRowsWritten, s.FailedBatches, s.MirrorHits, s.MirrorDownloads,
            s.IsComplete ? "" : " — incomplete: re-run the same window to retry");

        foreach (DateTimeOffset hour in s.MissingHours)
        {
            logger.LogInformation("  missing: {Hour:yyyy-MM-dd HH}:00 UTC", hour);
        }

        foreach (ReplayHourFailure failure in s.FailedHours)
        {
            logger.LogWarning("  failed: {Hour:yyyy-MM-dd HH}:00 UTC — {Error}", failure.Hour, failure.Error);
        }

        if (command.Rebuild && !command.MirrorOnly)
        {
            logger.LogInformation("  rebuild: only tokens created inside the window were refreshed (others keep their stored " +
                                  "rows), and only where the replay recomputes the same (mint, ts): use the same window " +
                                  "start and Pipeline:SnapshotCadenceSeconds as the original run");
        }
    }
}
