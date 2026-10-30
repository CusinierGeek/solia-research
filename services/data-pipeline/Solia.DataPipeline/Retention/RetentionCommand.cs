using Microsoft.Extensions.Options;
using Solia.DataPipeline.Config;
using Solia.DataPipeline.Ingestion.Replay;
using Solia.DataPipeline.Options;
using Solia.Shared.Data;
using Solia.Shared.Logging;

namespace Solia.DataPipeline.Retention;

/// <summary>
/// The one-shot <c>retention [--dry-run] [--Section:Key=value ...]</c> mode: one purge with the configured TTLs, then
/// exit. <c>--dry-run</c> only reports what would be deleted. Runs whether or not <c>Retention:Enabled</c> is set
/// (that switch only governs the periodic purge). Same configuration resolution as the live host and the replay.
/// <para>Exit codes: <see cref="ExitOk"/>, <see cref="ExitFailed"/>, <see cref="ExitUsage"/>.</para>
/// </summary>
public static class RetentionCommand
{
    public const string Verb = "retention";

    public const int ExitOk = 0;

    /// <summary>A table could not be purged (or counted): see the log; re-run to retry.</summary>
    public const int ExitFailed = 1;

    /// <summary>Invalid arguments or configuration; nothing was run.</summary>
    public const int ExitUsage = 2;

    public const string Usage =
        "usage: retention [--dry-run] [--Section:Key=value ...]\n" +
        "  --dry-run   report what would be deleted per table; change nothing\n" +
        "  --Section:Key=value, --environment <name>   configuration overrides (e.g. --Retention:FeaturesTtlDays=7)";

    public static bool IsRequested(string[] args) =>
        args.Length > 0 && string.Equals(args[0], Verb, StringComparison.OrdinalIgnoreCase);

    /// <summary>Parses the arguments after the verb; null with an error message on invalid input.</summary>
    public static (bool DryRun, List<string> ConfigArgs)? TryParse(IReadOnlyList<string> args, out string? error)
    {
        bool dryRun = false;
        var configArgs = new List<string>();
        for (int i = 0; i < args.Count; i++)
        {
            string arg = args[i];
            if (arg == "--dry-run")
            {
                dryRun = true;
            }
            else if (ReplayCommandLine.IsConfigOverride(arg))
            {
                configArgs.Add(arg);
                if (!arg.Contains('=', StringComparison.Ordinal))
                {
                    if (i + 1 >= args.Count || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                    {
                        error = $"{arg} needs a value.";
                        return null;
                    }

                    configArgs.Add(args[++i]);
                }
            }
            else
            {
                error = $"unknown argument '{arg}'.";
                return null;
            }
        }

        error = null;
        return (dryRun, configArgs);
    }

    public static int ExitCodeFor(RetentionReport report) => report.Failed ? ExitFailed : ExitOk;

    /// <summary>Runs the command; <paramref name="args"/> are the arguments after the verb.</summary>
    public static async Task<int> RunAsync(string[] args)
    {
        if (TryParse(args, out string? error) is not { } command)
        {
            Console.Error.WriteLine($"retention: {error}");
            Console.Error.WriteLine(Usage);
            return ExitUsage;
        }

        HostApplicationBuilder builder = ReplayCommand.CreateHostBuilder(command.ConfigArgs);
        builder.Configuration.AddFlatEnvAliases();
        builder.Services.AddPipelineOptions(builder.Configuration);
        builder.Services.AddSoliaDataSource(builder.Configuration);
        builder.Services.AddPostgresLogSink(builder.Configuration, ReplayCommand.LogService);
        builder.Services.AddRetention(hosted: false);

        using IHost host = builder.Build();
        try
        {
            await host.StartAsync(); // validates options (ValidateOnStart); no hosted services to run
        }
        catch (OptionsValidationException ex)
        {
            Console.Error.WriteLine($"retention: invalid configuration: {string.Join("; ", ex.Failures)}");
            return ExitUsage;
        }

        CancellationToken stopping = host.Services.GetRequiredService<IHostApplicationLifetime>().ApplicationStopping;
        try
        {
            RetentionReport report = await host.Services.GetRequiredService<RetentionJob>().RunAsync(command.DryRun, stopping);
            return ExitCodeFor(report);
        }
        catch (OperationCanceledException) when (stopping.IsCancellationRequested)
        {
            Console.Error.WriteLine("retention: interrupted; every finished batch is kept, re-run to complete");
            return ExitFailed;
        }
        finally
        {
            await host.StopAsync(CancellationToken.None);
        }
    }
}
