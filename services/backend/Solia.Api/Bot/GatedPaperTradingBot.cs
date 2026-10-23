using Microsoft.Extensions.Options;
using Solia.Bot;

namespace Solia.Api.Bot;

/// <summary>
/// The hosted service that runs <see cref="PaperTradingBot"/> inside the API host when
/// <see cref="BotHostOptions.Enabled"/> is set, its start/stop tied to the host lifetime. The switch is read
/// when the host starts (not when services are registered), so every configuration source counts; while it is
/// off the bot — and its feed — is never even built. A bot failure faults this service, which stops the host
/// (fail loud, as the bot itself does).
/// </summary>
public sealed class GatedPaperTradingBot(
    IServiceProvider services, IOptions<BotHostOptions> options, ILogger<GatedPaperTradingBot> logger) : BackgroundService
{
    private PaperTradingBot? _bot;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        // Building the bot (its feed, stores) doesn't hold up the host's startup.
        await Task.Yield();
        if (!options.Value.Enabled)
        {
            logger.LogInformation("paper-trading bot disabled (Bot:Enabled / PAPER_BOT_ENABLED)");
            return;
        }

        PaperTradingBot bot = services.GetRequiredService<PaperTradingBot>();
        Volatile.Write(ref _bot, bot);
        if (stoppingToken.IsCancellationRequested)
        {
            return; // stopped while it was being built: never started
        }

        try
        {
            // The bot's stopping token is linked to ours: stopping this service stops it.
            await bot.StartAsync(stoppingToken);
            await (bot.ExecuteTask ?? Task.CompletedTask);
        }
        catch (OperationCanceledException) when (stoppingToken.IsCancellationRequested)
        {
            // Stopped as it was starting.
        }
    }

    /// <summary>
    /// Cancels and waits for <see cref="ExecuteAsync"/>, which only returns once the bot's run has ended —
    /// whether or not the bot had been built when the stop came — then releases the bot.
    /// </summary>
    public override async Task StopAsync(CancellationToken cancellationToken)
    {
        await base.StopAsync(cancellationToken);
        if (Volatile.Read(ref _bot) is PaperTradingBot bot)
        {
            await bot.StopAsync(cancellationToken);
        }
    }
}
