using Microsoft.Extensions.DependencyInjection.Extensions;
using Solia.Secrets;

namespace Solia.Api.Logging;

/// <summary>Wires secret redaction into the host's logging (docs/SECRETS.md rule 2).</summary>
public static class SecretRedactionLoggingBuilderExtensions
{
    /// <summary>
    /// Replaces the <see cref="ILoggerFactory"/> with a <see cref="RedactingLoggerFactory"/> over the stock
    /// <see cref="LoggerFactory"/>, so no provider ever receives a known secret value. Needs a
    /// <see cref="SecretRedactor"/> in the container (see <c>AddSoliaSecrets</c>).
    /// </summary>
    public static ILoggingBuilder AddSecretRedaction(this ILoggingBuilder builder)
    {
        // The stock factory, built exactly as the default ILoggerFactory registration would build it.
        builder.Services.TryAddSingleton<LoggerFactory>();
        builder.Services.Replace(ServiceDescriptor.Singleton<ILoggerFactory>(sp => new RedactingLoggerFactory(
            sp.GetRequiredService<LoggerFactory>(), sp.GetRequiredService<SecretRedactor>())));
        return builder;
    }
}
