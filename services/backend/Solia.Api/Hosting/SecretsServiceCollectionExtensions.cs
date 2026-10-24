using Microsoft.Extensions.DependencyInjection.Extensions;
using Npgsql;
using Solia.Api.Logging;
using Solia.Bot.Logging;
using Solia.Secrets;
using Solia.Shared.Configuration;

namespace Solia.Api.Hosting;

/// <summary>
/// Wires the AES secrets file (docs/SECRETS.md) into the backend: the decrypted <see cref="ISecretsProvider"/>,
/// the <see cref="SecretRedactor"/> the logging pipeline scrubs with, the shared
/// <see cref="NpgsqlDataSource"/> whose connection string takes its credentials from the secrets, and the
/// bot's `logs` sink (which bypasses <c>ILogger</c>, so it is scrubbed on its own).
/// </summary>
public static class SecretsServiceCollectionExtensions
{
    /// <summary>
    /// Key of a second, <em>unpooled</em> <see cref="NpgsqlDataSource"/> over the same database, for connections
    /// held for the life of the process (the live hub's LISTEN): they never take a slot from the shared pool the
    /// bot, EF and the log sink write through, and closing one really closes it (its LISTEN goes with it).
    /// </summary>
    public const string UnpooledDataSourceKey = "unpooled";

    /// <summary>
    /// Registers the secrets provider (decrypted once, on first resolve — the logging pipeline resolves it while
    /// the host is built, so a missing key, wrong key or tampered file fails startup) and the secrets-aware data
    /// source. Call it before anything else registers the data source or the bot's log sink
    /// (<c>AddSoliaDbContext</c>, the bot host, the log sink): those use <c>TryAdd</c>, so this registration wins.
    /// </summary>
    public static IServiceCollection AddSoliaSecrets(this IServiceCollection services, IConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(configuration);

        services.TryAddSingleton<ISecretsProvider>(_ => SecretsProvider.Load(key => configuration[key]));
        services.TryAddSingleton(sp =>
        {
            ISecretsProvider secrets = sp.GetRequiredService<ISecretsProvider>();
            return SecretRedactor.For(secrets, [configuration[MasterKey.EnvironmentKey], .. DerivedValues(secrets)]);
        });
        services.TryAddSingleton(sp => NpgsqlDataSource.Create(ConnectionString(sp, configuration)));
        services.TryAddKeyedSingleton(UnpooledDataSourceKey, (sp, _) => NpgsqlDataSource.Create(
            new NpgsqlConnectionStringBuilder(ConnectionString(sp, configuration)) { Pooling = false }.ConnectionString));
        services.TryAddSingleton<IBotLogSink>(sp => new RedactingBotLogSink(
            new PostgresBotLogSink(sp.GetRequiredService<NpgsqlDataSource>()), sp.GetRequiredService<SecretRedactor>()));
        return services;
    }

    private static string ConnectionString(IServiceProvider services, IConfiguration configuration) =>
        ResolveConnectionString(DbConnectionSettings.Resolve(key => configuration[key]), services.GetRequiredService<ISecretsProvider>());

    /// <summary>
    /// The sensitive parts of composite secrets, redacted on their own: the password inside a
    /// <c>db_connection</c> secret, and that string as Npgsql re-serialises it (another key order/casing), so
    /// neither the bare password nor a normalised copy of the string slips past a whole-value match. A
    /// malformed <c>db_connection</c> yields nothing here — building the data source fails on it anyway.
    /// </summary>
    public static IEnumerable<string?> DerivedValues(ISecretsProvider secrets)
    {
        ArgumentNullException.ThrowIfNull(secrets);
        if (!secrets.TryGet(SecretNames.DbConnection, out string? connection))
        {
            return [];
        }
        try
        {
            var builder = new NpgsqlConnectionStringBuilder(connection);
            return [builder.Password, builder.ConnectionString];
        }
        catch (ArgumentException)
        {
            return [];
        }
    }

    /// <summary>
    /// Overlays the secrets on the plaintext-configured connection string: a <c>db_connection</c> secret
    /// replaces it outright; otherwise a <c>db_password</c> secret replaces its password. With neither, the
    /// configured string is used as is.
    /// </summary>
    public static string ResolveConnectionString(string configured, ISecretsProvider secrets)
    {
        ArgumentNullException.ThrowIfNull(secrets);

        if (secrets.TryGet(SecretNames.DbConnection, out string? connection))
        {
            return connection;
        }
        if (secrets.TryGet(SecretNames.DbPassword, out string? password))
        {
            return new NpgsqlConnectionStringBuilder(configured) { Password = password }.ConnectionString;
        }
        return configured;
    }
}
